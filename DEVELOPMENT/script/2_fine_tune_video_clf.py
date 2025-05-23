#!/usr/bin/env python3
import os
import glob
import logging
import pickle
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import einops
from torchvision.transforms import Compose, Lambda

from datasets import Dataset, Features, Value, ClassLabel
from transformers import (
    VideoMAEForVideoClassification,
    AutoImageProcessor,
    TrainingArguments,
)

from src.utils import Normalize
from src.training_setup import compute_metrics, CustomTrainer, MetricsCallback

# ──────────────────────────────────────────────────────────────────────────────
# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Fine-tune VideoMAE for video classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data/video/rgb',
        help="Root directory containing 'train' and 'val' subfolders",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for both train and validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/video",
        help="Base directory for saving checkpoints and logs",
    )
    return parser.parse_args()


def create_dataset(base_dir: str):
    """Load video data and labels into HuggingFace Datasets."""
    logger.info(f"Loading dataset from '{base_dir}'")

    train_dirs   = [os.path.join(base_dir, "train_aug")]
    val_dir      = os.path.join(base_dir, "val")
    
    # 2) Collect train file paths & labels
    train_paths, train_labels = [], []
    for d in train_dirs:
        for class_name in os.listdir(d):
            class_dir = os.path.join(d, class_name)
            if not os.path.isdir(class_dir):
                continue
            for npy_path in glob.glob(os.path.join(class_dir, "*.npy")):
                train_paths.append(npy_path)
                train_labels.append(class_name)
    
    # 3) Collect val file paths & labels
    val_paths, val_labels = [], []
    for class_name in os.listdir(val_dir):
        class_dir = os.path.join(val_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for npy_path in glob.glob(os.path.join(class_dir, "*.npy")):
            val_paths.append(npy_path)
            val_labels.append(class_name)
    
    # 4) Define the ordered class list
    ordered_labels = ["OH", "IH", "SF", "HC", "HH"]
    
    # 5) Create custom class label feature with specified order
    class_feature = ClassLabel(names=ordered_labels)
    
    # 6) Create train dataset with custom class feature
    train_ds = Dataset.from_dict({
        "image_path": train_paths,
        "label": train_labels
    }, features=Features({
        "image_path": Value("string"),
        "label": class_feature
    }))
    
    # 7) Create validation dataset with custom class feature
    val_ds = Dataset.from_dict({
        "image_path": val_paths,
        "label": val_labels
    }, features=Features({
        "image_path": Value("string"),
        "label": class_feature
    }))
    
    # 8) Build label2id / id2label with the specified order
    label2id = {label: idx for idx, label in enumerate(ordered_labels)}
    id2label = {idx: label for idx, label in enumerate(ordered_labels)}
    
    # Log dataset statistics
    logger.info("Found %d training clips in %d classes", len(train_paths), len(ordered_labels))
    logger.info("Found %d validation clips in %d classes", len(val_paths), len(ordered_labels))
    
    # Print explicit mapping for verification
    logger.info("\nClass Mapping Verification:")
    for i in range(len(id2label)):
        class_name = id2label[i]
        logger.info(f"ID {i}: {class_name}")
        
        # Count samples for this class in training set
        train_count = sum(1 for label in train_ds["label"] if label == i)
        logger.info(f"  - Training samples: {train_count}")
        
        # Count samples for this class in validation set
        val_count = sum(1 for label in val_ds["label"] if label == i)
        logger.info(f"  - Validation samples: {val_count}")
    
    return train_ds, val_ds, ordered_labels, label2id, id2label


def prepare_model(class_labels, label2id, id2label):
    """Instantiate processor and model from pretrained checkpoint."""
    ckpt = "MCG-NJU/videomae-base"
    logger.info("Loading model and processor from '%s'", ckpt)
    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
    model = VideoMAEForVideoClassification.from_pretrained(
        ckpt, num_labels=len(class_labels), id2label=id2label, label2id=label2id
    )
    return model, processor


def prepare_transforms(processor, model):
    """Build transform pipeline."""
    mean, std = processor.image_mean, processor.image_std
    height = processor.size.get("shortest_edge", processor.size.get("height"))
    width = height
    resize_to = (height, width)

    transform = Compose([
        Lambda(lambda x: x / 255.0),  # Scale to [0,1]
        Normalize(mean, std),         # Per-channel normalization
        Lambda(lambda x: F.interpolate(
            x, size=resize_to, mode="bilinear", align_corners=False
        )),  # Resize to model's expected input size
    ])

    return transform


def run():
    args = parse_args()
    logger.info("Arguments: %s", args)

    train_ds, val_ds, ordered_labels, label2id, id2label = create_dataset(
        args.data_dir
    )
    model, processor = prepare_model(ordered_labels, label2id, id2label)
    transform = prepare_transforms(processor, model)

    def preprocess_train(batch):
        pixel_values = []
        for path in batch["image_path"]:
            arr = np.load(path)
            
            # Convert numpy array to tensor
            vid = torch.as_tensor(arr, dtype=torch.float32)
            
            # Rearrange from (T, H, W, C) to (T, C, H, W) using einops
            vid = einops.rearrange(vid, 't h w c -> t c h w')
            
            # Apply transformations
            vid_t = transform(vid)
            pixel_values.append(vid_t)
        
        batch["pixel_values"] = pixel_values
        batch["labels"] = batch["label"]
        return batch

    def preprocess_val(batch):
        return preprocess_train(batch)  # Same preprocessing for validation

    logger.info("Applying transforms to training set")
    train_ds = train_ds.map(
        preprocess_train,
        batched=True, batch_size=args.batch_size // 2,
        remove_columns=["image_path", "label"]
    )
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    logger.info("Applying transforms to validation set")
    val_ds = val_ds.map(
        preprocess_val,
        batched=True, batch_size=args.batch_size // 2,
        remove_columns=["image_path", "label"]
    )
    val_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    # prepare training arguments
    today = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(args.output_dir, f"{today}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="best",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True
    )

    logger.info("Initializing Trainer")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    metrics_cb = MetricsCallback(trainer)
    trainer.add_callback(metrics_cb)

    logger.info("Starting training")
    trainer.train()

    # save training history
    hist = {
        "train_loss":     metrics_cb.train_losses,
        "train_accuracy": metrics_cb.train_accuracies,
        "eval_loss":      metrics_cb.eval_losses,
        "eval_accuracy":  metrics_cb.eval_accuracies,
        "eval_confusion_matrix": metrics_cb.eval_confusion_matrices,
    }
    hist_path = os.path.join(output_dir, "history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(hist, f)
    logger.info("Training complete, history saved to %s", hist_path)


if __name__ == "__main__":
    run()