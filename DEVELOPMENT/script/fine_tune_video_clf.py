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
from torchvision.transforms import Compose, Lambda

from datasets import Dataset
from transformers import (
    VideoMAEForVideoClassification,
    AutoImageProcessor,
    TrainingArguments,
)

from src.utils import RandomShortSideScale, Normalize, UniformTemporalSubsample
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


def create_dataset(data_dir: str):
    """Load .npy videos and labels into HuggingFace Datasets."""
    logger.info("Loading dataset from '%s' …", data_dir)
    train_paths, train_labels = [], []
    for class_name in sorted(os.listdir(os.path.join(data_dir, "train"))):
        class_dir = os.path.join(data_dir, "train", class_name)
        if not os.path.isdir(class_dir):
            continue
        for p in glob.glob(os.path.join(class_dir, "*.npy")):
            train_paths.append(p)
            train_labels.append(class_name)

    val_paths, val_labels = [], []
    for class_name in sorted(os.listdir(os.path.join(data_dir, "val"))):
        class_dir = os.path.join(data_dir, "val", class_name)
        if not os.path.isdir(class_dir):
            continue
        for p in glob.glob(os.path.join(class_dir, "*.npy")):
            val_paths.append(p)
            val_labels.append(class_name)

    logger.info("Found %d training clips in %d classes", len(train_paths), len(set(train_labels)))
    logger.info("Found %d validation clips in %d classes", len(val_paths), len(set(val_labels)))

    # load into memory
    train_arrays = [np.load(p) for p in train_paths]
    val_arrays = [np.load(p) for p in val_paths]

    # build datasets
    train_ds = Dataset.from_dict({"video": train_arrays, "label": train_labels})
    val_ds = Dataset.from_dict({"video": val_arrays, "label": val_labels})

    train_ds = train_ds.class_encode_column("label")
    val_ds = val_ds.class_encode_column("label")

    unique_labels = sorted(set(train_labels))
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    return train_ds, val_ds, unique_labels, label2id, id2label


def prepare_model(unique_labels, label2id, id2label):
    """Instantiate processor and model from pretrained checkpoint."""
    ckpt = "MCG-NJU/videomae-base"
    logger.info("Loading model and processor from '%s'", ckpt)
    processor = AutoImageProcessor.from_pretrained(ckpt, use_fast=True)
    model = VideoMAEForVideoClassification.from_pretrained(
        ckpt, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )
    return model, processor


def prepare_transforms(processor, model):
    """Build train/val transform pipelines."""
    mean, std = processor.image_mean, processor.image_std
    num_frames = model.config.num_frames
    size = processor.size.get("shortest_edge", processor.size.get("height"))
    resize_to = (size, size)

    train_transform = Compose([
        UniformTemporalSubsample(num_frames),
        RandomShortSideScale(256, 320),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        Lambda(lambda x: F.interpolate(
            x, size=resize_to, mode="bilinear", align_corners=False
        )),
    ])

    val_transform = Compose([
        UniformTemporalSubsample(num_frames),
        Lambda(lambda x: x / 255.0),
        Normalize(mean, std),
        Lambda(lambda x: F.interpolate(
            x, size=resize_to, mode="bilinear", align_corners=False
        )),
    ])

    return train_transform, val_transform


def run():
    args = parse_args()
    logger.info("Arguments: %s", args)

    train_ds, val_ds, unique_labels, label2id, id2label = create_dataset(args.data_dir)
    model, processor = prepare_model(unique_labels, label2id, id2label)
    train_tf, val_tf = prepare_transforms(processor, model)

    def preprocess(batch, tf):
        pixel_values = []
        for arr in batch["video"]:
            vid = torch.as_tensor(arr, dtype=torch.float32)
            pixel_values.append(tf(vid))
        return {"pixel_values": pixel_values, "labels": batch["label"]}

    logger.info("Applying transforms to training set")
    train_ds = train_ds.map(
        lambda b: preprocess(b, train_tf),
        batched=True, batch_size=args.batch_size,
        remove_columns=["video", "label"]
    )
    train_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    logger.info("Applying transforms to validation set")
    val_ds = val_ds.map(
        lambda b: preprocess(b, val_tf),
        batched=True, batch_size=args.batch_size,
        remove_columns=["video", "label"]
    )
    val_ds.set_format(type="torch", columns=["pixel_values", "labels"])

    # prepare training arguments
    today = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(args.output_dir, f"{today}-finetuned")
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
