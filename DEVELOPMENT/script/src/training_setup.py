from transformers import Trainer, TrainerCallback
import evaluate
import numpy as np

accuracy  = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall    = evaluate.load("recall")
f1        = evaluate.load("f1")
confusion = evaluate.load("confusion_matrix")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # scalar metrics as before…
    acc   = accuracy.compute(predictions=preds, references=labels)["accuracy"]
    prec  = precision.compute(predictions=preds, references=labels, average="macro")["precision"]
    rec   = recall.compute(predictions=preds, references=labels, average="macro")["recall"]
    f1sc  = f1.compute(predictions=preds, references=labels, average="macro")["f1"]

    # get confusion matrix and turn it into a nested Python list
    cm = confusion.compute(predictions=preds, references=labels)["confusion_matrix"]
    cm_list = cm.tolist()

    return {
        "accuracy":          acc,
        "precision":         prec,
        "recall":            rec,
        "f1":                f1sc,
        "confusion_matrix":  cm_list,    # now JSON‑serializable
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # these will accumulate _all_ batches in the current epoch
        self.epoch_losses     = []
        self.epoch_preds      = []
        self.epoch_labels     = []

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides Trainer.compute_loss to store batch‐level loss & preds/labels.
        """
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        loss   = outputs.loss
        logits = outputs.logits

        if labels is not None:
            # 1) store the loss
            self.epoch_losses.append(loss.item())
            # 2) store predictions + labels as 1D arrays
            preds = logits.argmax(dim=-1).detach().cpu().numpy()
            labs  = labels.detach().cpu().numpy()
            self.epoch_preds .extend(preds.tolist())
            self.epoch_labels.extend(labs.tolist())

        return (loss, outputs) if return_outputs else loss

class MetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

        # lists to hold epoch‐by‐epoch values
        self.train_losses     = []
        self.train_accuracies = []
        self.eval_losses      = []
        self.eval_accuracies  = []
        self.eval_confusion_matrices = [] 

    def on_epoch_end(self, args, state, control, **kwargs):
        # Compute average training loss & accuracy for the epoch
        t = self.trainer
        avg_loss = float(np.mean(t.epoch_losses))
        acc      = np.mean(
            np.array(t.epoch_preds) == np.array(t.epoch_labels)
        )

        # Store & clear for next epoch
        self.train_losses    .append(avg_loss)
        self.train_accuracies.append(acc)
        t.epoch_losses .clear()
        t.epoch_preds  .clear()
        t.epoch_labels .clear()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # metrics come prefixed with "eval_"
        self.eval_losses   .append(metrics["eval_loss"])
        self.eval_accuracies.append(metrics["eval_accuracy"])
        self.eval_confusion_matrices.append(metrics["eval_confusion_matrix"])