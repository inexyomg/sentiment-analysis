import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


_DEFAULT_LABEL_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]


# ─────────────────────────────────────────────────────────────────
# Losses for class imbalance
# ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017). Down-weights easy examples."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class WeightedTrainer(Trainer):
    """
    Trainer with configurable loss for imbalanced classification.

    loss_type:
      'ce'    — cross-entropy (optionally class-weighted + label-smoothed)
      'focal' — focal loss (optionally class-weighted)
      'bce'   — binary cross-entropy with logits (multi-label, optional pos_weight)
    """

    def __init__(self, *args, loss_type="ce", class_weights=None,
                 label_smoothing=0.0, focal_gamma=2.0, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        device = logits.device

        if self.loss_type == "bce":
            pw = self.pos_weight.to(device) if self.pos_weight is not None else None
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pw)
        elif self.loss_type == "focal":
            w = self.class_weights.to(device) if self.class_weights is not None else None
            loss = FocalLoss(gamma=self.focal_gamma, weight=w)(logits, labels)
        else:  # ce
            w = self.class_weights.to(device) if self.class_weights is not None else None
            loss = F.cross_entropy(logits, labels, weight=w, label_smoothing=self.label_smoothing)

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    """Balanced class weights from training labels (single-label)."""
    classes = np.arange(num_labels)
    present = np.unique(labels)
    weights = np.ones(num_labels, dtype=np.float32)
    cw = compute_class_weight("balanced", classes=present, y=labels)
    for c, w in zip(present, cw):
        weights[c] = w
    return torch.tensor(weights, dtype=torch.float)


def compute_pos_weights(label_matrix: np.ndarray) -> torch.Tensor:
    """Per-class pos_weight = #neg / #pos for multi-label BCE."""
    pos = label_matrix.sum(axis=0)
    neg = label_matrix.shape[0] - pos
    pos = np.clip(pos, 1, None)
    return torch.tensor(neg / pos, dtype=torch.float)


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────

def _metrics_single(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }


def _metrics_multi(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
    }


def tune_thresholds(probs: np.ndarray, labels: np.ndarray,
                    grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Find per-class threshold maximizing F1 (multi-label)."""
    if grid is None:
        grid = np.arange(0.1, 0.9, 0.05)
    n_classes = probs.shape[1]
    thresholds = np.full(n_classes, 0.5)
    for c in range(n_classes):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(labels[:, c], (probs[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


# ─────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────

def tokenize_dataset(dataset: DatasetDict, tokenizer,
                     text_column: str = "text", max_length: int = 128,
                     label_column: str = "label") -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length, padding=False)

    remove_cols = [c for c in dataset["train"].column_names if c != label_column]
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column(label_column, "labels")
    tokenized.set_format("torch")
    return tokenized


# ─────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    dataset: DatasetDict,
    output_dir: str,
    num_labels: int = 7,
    label_names: Optional[list] = None,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 2,
    # ── new options ──
    loss_type: str = "ce",          # 'ce' | 'focal' | 'bce'
    use_class_weights: bool = False,
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    multi_label: bool = False,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    gradient_accumulation_steps: int = 1,
) -> Tuple[object, object, Dict]:
    """
    Fine-tune a transformer for emotion classification.

    Single-label (default): softmax + CE/focal loss, argmax prediction.
    Multi-label (multi_label=True): sigmoid + BCE loss, per-class threshold tuning.
        Requires the dataset 'label' column to contain multi-hot float vectors.
    """
    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"Output: {output_dir}")
    print(f"Mode  : {'multi-label' if multi_label else 'single-label'} | loss={loss_type if not multi_label else 'bce'}")
    print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    lnames = label_names or _DEFAULT_LABEL_NAMES[:num_labels]
    problem_type = "multi_label_classification" if multi_label else "single_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type=problem_type,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {i: n for i, n in enumerate(lnames)}
    model.config.label2id = {n: i for i, n in enumerate(lnames)}

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        target = ["query", "value"] if "bert" in model_name.lower() else ["query", "key", "value", "dense"]
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_r * 2,
            lora_dropout=0.1, target_modules=target,
        ))
        model.print_trainable_parameters()

    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    val_split = "validation" if "validation" in tokenized else "test"

    # Loss configuration
    class_weights, pos_weight = None, None
    if multi_label:
        loss_type = "bce"
        train_labels = np.array(dataset["train"]["label"], dtype=np.float32)
        pos_weight = compute_pos_weights(train_labels)
    elif use_class_weights:
        train_labels = np.array(dataset["train"]["label"])
        class_weights = compute_class_weights(train_labels, num_labels)
        print(f"Class weights: {dict(zip(lnames, class_weights.tolist()))}")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=fp16 and torch.cuda.is_available(),
        seed=seed,
        logging_steps=100,
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[val_split],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_metrics_multi if multi_label else _metrics_single,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        loss_type=loss_type,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Predictions on validation (for stacking, no leakage) and test ──
    def _logits_to_probs(logits):
        t = torch.tensor(logits, dtype=torch.float)
        if multi_label:
            return torch.sigmoid(t).numpy()
        return torch.softmax(t, dim=-1).numpy()

    val_out  = trainer.predict(tokenized[val_split])
    test_out = trainer.predict(tokenized["test"])
    val_probs  = _logits_to_probs(val_out.predictions)
    test_probs = _logits_to_probs(test_out.predictions)
    val_labels  = val_out.label_ids
    test_labels = test_out.label_ids

    if multi_label:
        thresholds = tune_thresholds(val_probs, val_labels)
        test_preds_hard = (test_probs >= thresholds).astype(int)
        # For reporting, use argmax as the dominant single label
        test_preds = np.argmax(test_probs, axis=-1)
        true_single = np.argmax(test_labels, axis=-1)
        report = classification_report(true_single, test_preds, target_names=lnames,
                                       output_dict=True, zero_division=0)
        print(f"\n--- Test (multi-label, dominant class): {model_name} ---")
        print(classification_report(true_single, test_preds, target_names=lnames, zero_division=0))
        np.save(os.path.join(output_dir, "thresholds.npy"), thresholds)
        report_labels = true_single
    else:
        thresholds = None
        test_preds = np.argmax(test_probs, axis=-1)
        report = classification_report(test_labels, test_preds, target_names=lnames,
                                       output_dict=True, zero_division=0)
        print(f"\n--- Test results: {model_name} ---")
        print(classification_report(test_labels, test_preds, target_names=lnames, zero_division=0))
        report_labels = test_labels

    val_preds = np.argmax(val_probs, axis=-1)

    results = {
        "model_name": model_name,
        "label_names": lnames,
        "multi_label": multi_label,
        "loss_type": loss_type,
        "train_metrics": train_result.metrics,
        "test_report": report,
        "predictions": test_preds.tolist(),
        "true_labels": report_labels.tolist() if hasattr(report_labels, "tolist") else report_labels,
    }

    os.makedirs(output_dir, exist_ok=True)
    saveable = {k: v for k, v in results.items() if k not in ("predictions", "true_labels")}
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)

    # Save arrays for ensembling (test + validation for proper OOF-style stacking)
    np.save(os.path.join(output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(output_dir, "test_preds.npy"), test_preds)
    np.save(os.path.join(output_dir, "test_labels.npy"), test_labels)
    np.save(os.path.join(output_dir, "val_probs.npy"), val_probs)
    np.save(os.path.join(output_dir, "val_preds.npy"), val_preds)
    np.save(os.path.join(output_dir, "val_labels.npy"), val_labels)

    return model, tokenizer, results


def get_predictions(model_path: str, dataset: DatasetDict,
                    max_length: int = 128, batch_size: int = 64,
                    multi_label: bool = False) -> Dict[str, Dict]:
    """Load a saved model and return probabilities for validation + test splits."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    args = TrainingArguments(output_dir=model_path, per_device_eval_batch_size=batch_size, report_to="none")
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    out = {}
    for split in ["validation", "test"]:
        if split not in tokenized:
            continue
        pred_out = trainer.predict(tokenized[split])
        t = torch.tensor(pred_out.predictions, dtype=torch.float)
        probs = (torch.sigmoid(t) if multi_label else torch.softmax(t, dim=-1)).numpy()
        out[split] = {
            "probabilities": probs,
            "predictions": np.argmax(probs, axis=-1),
            "labels": pred_out.label_ids,
        }
    return out
