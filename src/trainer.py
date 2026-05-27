import os
import json
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
import evaluate
from sklearn.metrics import classification_report


_accuracy = evaluate.load("accuracy")
_f1 = evaluate.load("f1")

LABEL_NAMES = ["negative", "neutral", "positive"]


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": _accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": _f1.compute(predictions=preds, references=labels, average="macro")["f1"],
    }


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    text_column: str = "text",
    max_length: int = 128,
) -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(batch[text_column], truncation=True, max_length=max_length, padding=False)

    remove_cols = [c for c in dataset["train"].column_names if c != "label"]
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized


def train_model(
    model_name: str,
    dataset: DatasetDict,
    output_dir: str,
    num_labels: int = 3,
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    fp16: bool = True,
    seed: int = 42,
    early_stopping_patience: int = 2,
) -> Tuple[object, object, Dict]:
    """Fine-tune a pretrained transformer for sentiment classification."""

    print(f"\n{'='*60}")
    print(f"Model : {model_name}")
    print(f"Output: {output_dir}")
    print(f"GPU   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    val_split = "validation" if "validation" in tokenized else "test"

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[val_split],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    train_result = trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    test_out = trainer.predict(tokenized["test"])
    preds = np.argmax(test_out.predictions, axis=-1)
    labels = test_out.label_ids
    probs = torch.softmax(torch.tensor(test_out.predictions, dtype=torch.float), dim=-1).numpy()

    print(f"\n--- Test results: {model_name} ---")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    results = {
        "model_name": model_name,
        "train_metrics": train_result.metrics,
        "test_report": classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True),
        "predictions": preds.tolist(),
        "probabilities": probs.tolist(),
        "true_labels": labels.tolist(),
    }

    # Persist results (without large arrays)
    saveable = {k: v for k, v in results.items() if k not in ("predictions", "probabilities", "true_labels")}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)

    # Save predictions separately for ensemble
    np.save(os.path.join(output_dir, "test_probs.npy"), probs)
    np.save(os.path.join(output_dir, "test_preds.npy"), preds)
    np.save(os.path.join(output_dir, "test_labels.npy"), labels)

    return model, tokenizer, results


def get_predictions(
    model_path: str,
    dataset: DatasetDict,
    max_length: int = 128,
    batch_size: int = 64,
) -> Dict[str, Dict]:
    """Load saved model and return probabilities for all splits."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=max_length)

    args = TrainingArguments(
        output_dir=model_path,
        per_device_eval_batch_size=batch_size,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer)

    out = {}
    for split in ["validation", "test"]:
        if split not in tokenized:
            continue
        pred_out = trainer.predict(tokenized[split])
        probs = torch.softmax(torch.tensor(pred_out.predictions, dtype=torch.float), dim=-1).numpy()
        out[split] = {
            "probabilities": probs,
            "predictions": np.argmax(probs, axis=-1),
            "labels": pred_out.label_ids,
        }
    return out
