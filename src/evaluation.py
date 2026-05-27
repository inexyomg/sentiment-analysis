import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, List, Optional

LABEL_NAMES = ["negative", "neutral", "positive"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict:
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            pass

    per_class = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    for cls in LABEL_NAMES:
        metrics[f"f1_{cls}"] = per_class[cls]["f1-score"]

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Counts", "Normalized"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        ax.set_title(f"{model_name} — {title}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def compare_models(
    results: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    df = pd.DataFrame(results).set_index("model")
    metrics = [m for m in ["accuracy", "f1_macro", "f1_weighted"] if m in df.columns]

    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 3), 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i * width, df[metric], width, label=metric.replace("_", " ").title())
        for bar, val in zip(bars, df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(df.index, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
