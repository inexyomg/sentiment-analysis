import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, List


# ── Ekman emotion taxonomy ──────────────────────────────────────────
# Maps GoEmotions 28 fine-grained emotions → 7 Ekman classes

EKMAN_LABEL_NAMES: List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EKMAN_LABEL2ID: Dict[str, int] = {name: i for i, name in enumerate(EKMAN_LABEL_NAMES)}
EKMAN_ID2LABEL: Dict[int, str] = {i: name for i, name in enumerate(EKMAN_LABEL_NAMES)}

# GoEmotions 28 emotions → Ekman class
EMOTION_TO_EKMAN: Dict[str, str] = {
    # anger
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    # disgust
    "disgust": "disgust",
    # fear
    "fear": "fear", "nervousness": "fear",
    # joy
    "admiration": "joy", "amusement": "joy", "approval": "joy",
    "caring": "joy", "desire": "joy", "excitement": "joy",
    "gratitude": "joy", "joy": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "relief": "joy",
    # sadness
    "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness", "sadness": "sadness",
    # surprise
    "confusion": "surprise", "curiosity": "surprise",
    "realization": "surprise", "surprise": "surprise",
    # neutral
    "neutral": "neutral",
}

# GoEmotions integer index → emotion name (HuggingFace standard order)
GOEMOTION_IDX2NAME: Dict[int, str] = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "neutral", 21: "optimism", 22: "pride", 23: "realization",
    24: "relief", 25: "remorse", 26: "sadness", 27: "surprise",
}


def _multilabel_to_ekman(label_ids) -> int:
    """
    Convert a list of GoEmotions label indices (multi-label) to a single Ekman class.
    Strategy: count votes per Ekman class; break ties by Ekman order; neutral is last resort.
    """
    if not label_ids:
        return EKMAN_LABEL2ID["neutral"]

    votes: Dict[str, int] = {}
    for idx in label_ids:
        emotion = GOEMOTION_IDX2NAME.get(int(idx))
        if emotion:
            ekman = EMOTION_TO_EKMAN.get(emotion, "neutral")
            votes[ekman] = votes.get(ekman, 0) + 1

    if not votes:
        return EKMAN_LABEL2ID["neutral"]

    # Pick class with most votes; prefer non-neutral on tie
    best = max(votes, key=lambda k: (votes[k], k != "neutral"))
    return EKMAN_LABEL2ID[best]


# ── Dataset loaders ─────────────────────────────────────────────────

def load_ru_go_emotions(seed: int = 42) -> Tuple[DatasetDict, List[str]]:
    """
    Load Ru-GoEmotions from HuggingFace and convert to single-label Ekman (7 classes).
    Falls back to the original English GoEmotions if the Russian version is unavailable.

    Returns:
        dataset  — DatasetDict with 'train'/'validation'/'test', columns: text, label
        label_names — list of 7 Ekman class names
    """
    candidates = [
        ("s-nlp/ru_go_emotions", None),
        ("google-research-datasets/go_emotions", "simplified"),
        ("google-research-datasets/go_emotions", "raw"),
    ]

    ds = None
    for name, config in candidates:
        try:
            print(f"Trying: {name}" + (f" ({config})" if config else ""))
            ds = load_dataset(name, config)
            print(f"  Loaded successfully")
            break
        except Exception as e:
            print(f"  Failed: {e}")

    if ds is None:
        raise RuntimeError("Could not load Ru-GoEmotions. Check your internet connection or HuggingFace token.")

    # Normalize split names
    for old, new in [("dev", "validation"), ("development", "validation")]:
        if old in ds and new not in ds:
            ds[new] = ds.pop(old)

    # Detect text and label columns
    sample = ds[list(ds.keys())[0]]
    text_col = next((c for c in sample.column_names if "text" in c.lower()), sample.column_names[0])
    label_col = next((c for c in sample.column_names if "label" in c.lower()), None)

    def _convert(example):
        raw = example.get(label_col) or []
        # Handle both list-of-ints and list-of-strings
        if raw and isinstance(raw[0], str):
            ids = [EKMAN_LABEL2ID.get(EMOTION_TO_EKMAN.get(r.lower(), "neutral"), 6) for r in raw]
        else:
            ids = [int(x) for x in raw]
        return {"text": example[text_col], "label": _multilabel_to_ekman(ids)}

    ds = ds.map(_convert, remove_columns=sample.column_names)

    # Ensure validation split
    if "validation" not in ds and "test" in ds:
        ds["validation"] = ds["test"]

    return ds, EKMAN_LABEL_NAMES


def load_from_csv(
    filepath: str,
    text_col: str = "text",
    label_col: str = "label",
    label_names: Optional[List[str]] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> Tuple[DatasetDict, List[str]]:
    """Load a CSV with text + label columns. Labels can be strings or integers."""
    df = pd.read_csv(filepath)

    # Auto-detect columns
    if text_col not in df.columns:
        cands = [c for c in df.columns if any(k in c.lower() for k in ("text", "review", "content", "comment"))]
        text_col = cands[0] if cands else df.columns[0]
        print(f"Using '{text_col}' as text column")

    if label_col not in df.columns:
        cands = [c for c in df.columns if any(k in c.lower() for k in ("label", "sentiment", "emotion", "class"))]
        if not cands:
            raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")
        label_col = cands[0]
        print(f"Using '{label_col}' as label column")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}).dropna(subset=["text"])

    # Build label vocab if labels are strings
    if df["label"].dtype == object:
        if label_names is None:
            label_names = sorted(df["label"].unique().tolist())
        name2id = {n: i for i, n in enumerate(label_names)}
        df["label"] = df["label"].map(name2id)
    else:
        df["label"] = df["label"].astype(int)
        if label_names is None:
            label_names = [str(i) for i in sorted(df["label"].unique())]

    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["label"])

    ds = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    return ds, label_names


def load_data(
    csv_path: Optional[str] = None,
    use_ekman: bool = True,
    seed: int = 42,
) -> Tuple[DatasetDict, List[str], Dict]:
    """
    Main entry point.

    Priority:
      1. CSV file (csv_path)
      2. Kaggle input CSV (auto-detect)
      3. Ru-GoEmotions from HuggingFace (default)

    Returns:
        dataset     — DatasetDict
        label_names — ordered list of class names
        info        — split statistics
    """
    dataset, label_names = None, None

    if csv_path and os.path.exists(csv_path):
        dataset, label_names = load_from_csv(csv_path, seed=seed)

    elif os.path.exists("/kaggle/input"):
        csv_files = []
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        for path in sorted(csv_files, key=lambda p: any(k in p.lower() for k in ("emotion", "sentiment", "train")), reverse=True):
            try:
                print(f"Trying Kaggle CSV: {path}")
                dataset, label_names = load_from_csv(path, seed=seed)
                break
            except Exception as e:
                print(f"  Skipped: {e}")

    if dataset is None:
        dataset, label_names = load_ru_go_emotions(seed=seed)

    info = _compute_stats(dataset, label_names)
    return dataset, label_names, info


def _compute_stats(dataset: DatasetDict, label_names: List[str]) -> Dict:
    id2name = {i: n for i, n in enumerate(label_names)}
    stats = {}
    for split, data in dataset.items():
        df = data.to_pandas()
        dist = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
        stats[split] = {
            "size": len(df),
            "label_distribution": {id2name.get(k, k): v for k, v in dist.items()},
        }
    return stats
