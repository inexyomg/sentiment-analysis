import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict


LABEL_MAPPING = {
    "positive": 2, "pos": 2, "1": 2, 1: 2,
    "neutral":  1, "neu": 1,
    "negative": 0, "neg": 0, "0": 0, 0: 0,
    # Numeric ratings → sentiment
    5: 2, 4: 2, 3: 1, 2: 0, 1: 0,
}

LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}

# Known HuggingFace Russian sentiment datasets to try in order
HF_DATASET_CANDIDATES = [
    ("sepidmnorozy/Russian_sentiment", None),
    ("cedr", "cedr-multi-11k"),
    ("ai-forever/rureviews", None),
]


def _normalize_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].str.strip().str.lower()
    df[label_col] = df[label_col].map(lambda x: LABEL_MAPPING.get(x, LABEL_MAPPING.get(str(x).lower())))
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    return df


def load_from_hf(dataset_name: str, config_name: Optional[str] = None) -> DatasetDict:
    print(f"Loading from HuggingFace: {dataset_name}" + (f" ({config_name})" if config_name else ""))
    ds = load_dataset(dataset_name, config_name)

    # Normalize split names
    rename_map = {}
    for split in list(ds.keys()):
        if split in ("dev", "development"):
            rename_map[split] = "validation"
    for old, new in rename_map.items():
        ds[new] = ds.pop(old)

    return ds


def load_from_csv(filepath: str,
                  text_col: str = "text",
                  label_col: str = "label",
                  test_size: float = 0.15,
                  val_size: float = 0.15,
                  seed: int = 42) -> DatasetDict:
    df = pd.read_csv(filepath)

    # Auto-detect columns if defaults not found
    if text_col not in df.columns:
        candidates = [c for c in df.columns if "text" in c.lower() or "review" in c.lower() or "content" in c.lower()]
        if not candidates:
            raise ValueError(f"Cannot find text column in: {df.columns.tolist()}")
        text_col = candidates[0]
        print(f"Using '{text_col}' as text column")

    if label_col not in df.columns:
        candidates = [c for c in df.columns if "label" in c.lower() or "sentiment" in c.lower() or "class" in c.lower()]
        if not candidates:
            raise ValueError(f"Cannot find label column in: {df.columns.tolist()}")
        label_col = candidates[0]
        print(f"Using '{label_col}' as label column")

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna(subset=["text"])
    df = _normalize_labels(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["label"])

    return DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })


def _try_kaggle_csv(kaggle_dir: str = "/kaggle/input") -> Optional[DatasetDict]:
    """Find and load a CSV dataset from Kaggle input directory."""
    csv_files = []
    for root, _, files in os.walk(kaggle_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    if not csv_files:
        return None

    # Prefer files with sentiment-related names
    ranked = sorted(csv_files, key=lambda p: any(
        kw in p.lower() for kw in ["sentiment", "review", "train"]), reverse=True)

    for path in ranked:
        try:
            print(f"Trying Kaggle CSV: {path}")
            return load_from_csv(path)
        except Exception as e:
            print(f"  Skipped: {e}")

    return None


def load_data(
    dataset_name: Optional[str] = "sepidmnorozy/Russian_sentiment",
    csv_path: Optional[str] = None,
) -> Tuple[DatasetDict, Dict]:
    """
    Load Russian sentiment data with fallback logic:
      1. CSV path (if provided)
      2. Kaggle input directory (if running on Kaggle)
      3. HuggingFace datasets
    """
    dataset = None

    if csv_path and os.path.exists(csv_path):
        dataset = load_from_csv(csv_path)

    elif os.path.exists("/kaggle/input"):
        dataset = _try_kaggle_csv()

    if dataset is None:
        candidates = HF_DATASET_CANDIDATES if dataset_name is None else [(dataset_name, None)] + HF_DATASET_CANDIDATES
        for name, config in candidates:
            try:
                dataset = load_from_hf(name, config)
                break
            except Exception as e:
                print(f"  Failed ({name}): {e}")

    if dataset is None:
        raise RuntimeError("Could not load dataset from any source. "
                           "Provide a CSV file via csv_path= or add a Kaggle dataset.")

    # Ensure validation split exists
    if "validation" not in dataset and "test" in dataset:
        dataset["validation"] = dataset["test"]

    info = _compute_stats(dataset)
    return dataset, info


def _compute_stats(dataset: DatasetDict) -> Dict:
    stats = {}
    for split, data in dataset.items():
        df = data.to_pandas()
        dist = df["label"].value_counts().sort_index().to_dict() if "label" in df.columns else {}
        stats[split] = {"size": len(df), "label_distribution": {LABEL_NAMES.get(k, k): v for k, v in dist.items()}}
    return stats
