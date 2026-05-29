import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, List, Union


# ── Ekman taxonomy ─────────────────────────────────────────────────

EKMAN_LABEL_NAMES: List[str] = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
EKMAN_LABEL2ID:   Dict[str, int] = {n: i for i, n in enumerate(EKMAN_LABEL_NAMES)}
EKMAN_ID2LABEL:   Dict[int, str] = {i: n for i, n in enumerate(EKMAN_LABEL_NAMES)}

# GoEmotions 28 fine-grained → Ekman
EMOTION_TO_EKMAN: Dict[str, str] = {
    "anger": "anger", "annoyance": "anger", "disapproval": "anger",
    "disgust": "disgust",
    "fear": "fear", "nervousness": "fear",
    "admiration": "joy", "amusement": "joy", "approval": "joy",
    "caring": "joy", "desire": "joy", "excitement": "joy",
    "gratitude": "joy", "joy": "joy", "love": "joy",
    "optimism": "joy", "pride": "joy", "relief": "joy",
    "disappointment": "sadness", "embarrassment": "sadness",
    "grief": "sadness", "remorse": "sadness", "sadness": "sadness",
    "confusion": "surprise", "curiosity": "surprise",
    "realization": "surprise", "surprise": "surprise",
    "neutral": "neutral",
}

# GoEmotions index → emotion name
GOEMOTION_IDX2NAME: Dict[int, str] = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
    12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
    16: "grief", 17: "joy", 18: "love", 19: "nervousness",
    20: "neutral", 21: "optimism", 22: "pride", 23: "realization",
    24: "relief", 25: "remorse", 26: "sadness", 27: "surprise",
}

# CEDR: 5 emotion indices → Ekman class
CEDR_IDX2EKMAN: Dict[int, str] = {
    0: "joy", 1: "sadness", 2: "surprise", 3: "fear", 4: "anger",
}

# Djacon/ru-izard-emotions: 10 Izard indices → Ekman class
IZARD_IDX2EKMAN: Dict[int, str] = {
    0: "neutral",
    1: "joy",
    2: "sadness",
    3: "anger",
    4: "joy",      # interest / enthusiasm
    5: "surprise",
    6: "disgust",
    7: "fear",
    8: "sadness",  # guilt
    9: "disgust",  # shame
}


def _multilabel_to_ekman(label_ids, idx2ekman: Dict[int, str]) -> int:
    """
    Majority-vote conversion of a multi-label list → single Ekman class.
    Non-neutral wins on tie; neutral is fallback when empty.
    """
    if not label_ids:
        return EKMAN_LABEL2ID["neutral"]

    votes: Dict[str, int] = {}
    for idx in label_ids:
        ekman = idx2ekman.get(int(idx), "neutral")
        votes[ekman] = votes.get(ekman, 0) + 1

    best = max(votes, key=lambda k: (votes[k], k != "neutral"))
    return EKMAN_LABEL2ID[best]


# ── Individual dataset loaders ──────────────────────────────────────

def load_ru_go_emotions(config: str = "simplified") -> DatasetDict:
    """
    seara/ru_go_emotions — Russian translation of Google GoEmotions.
    config='simplified' (~54k): columns include 'labels' (list[int 0-27]), train/val/test splits.
    config='raw' (~211k): one binary column per emotion name, only 'train' split.
    Russian text is always in 'ru_text'; 'text' is the English original.
    28 GoEmotions labels → Ekman 7 single-label.
    """
    print(f"Loading seara/ru_go_emotions ({config})...")
    ds = load_dataset("seara/ru_go_emotions", config)

    cols     = ds["train"].column_names
    cols_set = set(c.lower() for c in cols)

    # Russian text is in 'ru_text'; 'text' is the English original
    text_col = "ru_text" if "ru_text" in cols else next(
        (c for c in cols if "ru" in c.lower() and "text" in c.lower()), "text"
    )

    _idx2ekman    = {i: EMOTION_TO_EKMAN.get(n, "neutral") for i, n in GOEMOTION_IDX2NAME.items()}
    _name2idx     = {v: k for k, v in GOEMOTION_IDX2NAME.items()}

    # Detect label format
    label_col = next((c for c in cols if "label" in c.lower()), None)

    # raw config: has one binary column per GoEmotion name (no 'labels' list column)
    emotion_bin_cols = [c for c in cols if c.lower() in _name2idx]

    if label_col:
        # simplified config: 'labels' is a list of present GoEmotion indices → majority vote
        def _convert(ex):
            raw = ex.get(label_col)
            if isinstance(raw, (list, tuple)):
                ids = [int(x) for x in raw]
            elif raw is not None:
                ids = [int(raw)]
            else:
                ids = []
            return {"text": str(ex[text_col] or ""), "label": _multilabel_to_ekman(ids, _idx2ekman)}

        print(f"  columns: text='{text_col}', label format=list")
        ds = ds.map(_convert, remove_columns=cols)
        ds = _normalize_splits(ds)

    elif emotion_bin_cols:
        # raw config: expand multi-label → one row per unique Ekman class present.
        # Majority vote would lose disgust/fear whenever they co-occur with commoner emotions.
        print(f"  columns: text='{text_col}', label format=binary per-emotion (expanding multi-label)")
        dfs = []
        for split_name, split_ds in ds.items():
            df = split_ds.select_columns([text_col] + emotion_bin_cols).to_pandas()
            df["_split"] = split_name
            dfs.append(df)
        raw_df = pd.concat(dfs, ignore_index=True)

        # Melt → one row per (text, emotion that is 1)
        melted = raw_df.melt(
            id_vars=[text_col, "_split"],
            value_vars=emotion_bin_cols,
            var_name="emotion",
            value_name="present",
        )
        melted = melted[melted["present"] == 1].copy()
        melted["label"] = (melted["emotion"].str.lower()
                                            .map(EMOTION_TO_EKMAN)
                                            .fillna("neutral")
                                            .map(EKMAN_LABEL2ID))
        melted["text"] = melted[text_col].astype(str).str.strip()

        # Deduplicate: keep one row per (text, ekman label) per split
        melted = (melted[[text_col, "_split", "label"]]
                  .rename(columns={text_col: "text"})
                  .dropna(subset=["label"])
                  .drop_duplicates(subset=["text", "label"]))
        melted["label"] = melted["label"].astype(int)

        ds = DatasetDict({
            split: Dataset.from_pandas(
                melted[melted["_split"] == split][["text", "label"]].reset_index(drop=True)
            )
            for split in melted["_split"].unique()
        })
        ds = _normalize_splits(ds)

    else:
        raise ValueError(
            f"Cannot find labels in seara/ru_go_emotions ({config}).\n"
            f"Columns: {cols}\n"
            f"Expected: 'labels' list column OR individual emotion binary columns."
        )

    print(f"  → {sum(len(ds[s]) for s in ds):,} examples (ru_go_emotions/{config}, RU text)")
    return ds


def load_cedr() -> DatasetDict:
    """
    cedr — Corpus for Emotion Detection in Russian.
    Native Russian texts from blogs, news, Twitter.
    5 multi-label emotion classes → Ekman 7 (neutral inferred from empty label set).
    """
    print("Loading cedr...")
    try:
        ds = load_dataset("cedr")
    except Exception:
        ds = load_dataset("sagteam/cedr_v1")

    # CEDR has binary columns per emotion or a list — handle both
    sample = ds["train"].column_names

    def _convert(ex):
        # Format A: separate binary columns — joy, sadness, surprise, fear, anger
        if "joy" in sample:
            present = [i for i, col in enumerate(["joy", "sadness", "surprise", "fear", "anger"])
                       if ex.get(col, 0)]
        # Format B: 'labels' list of ints
        elif "labels" in sample:
            present = [int(x) for x in (ex.get("labels") or [])]
        else:
            present = []

        text = ex.get("text") or ex.get("sentence") or ""
        return {"text": text, "label": _multilabel_to_ekman(present, CEDR_IDX2EKMAN)}

    remove = [c for c in sample if c not in ("text", "sentence")]
    ds = ds.map(_convert, remove_columns=remove)
    if "sentence" in ds["train"].column_names:
        ds = ds.rename_column("sentence", "text")
    ds = _normalize_splits(ds)
    print(f"  → {sum(len(ds[s]) for s in ds):,} examples")
    return ds


def load_ru_izard_emotions() -> DatasetDict:
    """
    Djacon/ru-izard-emotions — ~30k Russian texts (DeepL translation from GoEmotions).
    10 Izard emotion classes → Ekman 7.

    Dataset columns: text + either:
      a) 'labels' — list of present Izard indices [0-9]  (multi-label)
      b) 'label'  — single Izard index int
      c) separate binary columns: Neutral, Joy, Sadness, Anger, Enthusiasm,
                                   Surprise, Disgust, Fear, Guilt, Shame
    """
    # Ordered by Izard index — must match IZARD_IDX2EKMAN keys
    IZARD_COL_NAMES = ("neutral", "joy", "sadness", "anger", "enthusiasm",
                       "surprise", "disgust", "fear", "guilt", "shame")

    print("Loading Djacon/ru-izard-emotions...")
    ds = load_dataset("Djacon/ru-izard-emotions")

    cols      = ds["train"].column_names
    cols_low  = {c.lower(): c for c in cols}
    text_col  = next((c for c in cols if "text" in c.lower()), cols[0])

    # Detect label format
    label_col    = cols_low.get("labels") or cols_low.get("label")
    emotion_cols = [cols_low[n] for n in IZARD_COL_NAMES if n in cols_low]

    if label_col:
        def _convert(ex):
            raw = ex.get(label_col)
            if isinstance(raw, (list, tuple)):
                ids = [int(x) for x in raw]
            elif raw is not None:
                ids = [int(raw)]   # single int label
            else:
                ids = []
            return {"text": str(ex[text_col] or ""), "label": _multilabel_to_ekman(ids, IZARD_IDX2EKMAN)}
    elif emotion_cols:
        # Separate binary columns — collect indices of non-zero ones
        def _convert(ex):
            ids = [i for i, col in enumerate(emotion_cols) if ex.get(col, 0)]
            return {"text": str(ex[text_col] or ""), "label": _multilabel_to_ekman(ids, IZARD_IDX2EKMAN)}
    else:
        raise ValueError(
            f"Cannot find label column in Djacon/ru-izard-emotions.\n"
            f"Columns: {cols}\n"
            f"Expected: 'label', 'labels', or individual emotion columns"
        )

    ds = ds.map(_convert, remove_columns=cols)
    ds = _normalize_splits(ds)
    print(f"  → {sum(len(ds[s]) for s in ds):,} examples (ru-izard-emotions)")
    return ds


def _normalize_splits(ds: DatasetDict, seed: int = 42) -> DatasetDict:
    """Standardize split names and ensure train/validation/test exist."""
    for old, new in [("dev", "validation"), ("development", "validation"), ("valid", "validation")]:
        if old in ds and new not in ds:
            ds[new] = ds.pop(old)
    if "validation" not in ds and "test" in ds:
        ds["validation"] = ds["test"]

    # If only "train" exists (e.g. seara/ru_go_emotions raw config), create val/test
    if set(ds.keys()) == {"train"}:
        df = ds["train"].to_pandas()
        stratify_col = df["label"] if "label" in df.columns else None
        train_df, tmp_df = train_test_split(df, test_size=0.20, random_state=seed, stratify=stratify_col)
        stratify_tmp = tmp_df["label"] if "label" in tmp_df.columns else None
        val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=seed, stratify=stratify_tmp)
        ds = DatasetDict({
            "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
        })
    return ds


# ── Multi-dataset merge ─────────────────────────────────────────────

def merge_datasets(
    datasets: Dict[str, DatasetDict],
    test_size:     float = 0.15,
    val_size:      float = 0.15,
    seed:          int   = 42,
    max_per_class: Optional[int] = None,
) -> DatasetDict:
    """
    Concatenate multiple DatasetDicts (each must have 'text' and 'label' columns)
    and re-split into train / validation / test, stratified by label.

    max_per_class — cap per label before splitting (all splits stay proportional).
    """
    print("\nMerging datasets...")
    all_records = []
    for name, ds in datasets.items():
        for split in ds:
            df = ds[split].to_pandas()[["text", "label"]]
            df["source"] = name
            all_records.append(df)
        print(f"  {name}: {sum(len(ds[s]) for s in ds):,} examples")

    full_df = pd.concat(all_records, ignore_index=True)
    full_df = full_df.dropna(subset=["text", "label"])
    full_df["label"] = full_df["label"].astype(int)

    print(f"\nTotal before split: {len(full_df):,}")
    print("Label distribution:")
    for lbl, cnt in sorted(full_df["label"].value_counts().items()):
        print(f"  {EKMAN_ID2LABEL[lbl]:<12s}: {cnt:>6,}  ({cnt/len(full_df)*100:.1f}%)")

    if max_per_class is not None:
        full_df = (
            full_df
            .groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        print(f"\nAfter cap ({max_per_class:,}/class): {len(full_df):,} examples")

    train_df, test_df = train_test_split(
        full_df, test_size=test_size, random_state=seed, stratify=full_df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df["label"]
    )

    merged = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    print(f"\nFinal splits:")
    for split in merged:
        print(f"  {split:12s}: {len(merged[split]):,}")
    return merged


# ── Sentiment → Ekman (approximate) ───────────────────────────────
# Используется для датасетов без разметки по эмоциям (только pos/neu/neg).
# Маппинг приблизительный: negative охватывает sadness, anger, disgust.
# Такие примеры добавляют объём, но снижают точность — учитывай при анализе.

SENTIMENT_TO_EKMAN: Dict[str, str] = {
    "positive": "joy",
    "neutral":  "neutral",
    "negative": "sadness",   # консервативный выбор; anger/disgust тоже возможны
    # числовые варианты
    "2": "joy", "1": "neutral", "0": "sadness",
    2:   "joy",  1:  "neutral",  0:  "sadness",
}


def _sentiment_df_to_ekman(df: pd.DataFrame,
                            text_col: str = "text",
                            label_col: str = "label") -> pd.DataFrame:
    """Convert pos/neu/neg sentiment labels → Ekman integer labels."""
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
    df = df.dropna(subset=["text", "raw_label"])
    df["raw_label"] = df["raw_label"].astype(str).str.strip().str.lower()
    df["label"] = df["raw_label"].map(lambda x: EKMAN_LABEL2ID.get(
        SENTIMENT_TO_EKMAN.get(x, SENTIMENT_TO_EKMAN.get(x.split(".")[0], None)), None
    ))
    return df.dropna(subset=["label"])[["text", "label"]].copy()


def load_rureviews(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    sismetanin/rureviews — 90k balanced Russian e-commerce reviews.

    Priority:
      1. Kaggle manual annotation — /kaggle/input/datasets/inexyy/ru-reviews-tweets/rureviews/
         Files: ru_reviews_ekman7_{train,val,test}.csv  (column: ekman_emotion)
      2. Fallback — download from GitHub + coarse pos/neg/neu→Ekman mapping.
    """
    _KAGGLE_DIR = "/kaggle/input/datasets/inexyy/ru-reviews-tweets/rureviews"
    _SPLIT_FILES = {
        "train":      "ru_reviews_ekman7_train.csv",
        "validation": "ru_reviews_ekman7_val.csv",
        "test":       "ru_reviews_ekman7_test.csv",
    }

    if os.path.isdir(_KAGGLE_DIR):
        splits: dict = {}
        total = 0
        for split_name, fname in _SPLIT_FILES.items():
            fpath = os.path.join(_KAGGLE_DIR, fname)
            df = pd.read_csv(fpath)
            text_col = next(
                (c for c in df.columns if "review" in c.lower() or "text" in c.lower()),
                df.columns[0],
            )
            df["text"]  = df[text_col].astype(str).str.strip()
            df["label"] = df["ekman_emotion"].astype(str).str.lower().map(EKMAN_LABEL2ID)
            df = df.dropna(subset=["label"])[["text", "label"]].copy()
            df["label"] = df["label"].astype(int)
            splits[split_name] = Dataset.from_pandas(df.reset_index(drop=True))
            total += len(df)
        print(f"RuReviews (Kaggle): {total:,} examples (ekman_emotion, manual annotation)")
        return _normalize_splits(DatasetDict(splits))

    # ── Fallback: GitHub download + coarse sentiment→Ekman mapping ────────────
    import urllib.request

    URL = ("https://raw.githubusercontent.com/sismetanin/rureviews/master/"
           "women-clothing-accessories.3-class.balanced.csv")

    save_path = os.path.join(cache_dir or os.path.expanduser("~/.cache/rureviews"),
                             "rureviews.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print(f"Downloading RuReviews → {save_path} ...")
        urllib.request.urlretrieve(URL, save_path)

    df = pd.read_csv(save_path, sep="\t" if "\t" in open(save_path).read(200) else ",")

    text_col  = next((c for c in df.columns if "review" in c.lower() or "text" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "sentiment" in c.lower() or "label" in c.lower() or "class" in c.lower()), df.columns[-1])

    df = _sentiment_df_to_ekman(df, text_col, label_col)
    df["label"] = df["label"].astype(int)

    print(f"RuReviews: {len(df):,} examples (sentiment→Ekman, approx)")

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])

    ds = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    return _normalize_splits(ds)


def load_rusentitweet(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    sismetanin/rusentitweet — 13.4k manually annotated Russian tweets.

    Priority:
      1. Kaggle manual annotation — /kaggle/input/datasets/inexyy/ru-reviews-tweets/rusentitweet/
         Files: rusentitweet_ekman7_{train,val,test}.csv  (column: ekman_emotion)
         Rule: if original_label == 'neutral' → force ekman_emotion = 'neutral'.
      2. Fallback — download from GitHub + coarse pos/neg/neu→Ekman mapping.
    """
    _KAGGLE_DIR = "/kaggle/input/datasets/inexyy/ru-reviews-tweets/rusentitweet"
    _SPLIT_FILES = {
        "train":      "rusentitweet_ekman7_train.csv",
        "validation": "rusentitweet_ekman7_val.csv",
        "test":       "rusentitweet_ekman7_test.csv",
    }

    if os.path.isdir(_KAGGLE_DIR):
        splits: dict = {}
        total = 0
        for split_name, fname in _SPLIT_FILES.items():
            fpath = os.path.join(_KAGGLE_DIR, fname)
            df = pd.read_csv(fpath)
            text_col = next(
                (c for c in df.columns if "text" in c.lower() or "tweet" in c.lower()),
                df.columns[0],
            )
            df["text"] = df[text_col].astype(str).str.strip()
            # Rule: original_label == neutral → override ekman_emotion
            if "original_label" in df.columns:
                mask = df["original_label"].astype(str).str.lower() == "neutral"
                df.loc[mask, "ekman_emotion"] = "neutral"
            df["label"] = df["ekman_emotion"].astype(str).str.lower().map(EKMAN_LABEL2ID)
            df = df.dropna(subset=["label"])[["text", "label"]].copy()
            df["label"] = df["label"].astype(int)
            splits[split_name] = Dataset.from_pandas(df.reset_index(drop=True))
            total += len(df)
        print(f"RuSentiTweet (Kaggle): {total:,} examples (ekman_emotion, manual annotation)")
        return _normalize_splits(DatasetDict(splits))

    # ── Fallback: GitHub download + coarse sentiment→Ekman mapping ────────────
    import urllib.request

    URL = ("https://raw.githubusercontent.com/sismetanin/rusentitweet/master/"
           "rusentitweet_full.csv")

    save_path = os.path.join(cache_dir or os.path.expanduser("~/.cache/rusentitweet"),
                             "rusentitweet.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if not os.path.exists(save_path):
        print(f"Downloading RuSentiTweet → {save_path} ...")
        urllib.request.urlretrieve(URL, save_path)

    df = pd.read_csv(save_path)
    text_col  = next((c for c in df.columns if "text" in c.lower() or "tweet" in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if "sentiment" in c.lower() or "label" in c.lower()), df.columns[-1])

    # Keep only pos/neu/neg; drop speech_act / skip
    df = df[df[label_col].astype(str).str.lower().isin(["positive", "neutral", "negative",
                                                          "pos", "neu", "neg", "0", "1", "2"])]
    df = _sentiment_df_to_ekman(df, text_col, label_col)
    df["label"] = df["label"].astype(int)

    print(f"RuSentiTweet: {len(df):,} examples (sentiment→Ekman, approx)")

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])

    ds = DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })
    return _normalize_splits(ds)


def load_aniemore_resd() -> DatasetDict:
    """
    Aniemore/resd (+ Aniemore/resd_annotated fallback) — Russian Emotional Speech Dataset.
    Native Russian speech transcripts, ~4.5k examples.
    7 classes: angry, disgust, enthusiasm, fear, happy, neutral, sad → Ekman 7.
    Покрывает disgust, которого нет в CEDR и Dusha — ценный источник для этапа 2.
    """
    ANIEMORE_TO_EKMAN: Dict[str, str] = {
        "angry":      "anger",
        "anger":      "anger",
        "disgust":    "disgust",
        "enthusiasm": "joy",
        "fear":       "fear",
        "happy":      "joy",
        "happiness":  "joy",
        "neutral":    "neutral",
        "sad":        "sadness",
        "sadness":    "sadness",
        "surprise":   "surprise",
    }

    TEXT_CANDIDATES  = ("text", "transcription", "raw_text", "sentence", "utterance", "phrase")
    LABEL_CANDIDATES = ("emotion", "label", "tag", "sentiment")

    def _find_col(cols, candidates):
        cols_lower = {c.lower(): c for c in cols}
        for name in candidates:
            if name in cols_lower:
                return cols_lower[name]
        for cand in candidates:
            for c in cols:
                if cand in c.lower():
                    return c
        return None

    for dataset_id in ("Aniemore/resd_annotated", "Aniemore/resd"):
        try:
            print(f"  Loading {dataset_id}...")
            ds = load_dataset(dataset_id)
            split0 = list(ds.keys())[0]
            cols = ds[split0].column_names
            text_col  = _find_col(cols, TEXT_CANDIDATES)
            label_col = _find_col(cols, LABEL_CANDIDATES)
            if not text_col or not label_col:
                print(f"    ✗ {dataset_id}: columns not found {cols}")
                continue

            def _convert(ex, tc=text_col, lc=label_col):
                raw = str(ex[lc]).strip().lower()
                ekman = ANIEMORE_TO_EKMAN.get(raw)
                return {"text": str(ex[tc] or "").strip(), "label": EKMAN_LABEL2ID[ekman] if ekman else -1}

            ds = ds.map(_convert, remove_columns=cols)
            for split in list(ds.keys()):
                ds[split] = ds[split].filter(lambda ex: ex["label"] >= 0 and len(ex["text"]) > 3)
            ds = _normalize_splits(ds)
            total = sum(len(ds[s]) for s in ds)
            print(f"  → {total:,} examples ({dataset_id}, native RU, 7 classes)")
            return ds
        except Exception as e:
            print(f"    ✗ {dataset_id}: {e}")

    raise RuntimeError("Could not load Aniemore/resd from any source.")


def load_cedr_m7() -> DatasetDict:
    """
    Aniemore/cedr-m7 — CEDR corpus extended to 7 Ekman classes.
    Adds disgust and neutral on top of the base 5-class CEDR.
    Native Russian text from LiveJournal, Lenta.ru, Twitter.
    Columns: text (str), labels (str emotion name), source (str).
    """
    CEDR_M7_TO_EKMAN: Dict[str, str] = {
        "neutral":    "neutral",
        "happiness":  "joy",
        "sadness":    "sadness",
        "enthusiasm": "joy",    # energetic positive state
        "fear":       "fear",
        "anger":      "anger",
        "disgust":    "disgust",
    }

    print("Loading Aniemore/cedr-m7...")
    ds = load_dataset("Aniemore/cedr-m7")

    split0 = list(ds.keys())[0]
    cols = ds[split0].column_names
    cols_lower = {c.lower(): c for c in cols}

    text_col  = cols_lower.get("text") or next((c for c in cols if "text" in c.lower()), cols[0])
    label_col = cols_lower.get("labels") or cols_lower.get("label") or next(
        (c for c in cols if "label" in c.lower() or "emotion" in c.lower()), None
    )
    if not label_col:
        raise ValueError(f"Cannot find label column in Aniemore/cedr-m7. Columns: {cols}")

    def _convert(ex):
        raw   = str(ex.get(label_col, "neutral")).strip().lower()
        ekman = CEDR_M7_TO_EKMAN.get(raw)
        label = EKMAN_LABEL2ID[ekman] if ekman else EKMAN_LABEL2ID["neutral"]
        return {"text": str(ex[text_col] or "").strip(), "label": label}

    ds = ds.map(_convert, remove_columns=cols)
    ds = _normalize_splits(ds)
    print(f"  → {sum(len(ds[s]) for s in ds):,} examples (cedr-m7, 7 classes, native RU)")
    return ds


def load_xed_russian(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    XED — Multilingual Emotion Dataset, Russian subset (~2,400 lines).
    Source: Helsinki-NLP/XED, movie subtitles (OPUS).
    8 Plutchik emotions (multi-label integers 1-8) → Ekman 7 single-label.

    Plutchik → Ekman mapping:
      1 anger       → anger
      2 anticipation→ surprise
      3 disgust     → disgust
      4 fear        → fear
      5 joy         → joy
      6 sadness     → sadness
      7 surprise    → surprise
      8 trust       → joy
    """
    import urllib.request

    # Index 1-8 (alphabetical Plutchik) → Ekman label ID
    _XED2EKMAN: Dict[int, int] = {
        1: EKMAN_LABEL2ID["anger"],
        2: EKMAN_LABEL2ID["surprise"],   # anticipation
        3: EKMAN_LABEL2ID["disgust"],
        4: EKMAN_LABEL2ID["fear"],
        5: EKMAN_LABEL2ID["joy"],
        6: EKMAN_LABEL2ID["sadness"],
        7: EKMAN_LABEL2ID["surprise"],
        8: EKMAN_LABEL2ID["joy"],        # trust
    }

    cache_root = cache_dir or os.path.expanduser("~/.cache/xed_russian")
    os.makedirs(cache_root, exist_ok=True)

    # Try several candidate locations in the Helsinki-NLP/XED GitHub repo
    BASE = "https://raw.githubusercontent.com/Helsinki-NLP/XED/master"
    url_candidates = [
        (f"{BASE}/xed_splits/ru-EXT/train.tsv", f"{BASE}/xed_splits/ru-EXT/test.tsv"),
        (f"{BASE}/xed_splits/ru/train.tsv",     f"{BASE}/xed_splits/ru/test.tsv"),
        (f"{BASE}/ru-finegrained.tsv",           None),
        (f"{BASE}/data/ru.tsv",                  None),
    ]

    def _parse_tsv(path: str) -> pd.DataFrame:
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                text  = parts[0].strip()
                # labels: comma-separated integers, or "0" for neutral
                raw_labels = [s.strip() for s in parts[-1].split(",") if s.strip()]
                ids = []
                for s in raw_labels:
                    try:
                        n = int(s)
                        if 1 <= n <= 8:
                            ids.append(n)
                    except ValueError:
                        pass
                if text:
                    rows.append({"text": text, "emotion_ids": ids})
        return pd.DataFrame(rows)

    def _download(url: str, save_path: str) -> bool:
        try:
            urllib.request.urlretrieve(url, save_path)
            return True
        except Exception:
            return False

    dfs: List[pd.DataFrame] = []

    for train_url, test_url in url_candidates:
        train_path = os.path.join(cache_root, "xed_ru_train.tsv")
        test_path  = os.path.join(cache_root, "xed_ru_test.tsv")
        single_path = os.path.join(cache_root, "xed_ru_all.tsv")

        ok = False
        if test_url:
            # Two-file layout (train + test)
            if (os.path.exists(train_path) or _download(train_url, train_path)) and \
               (os.path.exists(test_path)  or _download(test_url,  test_path)):
                try:
                    dfs = [_parse_tsv(train_path), _parse_tsv(test_path)]
                    ok = True
                    print(f"  XED: loaded from {train_url}")
                except Exception as e:
                    print(f"  XED parse error: {e}")
        else:
            # Single-file layout
            if os.path.exists(single_path) or _download(train_url, single_path):
                try:
                    dfs = [_parse_tsv(single_path)]
                    ok = True
                    print(f"  XED: loaded from {train_url}")
                except Exception as e:
                    print(f"  XED parse error: {e}")

        if ok:
            break

    if not dfs:
        raise RuntimeError(
            "Could not download XED Russian data from Helsinki-NLP/XED GitHub. "
            "Check network access."
        )

    full_df = pd.concat(dfs, ignore_index=True)

    def _to_ekman(ids):
        if not ids:
            return EKMAN_LABEL2ID["neutral"]
        votes: Dict[int, int] = {}
        for i in ids:
            lbl = _XED2EKMAN.get(i, EKMAN_LABEL2ID["neutral"])
            votes[lbl] = votes.get(lbl, 0) + 1
        return max(votes, key=lambda k: (votes[k], k != EKMAN_LABEL2ID["neutral"]))

    full_df["label"] = full_df["emotion_ids"].apply(_to_ekman)
    full_df = full_df[["text", "label"]].query("text.str.len() > 3").copy()
    full_df["label"] = full_df["label"].astype(int)

    print(f"  → {len(full_df):,} examples (XED Russian, Plutchik→Ekman, movie subtitles)")

    train_df, tmp = train_test_split(full_df, test_size=0.20, random_state=42, stratify=full_df["label"])
    val_df, test_df = train_test_split(tmp,   test_size=0.50, random_state=42, stratify=tmp["label"])

    return _normalize_splits(DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    }))


def load_stage2_clean(
    use_cedr:         bool = True,
    use_cedr_m7:      bool = True,
    use_brighter_hf:  bool = True,
    use_aniemore:     bool = True,
    seed:             int  = 42,
) -> DatasetDict:
    """
    Stage-2 corpus: только высококачественные нативные RU датасеты.
    Совместное покрытие всех 7 классов Ekman:
      CEDR-M7:   все 7 классов Ekman incl. disgust + neutral  (расширенная версия CEDR)
      BRIGHTER:  anger, disgust, fear, joy, sadness, surprise  (Toloka, нативный RU)
      Aniemore:  anger, disgust, enthusiasm→joy, fear, neutral, sadness
    """
    sources: Dict[str, DatasetDict] = {}

    if use_cedr_m7:
        # cedr-m7 preferred over base cedr — covers disgust + neutral
        try:
            sources["cedr_m7"] = load_cedr_m7()
        except Exception as e:
            print(f"  WARNING: cedr-m7 failed: {e}")
            if use_cedr:
                try:
                    sources["cedr"] = load_cedr()
                except Exception as e2:
                    print(f"  WARNING: cedr fallback failed: {e2}")
    elif use_cedr:
        try:
            sources["cedr"] = load_cedr()
        except Exception as e:
            print(f"  WARNING: cedr failed: {e}")

    if use_brighter_hf:
        try:
            sources["brighter_hf"] = load_brighter_hf()
        except Exception as e:
            print(f"  WARNING: brighter_hf failed: {e}")

    if use_aniemore:
        try:
            sources["aniemore"] = load_aniemore_resd()
        except Exception as e:
            print(f"  WARNING: aniemore/resd failed: {e}")

    if not sources:
        raise RuntimeError("No stage-2 datasets loaded — check network access.")

    if len(sources) == 1:
        return list(sources.values())[0]

    return merge_datasets(sources, test_size=0.15, val_size=0.15, seed=seed)


def load_brighter_hf() -> DatasetDict:
    """
    BRIGHTER (SemEval-2025 Task 11) — Russian subset.
    HuggingFace: brighter-dataset/BRIGHTER-emotion-categories, config='rus'.
    6 Ekman emotions, Toloka-annotated native Russian texts.
    """
    print("Loading brighter-dataset/BRIGHTER-emotion-categories (rus)...")
    ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "rus")

    cols = ds["train"].column_names
    text_col  = next((c for c in cols if "text" in c.lower()), cols[0])
    emo_cols  = [c for c in cols if c.lower() in EKMAN_LABEL_NAMES]

    if emo_cols:
        def _to_ekman_multi(ex):
            scores = {c: float(ex.get(c, 0) or 0) for c in emo_cols}
            best   = max(scores, key=scores.get)
            label  = EKMAN_LABEL2ID.get(best, EKMAN_LABEL2ID["neutral"]) if scores[best] > 0 else EKMAN_LABEL2ID["neutral"]
            return {"text": ex[text_col], "label": label}
        ds = ds.map(_to_ekman_multi, remove_columns=cols)
    else:
        lc = next((c for c in cols if "label" in c.lower() or "emotion" in c.lower()), None)
        if not lc:
            raise ValueError(f"Cannot find emotion columns in BRIGHTER: {cols}")
        def _to_ekman_single(ex):
            raw = str(ex.get(lc, "neutral")).strip().lower()
            return {"text": ex[text_col], "label": EKMAN_LABEL2ID.get(raw, EKMAN_LABEL2ID["neutral"])}
        ds = ds.map(_to_ekman_single, remove_columns=cols)

    ds = _normalize_splits(ds)
    print(f"  → {sum(len(ds[s]) for s in ds):,} examples (BRIGHTER, native RU, 6 classes)")
    return ds


def load_dusha(
    max_samples: Optional[int] = None,
    local_dir: Optional[str] = None,
) -> DatasetDict:
    """
    SberDevices/Dusha — ~300k Russian speech transcripts (bi-modal corpus, Interspeech 2023).
    4 emotion classes: angry→anger, happy/sad→joy/sadness, neutral→neutral.

    Load priority:
      1. local_dir   — явно указанная папка с CSV/Parquet файлами
      2. Kaggle      — автопоиск в /kaggle/input/**/*dusha*/ и /kaggle/input/*dusha*/
      3. data/dusha/ — локальная разработка (gitignored)
      4. HuggingFace — SberDevices/Dusha (streaming, чтобы не качать аудио)

    На Kaggle: добавь датасет как "dusha" и файлы окажутся в /kaggle/input/dusha/.
    Audio колонки игнорируются; используются только текстовые транскрипты.
    max_samples ограничивает число примеров (None = без ограничений).
    """
    DUSHA_TO_EKMAN: Dict[str, str] = {
        "angry":      "anger",
        "anger":      "anger",
        "sad":        "sadness",
        "sadness":    "sadness",
        "neutral":    "neutral",
        # Dusha crowd labels use "happy" (not "positive")
        "happy":      "joy",
        "happiness":  "joy",
        "positive":   "joy",
        "enthusiasm": "joy",
        "joy":        "joy",
    }

    TEXT_COL_CANDIDATES  = ("raw_text", "transcription", "text", "sentence", "utterance", "phrase", "speaker_text")
    # Dusha: use aggregated 'emotion' column, not individual annotator columns
    LABEL_COL_CANDIDATES = ("emotion", "label", "tag", "sentiment", "category")

    def _find_col(cols, candidates):
        cols_lower = {c.lower(): c for c in cols}
        for name in candidates:
            if name in cols_lower:
                return cols_lower[name]
        # partial match fallback
        for cand in candidates:
            for c in cols:
                if cand in c.lower():
                    return c
        return None

    def _find_local_dirs() -> List[str]:
        """Return candidate local directories that might contain Dusha files."""
        candidates: List[str] = []
        if local_dir:
            candidates.append(local_dir)
        # Hardcoded Kaggle path for sber-dusha-crowd uploaded by inexyy
        _HARDCODED = "/kaggle/input/datasets/inexyy/sber-dusha-crowd"
        if os.path.isdir(_HARDCODED):
            candidates.append(_HARDCODED)
        # Kaggle: datasets can be nested several levels deep, e.g.
        #   /kaggle/input/datasets/<user>/sber-dusha-crowd/sber-dusha-crowd/
        if os.path.isdir("/kaggle/input"):
            import glob as _glob
            # Any directory anywhere under /kaggle/input whose name contains 'dusha'
            for d in sorted(_glob.glob("/kaggle/input/**/*dusha*/", recursive=True)):
                candidates.append(d)
            # Any TSV/CSV/Parquet file whose parent path contains 'dusha'
            for f in _glob.glob("/kaggle/input/**/*.tsv", recursive=True):
                if "dusha" in f.lower():
                    candidates.append(os.path.dirname(f))
            for ext in ("*.csv", "*.parquet"):
                for f in _glob.glob(f"/kaggle/input/**/{ext}", recursive=True):
                    if "dusha" in f.lower():
                        candidates.append(os.path.dirname(f))
        # Local development
        for rel in ("data/dusha", "../data/dusha"):
            p = os.path.abspath(rel)
            if os.path.isdir(p):
                candidates.append(p)
        return list(dict.fromkeys(candidates))  # deduplicate, preserve order

    def _load_local_dir(dirpath: str) -> Optional[DatasetDict]:
        """Load CSV/Parquet files from a local directory."""
        import glob as _glob
        files = (
            _glob.glob(os.path.join(dirpath, "**", "*.parquet"), recursive=True)
            + _glob.glob(os.path.join(dirpath, "**", "*.csv"),     recursive=True)
            + _glob.glob(os.path.join(dirpath, "**", "*.tsv"),     recursive=True)
        )
        if not files:
            return None

        parts: List[pd.DataFrame] = []
        for fpath in sorted(files):
            try:
                if fpath.endswith(".parquet"):
                    df_f = pd.read_parquet(fpath)
                elif fpath.endswith(".tsv"):
                    df_f = pd.read_csv(fpath, sep="\t")
                else:
                    df_f = pd.read_csv(fpath)
                parts.append(df_f)
                print(f"    read {os.path.basename(fpath)}: {len(df_f):,} rows, cols={list(df_f.columns)}")
            except Exception as e:
                print(f"    skip {os.path.basename(fpath)}: {e}")

        if not parts:
            return None

        df = pd.concat(parts, ignore_index=True)
        cols = list(df.columns)

        # Sber Dusha crowd files: keep only rows where annotator and speaker agree.
        # annotator_emo == speaker_emo means both perceived the same emotion.
        if "annotator_emo" in df.columns and "speaker_emo" in df.columns:
            before = len(df)
            df = df[df["annotator_emo"] == df["speaker_emo"]]
            print(f"    annotator_emo == speaker_emo: {before:,} → {len(df):,} rows ({len(df)/before*100:.0f}% kept)")
            text_col  = "speaker_text" if "speaker_text" in df.columns else _find_col(cols, TEXT_COL_CANDIDATES)
            label_col = "annotator_emo"
        else:
            text_col  = _find_col(cols, TEXT_COL_CANDIDATES)
            label_col = _find_col(cols, LABEL_COL_CANDIDATES)

        if not text_col or not label_col:
            print(f"    ✗ local: cannot find text/label columns. cols={cols}")
            return None

        print(f"    columns → text='{text_col}', label='{label_col}'")

        df["text"]  = df[text_col].astype(str).str.strip()
        df["label"] = df[label_col].astype(str).str.strip().str.lower().map(
            lambda x: EKMAN_LABEL2ID.get(DUSHA_TO_EKMAN.get(x, ""), -1)
        )
        df = df[["text", "label"]].query("label >= 0 and text.str.len() > 3")
        df["label"] = df["label"].astype(int)

        if df.empty:
            print("    ✗ local: 0 valid examples after label mapping")
            return None

        # Cap per-class
        per_class_n = (max_samples // 4) if max_samples else None
        if per_class_n:
            df = (df.groupby("label", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per_class_n), random_state=42)))
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df, tmp = train_test_split(df, test_size=0.20, random_state=42, stratify=df["label"])
        val_df, test_df = train_test_split(tmp, test_size=0.50, random_state=42, stratify=tmp["label"])
        return DatasetDict({
            "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
        })

    def _load_streaming(dataset_id, config, per_class_limit):
        """
        Load via streaming to avoid downloading audio files.
        Reads only text + label fields from each shard.
        """
        kwargs: dict = {"streaming": True}
        if config:
            kwargs["name"] = config

        raw_iter = load_dataset(dataset_id, **kwargs)

        split0   = list(raw_iter.keys())[0]
        first_ex = next(iter(raw_iter[split0].take(1)))
        cols     = list(first_ex.keys())
        text_col  = _find_col(cols, TEXT_COL_CANDIDATES)
        label_col = _find_col(cols, LABEL_COL_CANDIDATES)

        if not text_col:
            raise ValueError(f"No text column in {dataset_id}. Columns: {cols}")
        if not label_col:
            raise ValueError(f"No label column in {dataset_id}. Columns: {cols}")
        print(f"    columns → text='{text_col}', label='{label_col}'")

        result: dict = {}
        for split_name, split_iter in raw_iter.items():
            records: list = []
            per_class: dict = {}
            for ex in split_iter:
                raw_label = str(ex.get(label_col, "")).strip().lower()
                ekman = DUSHA_TO_EKMAN.get(raw_label)
                text  = str(ex.get(text_col) or "").strip()
                if ekman and len(text) > 3:
                    lid = EKMAN_LABEL2ID[ekman]
                    if per_class_limit is None or per_class.get(lid, 0) < per_class_limit:
                        records.append({"text": text, "label": lid})
                        per_class[lid] = per_class.get(lid, 0) + 1
                # Early exit when all 4 classes are saturated
                if per_class_limit and len(per_class) >= 4 and all(
                    per_class.get(lid, 0) >= per_class_limit
                    for lid in per_class
                ):
                    break
            result[split_name] = Dataset.from_list(records)
        return DatasetDict(result)

    def _load_normal(dataset_id, config):
        kwargs: dict = {}
        if config:
            kwargs["name"] = config
        raw = load_dataset(dataset_id, **kwargs)

        split0 = list(raw.keys())[0]
        cols   = raw[split0].column_names
        text_col  = _find_col(cols, TEXT_COL_CANDIDATES)
        label_col = _find_col(cols, LABEL_COL_CANDIDATES)

        if not text_col:
            raise ValueError(f"No text column in {dataset_id}. Columns: {cols}")
        if not label_col:
            raise ValueError(f"No label column in {dataset_id}. Columns: {cols}")
        print(f"    columns → text='{text_col}', label='{label_col}'")

        keep = {text_col, label_col}
        remove_cols = [c for c in cols if c not in keep]

        def _convert(ex):
            raw_label = str(ex[label_col]).strip().lower()
            ekman = DUSHA_TO_EKMAN.get(raw_label)
            return {"text": str(ex[text_col] or "").strip(), "label": EKMAN_LABEL2ID[ekman] if ekman else -1}

        return raw.map(_convert, remove_columns=remove_cols)

    per_class_limit = (max_samples // 4) if max_samples else None

    ds = None

    # 1. Local files (Kaggle /kaggle/input or data/dusha/)
    for dirpath in _find_local_dirs():
        print(f"  Trying local Dusha: {dirpath}")
        try:
            ds = _load_local_dir(dirpath)
            if ds is not None:
                print(f"    ✓ loaded from local: {dirpath}")
                break
        except Exception as e:
            print(f"    ✗ local {dirpath}: {e}")

    # 2. HuggingFace — streaming first (avoids audio download), then normal
    if ds is None:
        hf_sources = [
            ("SberDevices/Dusha", "crowd"),
            ("SberDevices/Dusha", None),
        ]
        for dataset_id, config in hf_sources:
            tag = f"{dataset_id}" + (f" ({config})" if config else "")
            for loader_name, loader_fn, loader_arg in [
                ("streaming", _load_streaming, per_class_limit),
                ("normal",    _load_normal,    None),
            ]:
                try:
                    print(f"  Loading {tag} [{loader_name}]...")
                    ds = loader_fn(dataset_id, config, loader_arg) if loader_name == "streaming" \
                         else loader_fn(dataset_id, config)
                    print(f"    ✓ loaded from {tag} [{loader_name}]")
                    break
                except Exception as e:
                    print(f"    ✗ {tag} [{loader_name}]: {e}")
            if ds is not None:
                break

    if ds is None:
        raise RuntimeError(
            "Could not load Dusha from any source.\n"
            "На Kaggle: добавь датасет через 'Add Data' → 'dusha', "
            "файлы окажутся в /kaggle/input/dusha/.\n"
            "Локально: положи CSV/Parquet файлы в data/dusha/."
        )

    for split in list(ds.keys()):
        if len(ds[split]) == 0:
            continue
        # Filter bad labels (only HuggingFace normal load can produce label=-1)
        sample_labels = [ds[split][i]["label"] for i in range(min(10, len(ds[split])))]
        if any(l < 0 for l in sample_labels):
            ds[split] = ds[split].filter(lambda ex: ex["label"] >= 0 and len(ex["text"]) > 3)
        # Per-class cap for HuggingFace normal-loaded splits (local + streaming already capped)
        if max_samples and len(ds[split]) > max_samples:
            df = ds[split].to_pandas()
            per_class_n = max_samples // max(df["label"].nunique(), 1)
            capped = (df.groupby("label", group_keys=False)
                        .apply(lambda g: g.sample(min(len(g), per_class_n), random_state=42)))
            ds[split] = Dataset.from_pandas(
                capped.sample(frac=1, random_state=42).reset_index(drop=True)
            )

    ds = _normalize_splits(ds)
    total = sum(len(ds[s]) for s in ds)
    print(f"  → {total:,} examples (Dusha, native RU, 4 classes)")
    return ds


def load_brighter(data_dir: str) -> DatasetDict:
    """
    BRIGHTER (SemEval-2025 Task 11) — Russian subset.
    Полное покрытие 6 Ekman-эмоций, размечен через Яндекс Толоку.

    Требует ручной загрузки:
      1. Зарегистрируйся на https://www.codabench.org/competitions/3863/
      2. Скачай архив и распакуй в папку data_dir
      3. Передай путь к папке с CSV-файлами в этот функцию

    Формат: multi-label с интенсивностью 0-3 для каждой эмоции.
    Конвертируется в single-label Ekman по максимальной интенсивности.
    """
    import glob

    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    ru_files  = [f for f in csv_files if "rus" in os.path.basename(f).lower() or "ru" in os.path.basename(f).lower()]
    target    = ru_files[0] if ru_files else (csv_files[0] if csv_files else None)

    if not target:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Download BRIGHTER from https://www.codabench.org/competitions/3863/"
        )

    df = pd.read_csv(target)
    emotion_cols = [c for c in df.columns if c.lower() in EKMAN_LABEL_NAMES]
    text_col     = next((c for c in df.columns if "text" in c.lower()), df.columns[0])

    if not emotion_cols:
        raise ValueError(f"Cannot find emotion columns in {df.columns.tolist()}")

    def _row_to_ekman(row):
        intensities = {col: float(row[col]) for col in emotion_cols}
        best = max(intensities, key=intensities.get)
        if intensities[best] == 0:
            return EKMAN_LABEL2ID["neutral"]
        return EKMAN_LABEL2ID.get(best.lower(), EKMAN_LABEL2ID["neutral"])

    df["label"] = df.apply(_row_to_ekman, axis=1)
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    print(f"BRIGHTER (RU): {len(df):,} examples (native Ekman annotation)")

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    train_df, val_df  = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df["label"])

    return _normalize_splits(DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    }))


# ── LLM-annotated data loader ──────────────────────────────────────

def load_llm_annotated(
    cache_paths: List[str],
    test_size:   float = 0.15,
    val_size:    float = 0.15,
    seed:        int   = 42,
) -> DatasetDict:
    """
    Load LLM-annotated texts from JSONL cache files produced by LLMAnnotator.

    Each JSONL line: {"text": "...", "label": "joy"}  (string label)
    Multiple files are concatenated (e.g., rureviews + rusentitweet caches).

    Parameters
    ----------
    cache_paths : list of paths to .jsonl annotation cache files
    """
    records: List[dict] = []
    for path in cache_paths:
        path = str(path)
        if not os.path.exists(path):
            print(f"  WARNING: annotation cache not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    text  = str(rec.get("text", "")).strip()
                    label_str = str(rec.get("label", "neutral")).strip().lower()
                    label = EKMAN_LABEL2ID.get(label_str, -1)
                    if label >= 0 and len(text) > 3:
                        records.append({"text": text, "label": label})
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  Loaded {len(records):,} entries from {path}")

    if not records:
        raise RuntimeError(f"No valid annotated examples found in: {cache_paths}")

    import json as _json_mod  # already imported at module level but make scope clear
    df = pd.DataFrame(records)
    df["label"] = df["label"].astype(int)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print(f"  → {len(df):,} unique LLM-annotated examples")

    min_class = df["label"].value_counts().min()
    can_stratify = min_class >= 3
    train_df, tmp = train_test_split(
        df, test_size=test_size, random_state=seed,
        stratify=df["label"] if can_stratify else None,
    )
    min_tmp = tmp["label"].value_counts().min() if can_stratify else 0
    val_df, test_df = train_test_split(
        tmp, test_size=0.50, random_state=seed,
        stratify=tmp["label"] if min_tmp >= 2 else None,
    )

    return DatasetDict({
        "train":      Dataset.from_pandas(train_df.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
        "test":       Dataset.from_pandas(test_df.reset_index(drop=True)),
    })


# ── CSV / Kaggle helpers ────────────────────────────────────────────

def load_from_csv(
    filepath: str,
    text_col: str = "text",
    label_col: str = "label",
    label_names: Optional[List[str]] = None,
    test_size: float = 0.15,
    val_size:  float = 0.15,
    seed: int = 42,
) -> Tuple[DatasetDict, List[str]]:
    df = pd.read_csv(filepath)

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


# ── Main entry point ────────────────────────────────────────────────

def load_data(
    csv_path:           Optional[str] = None,
    use_all:            bool = True,
    include_sentiment:  bool = False,
    brighter_dir:       Optional[str] = None,
    use_brighter_hf:    bool = False,
    use_dusha:          bool = False,
    use_aniemore:       bool = False,
    seed:               int  = 42,
) -> Tuple[DatasetDict, List[str], Dict]:
    """
    Load and (optionally) merge Russian emotion datasets.

    Параметры:
      use_all           — объединить все 3 emotion-датасета (HF)
      include_sentiment — добавить RuReviews + RuSentiTweet (sentiment→Ekman, приблизительно)
      brighter_dir      — путь к распакованному BRIGHTER (если скачан вручную)
      use_brighter_hf   — загрузить BRIGHTER с HuggingFace (brighter-dataset/BRIGHTER-emotion-categories)
      use_dusha         — добавить Dusha (~300k нативных RU транскриптов, SberDevices/Dusha)
      use_aniemore      — добавить Aniemore/resd (~4.5k нативных RU, 7 классов включая disgust)

    Приоритет источников:
      1. csv_path  (пользовательский CSV)
      2. /kaggle/input  (автопоиск CSV)
      3. HuggingFace + опциональные источники
    """
    # User-provided CSV
    if csv_path and os.path.exists(csv_path):
        ds, label_names = load_from_csv(csv_path, seed=seed)
        return ds, label_names, _compute_stats(ds, label_names)

    # Kaggle CSV auto-detect
    if os.path.exists("/kaggle/input"):
        csv_files = []
        for root, _, files in os.walk("/kaggle/input"):
            for f in files:
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))
        for path in sorted(csv_files, key=lambda p: any(k in p.lower() for k in ("emotion", "sentiment", "train")), reverse=True):
            try:
                print(f"Trying Kaggle CSV: {path}")
                ds, label_names = load_from_csv(path, seed=seed)
                return ds, label_names, _compute_stats(ds, label_names)
            except Exception as e:
                print(f"  Skipped: {e}")

    # HuggingFace emotion datasets
    all_ds: Dict[str, DatasetDict] = {}
    loaders = [
        ("ru_go_emotions",    load_ru_go_emotions),
        ("cedr",              load_cedr),
        ("ru_izard_emotions", load_ru_izard_emotions),
    ] if use_all else [("ru_go_emotions", load_ru_go_emotions)]

    for name, loader in loaders:
        try:
            all_ds[name] = loader()
        except Exception as e:
            print(f"  WARNING: could not load {name}: {e}")

    if not all_ds:
        raise RuntimeError("Could not load any dataset.")

    if len(all_ds) == 1:
        ds = list(all_ds.values())[0]
    else:
        ds = merge_datasets(all_ds, test_size=0.15, val_size=0.15, seed=seed)

    # Optional: BRIGHTER manual download
    if brighter_dir and os.path.isdir(brighter_dir):
        try:
            brighter_ds = load_brighter(brighter_dir)
            ds = merge_datasets(
                {"emotion": ds, "brighter": brighter_ds},
                test_size=0.15, val_size=0.15, seed=seed,
            )
        except Exception as e:
            print(f"  WARNING: BRIGHTER (local) load failed: {e}")

    # Optional: BRIGHTER via HuggingFace mirror
    if use_brighter_hf:
        try:
            brighter_ds = load_brighter_hf()
            ds = merge_datasets(
                {"emotion": ds, "brighter_hf": brighter_ds},
                test_size=0.15, val_size=0.15, seed=seed,
            )
        except Exception as e:
            print(f"  WARNING: BRIGHTER-HF load failed: {e}")

    # Optional: Dusha (~300k native RU transcripts)
    if use_dusha:
        try:
            dusha_ds = load_dusha()
            ds = merge_datasets(
                {"emotion": ds, "dusha": dusha_ds},
                test_size=0.15, val_size=0.15, seed=seed,
            )
        except Exception as e:
            print(f"  WARNING: Dusha load failed: {e}")

    # Optional: Aniemore/resd (~4.5k native RU, 7 classes)
    if use_aniemore:
        try:
            aniemore_ds = load_aniemore_resd()
            ds = merge_datasets(
                {"emotion": ds, "aniemore": aniemore_ds},
                test_size=0.15, val_size=0.15, seed=seed,
            )
        except Exception as e:
            print(f"  WARNING: Aniemore/resd load failed: {e}")

    # Optional: sentiment datasets (approximate Ekman mapping)
    if include_sentiment:
        sentiment_ds: Dict[str, DatasetDict] = {"emotion": ds}
        for name, loader in [("rureviews", load_rureviews), ("rusentitweet", load_rusentitweet)]:
            try:
                sentiment_ds[name] = loader()
            except Exception as e:
                print(f"  WARNING: {name} load failed: {e}")
        if len(sentiment_ds) > 1:
            print("\nNote: sentiment datasets mapped to Ekman approximately (pos→joy, neg→sadness, neu→neutral)")
            ds = merge_datasets(sentiment_ds, test_size=0.15, val_size=0.15, seed=seed)

    return ds, EKMAN_LABEL_NAMES, _compute_stats(ds, EKMAN_LABEL_NAMES)


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
