import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Dict, List


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
    Columns: ru_text (Russian), text (English original), labels (list[int] 0-27), id.
    config='simplified' (~54k) or 'raw' (~211k, noisier but more fear/disgust examples).
    28 multi-label emotions → Ekman 7 single-label.
    """
    print(f"Loading seara/ru_go_emotions ({config})...")
    ds = load_dataset("seara/ru_go_emotions", config)

    cols = ds["train"].column_names
    # Russian text is in 'ru_text'; 'text' is the English original — we must use ru_text
    text_col  = "ru_text" if "ru_text" in cols else next(
        (c for c in cols if "ru" in c.lower() and "text" in c.lower()), "text"
    )
    label_col = next((c for c in cols if "label" in c.lower()), None)
    if not label_col:
        raise ValueError(f"Cannot find labels column in seara/ru_go_emotions. Columns: {cols}")

    _idx2ekman = {i: EMOTION_TO_EKMAN.get(n, "neutral") for i, n in GOEMOTION_IDX2NAME.items()}

    def _convert(ex):
        raw = ex.get(label_col)
        # labels is a list of ints [0-27]; handle edge case of single int too
        if isinstance(raw, (list, tuple)):
            ids = [int(x) for x in raw]
        elif raw is not None:
            ids = [int(raw)]
        else:
            ids = []
        return {"text": str(ex[text_col] or ""), "label": _multilabel_to_ekman(ids, _idx2ekman)}

    ds = ds.map(_convert, remove_columns=cols)
    ds = _normalize_splits(ds)
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


def _normalize_splits(ds: DatasetDict) -> DatasetDict:
    """Standardize split names and ensure train/validation/test exist."""
    for old, new in [("dev", "validation"), ("development", "validation"), ("valid", "validation")]:
        if old in ds and new not in ds:
            ds[new] = ds.pop(old)
    if "validation" not in ds and "test" in ds:
        ds["validation"] = ds["test"]
    return ds


# ── Multi-dataset merge ─────────────────────────────────────────────

def merge_datasets(
    datasets: Dict[str, DatasetDict],
    test_size: float = 0.15,
    val_size:  float = 0.15,
    seed:      int   = 42,
) -> DatasetDict:
    """
    Concatenate multiple DatasetDicts (each must have 'text' and 'label' columns)
    and re-split into train / validation / test, stratified by label.
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
    3-class sentiment (pos/neu/neg) → approximate Ekman mapping.
    Скачивается напрямую с GitHub (~11 MB CSV).
    """
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

    # Auto-detect columns
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
    5-class (pos/neu/neg/speech/skip) → 3 usable classes → Ekman.
    Скачивается с GitHub.
    """
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


def load_stage2_clean(
    use_cedr:         bool = True,
    use_brighter_hf:  bool = True,
    use_aniemore:     bool = True,
    seed:             int  = 42,
) -> DatasetDict:
    """
    Stage-2 corpus: только высококачественные нативные RU датасеты.
    Совместное покрытие всех 7 классов Ekman:
      CEDR:       anger, fear, joy, sadness, surprise  (нативный RU, ручная разметка)
      BRIGHTER:   anger, disgust, fear, joy, sadness, surprise  (Toloka)
      Aniemore:   anger, disgust, enthusiasm→joy, fear, neutral, sadness  (нативный RU)
    """
    sources: Dict[str, DatasetDict] = {}

    if use_cedr:
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


def load_dusha(max_samples: Optional[int] = 50_000) -> DatasetDict:
    """
    SberDevices/Dusha — ~300k Russian speech transcripts (bi-modal corpus, Interspeech 2023).
    4 emotion classes: angry→anger, sad→sadness, neutral→neutral, positive→joy.
    Tries config='crowd' first, then base dataset.
    Audio columns are ignored; only text transcripts are used.
    max_samples caps examples per class to avoid memory pressure (None = no limit).
    """
    DUSHA_TO_EKMAN: Dict[str, str] = {
        "angry":      "anger",
        "anger":      "anger",
        "sad":        "sadness",
        "sadness":    "sadness",
        "neutral":    "neutral",
        "positive":   "joy",
        "enthusiasm": "joy",
        "joy":        "joy",
    }

    TEXT_COL_CANDIDATES  = ("raw_text", "transcription", "text", "sentence", "utterance", "phrase")
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

    def _load_and_convert(dataset_id, config=None):
        kwargs = {"path": dataset_id}
        if config:
            kwargs["name"] = config
        raw = load_dataset(**kwargs)

        split0 = list(raw.keys())[0]
        cols = raw[split0].column_names
        text_col  = _find_col(cols, TEXT_COL_CANDIDATES)
        label_col = _find_col(cols, LABEL_COL_CANDIDATES)

        if not text_col:
            raise ValueError(f"No text column in {dataset_id}. Columns: {cols}")
        if not label_col:
            raise ValueError(f"No label column in {dataset_id}. Columns: {cols}")

        keep = {text_col, label_col}
        remove_cols = [c for c in cols if c not in keep]

        def _convert(ex):
            raw_label = str(ex[label_col]).strip().lower()
            ekman = DUSHA_TO_EKMAN.get(raw_label)
            return {"text": str(ex[text_col] or "").strip(), "label": EKMAN_LABEL2ID[ekman] if ekman else -1}

        return raw.map(_convert, remove_columns=remove_cols)

    # SberDevices/Dusha — try 'crowd' subset first, then base dataset
    sources = [
        ("SberDevices/Dusha", "crowd"),
        ("SberDevices/Dusha", None),
    ]

    ds = None
    for dataset_id, config in sources:
        try:
            label = f"{dataset_id}" + (f" ({config})" if config else "")
            print(f"  Loading {label}...")
            ds = _load_and_convert(dataset_id, config)
            print(f"    ✓ loaded from {label}")
            break
        except Exception as e:
            print(f"    ✗ {dataset_id}: {e}")

    if ds is None:
        raise RuntimeError("Could not load Dusha from any source.")

    for split in list(ds.keys()):
        ds[split] = ds[split].filter(lambda ex: ex["label"] >= 0 and len(ex["text"]) > 3)
        if max_samples and len(ds[split]) > max_samples:
            # cap per-class to keep class balance within Dusha
            import pandas as pd
            from datasets import Dataset
            df = ds[split].to_pandas()
            parts = []
            per_class = max_samples // max(df["label"].nunique(), 1)
            for lid in df["label"].unique():
                sub = df[df["label"] == lid]
                if len(sub) > per_class:
                    sub = sub.sample(per_class, random_state=42)
                parts.append(sub)
            df_capped = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
            ds[split] = Dataset.from_pandas(df_capped)

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
      use_dusha         — добавить Dusha (~300k нативных RU транскриптов, KELONMYOSA/dusha_emotion_audio)
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
