import os
import numpy as np
import torch
from typing import List, Optional, Dict
from datasets import Dataset, DatasetDict
import pandas as pd


class TextAugmenter:
    """
    Text augmentation for Russian emotion data.

    Methods:
      'paraphrase'     — cointegrated/rut5-base-paraphraser (diverse beam search)
      'backtranslation'— Helsinki-NLP RU→EN→RU (different vocabulary via pivot)
      'both'           — run both, combine results
    """

    def __init__(self, method: str = "paraphrase", device: Optional[str] = None):
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._para_model = None
        self._para_tok = None
        self._ru2en_model = None
        self._ru2en_tok = None
        self._en2ru_model = None
        self._en2ru_tok = None

    # ── Paraphraser ────────────────────────────────────────────────

    def _load_paraphraser(self):
        if self._para_model is not None:
            return
        from transformers import T5ForConditionalGeneration, AutoTokenizer
        print("  Loading cointegrated/rut5-base-paraphraser...")
        self._para_tok = AutoTokenizer.from_pretrained("cointegrated/rut5-base-paraphraser")
        self._para_model = (
            T5ForConditionalGeneration
            .from_pretrained("cointegrated/rut5-base-paraphraser")
            .to(self.device)
            .eval()
        )

    def paraphrase_batch(
        self,
        texts: List[str],
        n_variants: int = 3,
        max_length: int = 128,
        batch_size: int = 8,
    ) -> List[List[str]]:
        """Return up to n_variants paraphrases per input text."""
        self._load_paraphraser()
        all_results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                inputs = self._para_tok(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                ).to(self.device)

                outputs = self._para_model.generate(
                    **inputs,
                    num_return_sequences=n_variants,
                    num_beam_groups=n_variants,
                    num_beams=n_variants,
                    diversity_penalty=2.0,
                    repetition_penalty=3.0,
                    max_length=max_length,
                )

            decoded = self._para_tok.batch_decode(outputs, skip_special_tokens=True)
            for j, orig in enumerate(batch):
                variants = decoded[j * n_variants : (j + 1) * n_variants]
                # keep only those that differ from the original
                variants = [v for v in variants if v.strip().lower() != orig.strip().lower() and len(v.strip()) > 5]
                all_results.append(variants)

        return all_results

    # ── Back-translation ───────────────────────────────────────────

    def _load_backtranslation(self):
        if self._ru2en_model is not None:
            return
        from transformers import MarianMTModel, MarianTokenizer
        print("  Loading Helsinki-NLP/opus-mt-ru-en and opus-mt-en-ru...")
        self._ru2en_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-en")
        self._ru2en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-en").to(self.device).eval()
        self._en2ru_tok   = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
        self._en2ru_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ru").to(self.device).eval()

    def _translate_batch(self, texts, model, tokenizer, max_length=128, batch_size=16):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                outputs = model.generate(**inputs, max_length=max_length)
            results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        return results

    def backtranslate_batch(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """RU → EN → RU back-translation for lexical diversity."""
        self._load_backtranslation()
        en = self._translate_batch(texts, self._ru2en_model, self._ru2en_tok, batch_size=batch_size)
        ru = self._translate_batch(en,    self._en2ru_model, self._en2ru_tok, batch_size=batch_size)
        return ru


def augment_rare_classes(
    dataset: DatasetDict,
    label_names: List[str],
    target_per_class: int = 2_000,
    method: str = "paraphrase",     # 'paraphrase' | 'backtranslation' | 'both'
    n_variants: int = 3,
    batch_size: int = 8,
    device: Optional[str] = None,
    seed: int = 42,
    min_text_len: int = 10,
) -> DatasetDict:
    """
    Augment rare classes in the training split to reach target_per_class.
    Validation and test splits are NOT touched.

    Args:
        dataset:          DatasetDict with 'train' / 'validation' / 'test' splits.
        label_names:      List of class names matching integer label ids.
        target_per_class: Augment classes below this count.
        method:           'paraphrase', 'backtranslation', or 'both'.
        n_variants:       Paraphrases generated per source sentence (paraphrase method).
        batch_size:       Inference batch size.
        device:           'cuda' or 'cpu' (auto-detected if None).
        seed:             Random seed for shuffling.
        min_text_len:     Minimum character length to keep an augmented example.
    """
    augmenter = TextAugmenter(method=method, device=device)
    train_df  = dataset["train"].to_pandas()
    new_rows: List[Dict] = []

    for label_id, label_name in enumerate(label_names):
        class_df = train_df[train_df["label"] == label_id].copy()
        current  = len(class_df)

        if current >= target_per_class:
            print(f"  {label_name:<12}: {current:>5}  ≥ {target_per_class} — skip")
            continue

        needed  = target_per_class - current
        texts   = class_df["text"].tolist()
        # How many paraphrases per original sentence needed?
        n_per   = max(1, int(np.ceil(needed / max(current, 1))))
        print(f"  {label_name:<12}: {current:>5}  → need +{needed}  ({n_per} variants/text)")

        generated: List[str] = []

        if method in ("paraphrase", "both"):
            paraphrases = augmenter.paraphrase_batch(texts, n_variants=n_per, batch_size=batch_size)
            for variants in paraphrases:
                generated.extend(variants)

        if method in ("backtranslation", "both"):
            bt = augmenter.backtranslate_batch(texts, batch_size=batch_size)
            generated.extend(bt)

        # Filter quality and deduplicate
        seen = set(class_df["text"].str.lower().tolist())
        filtered = []
        for text in generated:
            t = text.strip()
            if len(t) >= min_text_len and t.lower() not in seen:
                seen.add(t.lower())
                filtered.append(t)

        # Take only what's needed
        for text in filtered[:needed]:
            new_rows.append({"text": text, "label": label_id})

        added = min(len(filtered), needed)
        print(f"    → added {added} augmented examples (generated {len(filtered)} valid)")

    if not new_rows:
        print("No augmentation was needed.")
        return dataset

    aug_df   = pd.DataFrame(new_rows)
    train_df = train_df[["text", "label"]] if "source" in train_df.columns else train_df
    keep_cols = [c for c in ["text", "label"] if c in train_df.columns]
    full_df  = (
        pd.concat([train_df[keep_cols], aug_df], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print(f"\nTotal added: {len(new_rows)} examples")
    print("New training distribution:")
    for lid, cnt in full_df["label"].value_counts().sort_index().items():
        print(f"  {label_names[lid]:<12}: {cnt:>6,}")

    return DatasetDict({
        "train":      Dataset.from_pandas(full_df),
        "validation": dataset["validation"],
        "test":       dataset["test"],
    })
