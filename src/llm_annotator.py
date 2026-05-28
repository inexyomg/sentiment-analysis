"""
LLM-based emotion annotator using Claude API.

Annotates arbitrary Russian text with Ekman 7-class labels.
Designed for re-labelling coarse sentiment datasets (rureviews, rusentitweet,
and any Russian text corpus) at higher quality than rule-based pos→joy mapping.

Usage:
    from src.llm_annotator import LLMAnnotator, annotate_dataset

    ann = LLMAnnotator(api_key="sk-ant-...")
    ann.annotate_dataset(
        texts=df["text"].tolist(),
        cache_path="data/annotated/rureviews_ekman.jsonl",
    )
"""

import os
import json
import time
import random
from typing import List, Optional, Dict, Tuple
from pathlib import Path

from datasets import Dataset, DatasetDict

from .data_loader import EKMAN_LABEL_NAMES, EKMAN_LABEL2ID


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Ты — система разметки эмоций. Для каждого русского текста определи \
ОДНУ главную эмоцию по таксономии Экмана.

Допустимые метки (строго одно из):
  anger    — злость, раздражение, возмущение
  disgust  — отвращение, омерзение, брезгливость
  fear     — страх, тревога, опасение
  joy      — радость, счастье, восторг, энтузиазм
  sadness  — грусть, печаль, тоска, разочарование
  surprise — удивление, изумление, шок
  neutral  — нейтральный тон, факты, без эмоций

Отвечай ТОЛЬКО JSON-массивом строк, по одному элементу на текст, \
в том же порядке. Никакого лишнего текста."""

_USER_TEMPLATE = """\
Разметь следующие тексты (один ответ на каждый):

{numbered_texts}

Ответ — JSON-массив из {n} строк, например: ["joy","neutral","anger"]"""


# ── Core annotator ────────────────────────────────────────────────────────────

class LLMAnnotator:
    """
    Batch emotion annotator backed by Claude API.

    Parameters
    ----------
    api_key   : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    model     : Claude model ID to use; haiku is fast + cheap for annotation
    batch_size: texts per API call (50 is a good balance of speed vs reliability)
    max_retries: retries per batch on transient errors
    """

    VALID_LABELS = set(EKMAN_LABEL_NAMES)

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str           = "claude-haiku-4-5-20251001",
        batch_size:  int           = 50,
        max_retries: int           = 3,
    ):
        import anthropic  # deferred import — only needed at runtime
        self.client     = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model      = model
        self.batch_size = batch_size
        self.max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    def annotate(
        self,
        texts:      List[str],
        cache_path: Optional[str] = None,
        verbose:    bool          = True,
    ) -> List[str]:
        """
        Annotate a list of texts with Ekman labels.
        Loads from / saves to cache_path (JSONL) to avoid repeated API calls.

        Returns a list of label strings (same length as texts).
        """
        cache_path = Path(cache_path) if cache_path else None
        cache: Dict[str, str] = {}

        # Load existing cache
        if cache_path and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    cache[rec["text"]] = rec["label"]
            if verbose:
                print(f"  Cache: {len(cache):,} entries loaded from {cache_path}")

        missing_idx = [i for i, t in enumerate(texts) if t not in cache]
        if verbose:
            print(f"  Annotating {len(missing_idx):,} / {len(texts):,} texts"
                  f" (model={self.model}, batch={self.batch_size})")

        # Process missing texts in batches
        if missing_idx:
            writer = open(cache_path, "a", encoding="utf-8") if cache_path else None
            cache_path.parent.mkdir(parents=True, exist_ok=True) if cache_path else None

            batches = [missing_idx[i: i + self.batch_size]
                       for i in range(0, len(missing_idx), self.batch_size)]
            n_done = 0
            for b_idx, batch_indices in enumerate(batches):
                batch_texts = [texts[i] for i in batch_indices]
                labels = self._annotate_batch(batch_texts)

                for idx, label in zip(batch_indices, labels):
                    cache[texts[idx]] = label
                    if writer:
                        writer.write(json.dumps({"text": texts[idx], "label": label},
                                                ensure_ascii=False) + "\n")

                n_done += len(batch_indices)
                if verbose and (b_idx % 10 == 0 or b_idx == len(batches) - 1):
                    pct = n_done / len(missing_idx) * 100
                    print(f"  [{pct:5.1f}%] batch {b_idx+1}/{len(batches)}"
                          f" — {n_done:,}/{len(missing_idx):,} done")

            if writer:
                writer.close()

        return [cache.get(t, "neutral") for t in texts]

    def annotate_dataframe(
        self,
        df,
        text_col:   str           = "text",
        label_col:  str           = "label",
        cache_path: Optional[str] = None,
        verbose:    bool          = True,
    ):
        """
        Annotate a pandas DataFrame in-place.
        Adds integer Ekman label to label_col.
        Returns the modified DataFrame (with rows whose label is -1 dropped).
        """
        texts  = df[text_col].astype(str).tolist()
        labels = self.annotate(texts, cache_path=cache_path, verbose=verbose)
        df = df.copy()
        df[label_col] = [EKMAN_LABEL2ID.get(l, -1) for l in labels]
        return df[df[label_col] >= 0].reset_index(drop=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _annotate_batch(self, texts: List[str]) -> List[str]:
        numbered = "\n".join(f'{i+1}. "{t[:400]}"' for i, t in enumerate(texts))
        prompt   = _USER_TEMPLATE.format(numbered_texts=numbered, n=len(texts))

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=len(texts) * 12,  # ~10 chars per label + buffer
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                labels = self._parse_response(raw, expected=len(texts))
                return labels
            except Exception as e:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"  Batch error (attempt {attempt+1}/{self.max_retries}): {e} — retry in {wait:.1f}s")
                time.sleep(wait)

        # Fallback: neutral for all
        print(f"  WARNING: batch failed after {self.max_retries} attempts → neutral")
        return ["neutral"] * len(texts)

    def _parse_response(self, raw: str, expected: int) -> List[str]:
        """Parse JSON array from model response; repair common issues."""
        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == expected:
                return [l.strip().lower() if l.strip().lower() in self.VALID_LABELS
                        else "neutral"
                        for l in parsed]
        except json.JSONDecodeError:
            pass

        # Fallback: extract quoted words
        import re
        found = re.findall(r'"(\w+)"', raw)
        valid = [w.lower() for w in found if w.lower() in self.VALID_LABELS]
        if len(valid) == expected:
            return valid

        # Last resort: repeat neutral
        return ["neutral"] * expected

    # ── Cost estimation ───────────────────────────────────────────────────────

    @staticmethod
    def estimate_cost(n_texts: int, avg_chars: int = 120, batch_size: int = 50) -> Dict:
        """Rough cost estimate for annotating n_texts."""
        # haiku-4-5 pricing: $0.80/M input, $4.00/M output tokens
        chars_per_token = 4
        input_per_batch = (
            len(_SYSTEM_PROMPT) / chars_per_token
            + batch_size * avg_chars / chars_per_token
            + 100  # prompt overhead
        )
        output_per_batch = batch_size * 3  # ~3 tokens per label

        n_batches     = (n_texts + batch_size - 1) // batch_size
        total_input   = n_batches * input_per_batch
        total_output  = n_batches * output_per_batch
        cost_usd      = total_input / 1e6 * 0.80 + total_output / 1e6 * 4.00

        return {
            "n_texts":      n_texts,
            "n_batches":    n_batches,
            "input_tokens": int(total_input),
            "output_tokens": int(total_output),
            "cost_usd":     round(cost_usd, 2),
        }


# ── Convenience wrapper ───────────────────────────────────────────────────────

def annotate_dataset(
    df,
    text_col:    str           = "text",
    cache_path:  Optional[str] = None,
    api_key:     Optional[str] = None,
    model:       str           = "claude-haiku-4-5-20251001",
    batch_size:  int           = 50,
    verbose:     bool          = True,
    dry_run:     bool          = False,
) -> "pd.DataFrame":
    """
    Re-annotate a DataFrame with Ekman emotions using Claude.

    Parameters
    ----------
    df         : DataFrame with a text column
    text_col   : name of the text column
    cache_path : path to JSONL cache file (avoids re-calling API)
    api_key    : Anthropic API key (or set ANTHROPIC_API_KEY env var)
    model      : Claude model (haiku recommended for cost)
    batch_size : texts per API call
    verbose    : print progress
    dry_run    : estimate cost only, don't call API

    Returns
    -------
    DataFrame with 'text' and 'label' (int) columns.
    """
    import pandas as pd

    texts = df[text_col].astype(str).tolist()

    if dry_run or verbose:
        est = LLMAnnotator.estimate_cost(len(texts), batch_size=batch_size)
        print(f"  Cost estimate: {est['n_texts']:,} texts → "
              f"~{est['input_tokens']:,} input + {est['output_tokens']:,} output tokens "
              f"≈ ${est['cost_usd']:.2f} (haiku)")
        if dry_run:
            return df

    ann = LLMAnnotator(api_key=api_key, model=model, batch_size=batch_size)
    return ann.annotate_dataframe(df, text_col=text_col, cache_path=cache_path, verbose=verbose)
