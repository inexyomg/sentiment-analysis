import re
import html
from typing import List, Optional

# ── Optional heavy deps ────────────────────────────────────────────────────
try:
    import pymorphy2
    _morph = pymorphy2.MorphAnalyzer()
    _PYMORPHY_AVAILABLE = True
except ImportError:
    _PYMORPHY_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    try:
        _RU_STOPWORDS = set(stopwords.words("russian"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _RU_STOPWORDS = set(stopwords.words("russian"))
    _NLTK_AVAILABLE = True
except ImportError:
    _RU_STOPWORDS = set()
    _NLTK_AVAILABLE = False


# ── Compiled patterns ──────────────────────────────────────────────────────
_URL_RE       = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE   = re.compile(r"@\w+")
_HASHTAG_RE   = re.compile(r"#(\w+)")
_HTML_TAG_RE  = re.compile(r"<[^>]+>")
_MULTI_WS_RE  = re.compile(r"\s+")
_TYPOGRAPHIC  = str.maketrans({
    "‘": "'",  "’": "'",   # curly single quotes
    "“": '"',  "”": '"',   # curly double quotes
    "…": "...",                  # ellipsis
    " ": " ",                   # non-breaking space
    "​": "",                    # zero-width space
    "﻿": "",                    # BOM
})
# Collapse absurd repetition: аааааааа → ааа  (4+ same letter → 3)
_REPEAT_LETTER_RE = re.compile(r"(\w)\1{3,}", re.UNICODE)
# Collapse repeated punctuation: !!! → !, ... stays as-is
_REPEAT_PUNCT_RE  = re.compile(r"([!?;:]){2,}")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
    remove_html: bool = True,
    normalize_unicode: bool = True,
    collapse_repeats: bool = True,
    lowercase: bool = False,
) -> str:
    """
    Light cleaning for transformer input.

    What we DO:
      - Decode HTML entities (&amp; → &, &lt; → <, etc.)
      - Strip HTML tags
      - Remove URLs and @mentions
      - Extract text from #hashtags (keep word, drop #)
      - Normalise typographic chars (curly quotes, ellipsis, NBSP, BOM)
      - Collapse absurd character repetition (аааааа → ааа, !!! → !)
      - Collapse whitespace runs

    What we DON'T do (and why):
      - Lemmatisation: BERT-family models are trained on live morphology and
        exploit it. Lemmatising removes that signal — typically drops F1 by 2-5%.
      - Stop word removal: transformers use attention over the full sequence;
        removing stop words breaks syntactic cues the model relies on.
      - Lowercase by default: ruBERT is cased; lowercasing loses NER signals.
      - Emoji removal: emojis are strong emotion signals and XLM-R/ruBERT
        handle them through their BPE vocabulary.
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entities → unicode chars  (&amp; &lt; etc.)
    text = html.unescape(text)

    # 2. Strip HTML tags
    if remove_html:
        text = _HTML_TAG_RE.sub(" ", text)

    # 3. URLs
    if remove_urls:
        text = _URL_RE.sub(" ", text)

    # 4. @mentions
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)

    # 5. #hashtags → word only
    text = _HASHTAG_RE.sub(r"\1" if keep_hashtag_text else " ", text)

    # 6. Typographic normalisation
    if normalize_unicode:
        text = text.translate(_TYPOGRAPHIC)

    # 7. Collapse absurd repetition (спааааасибо → спааасибо, !!!! → !)
    if collapse_repeats:
        text = _REPEAT_LETTER_RE.sub(r"\1\1\1", text)
        text = _REPEAT_PUNCT_RE.sub(r"\1", text)

    # 8. Whitespace
    text = _MULTI_WS_RE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """
    Morphological lemmatisation via pymorphy2.

    NOTE: Do NOT use in the transformer training pipeline.
    This is kept for classical ML baselines or linguistic analysis only.
    """
    if not _PYMORPHY_AVAILABLE:
        return text

    tokens = text.split()
    result = []
    for token in tokens:
        parsed = _morph.parse(token)
        if not parsed:
            continue
        lemma = parsed[0].normal_form
        if remove_stopwords and lemma in _RU_STOPWORDS:
            continue
        result.append(lemma)

    return " ".join(result)


def preprocess_for_transformer(texts: List[str], clean: bool = True) -> List[str]:
    """Recommended preprocessing for ruBERT / XLM-RoBERTa. No lemmatisation."""
    if not clean:
        return [t if isinstance(t, str) else "" for t in texts]
    return [clean_text(t) for t in texts]


def preprocess_batch(batch: dict, text_column: str = "text", clean: bool = True) -> dict:
    """HuggingFace Dataset.map()-compatible batch preprocessor."""
    batch[text_column] = preprocess_for_transformer(batch[text_column], clean=clean)
    return batch
