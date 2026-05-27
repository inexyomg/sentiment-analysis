import re
from typing import List, Optional

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


_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HASHTAG_RE  = re.compile(r"#(\w+)")
_MULTI_WS_RE = re.compile(r"\s+")
_NOISE_RE    = re.compile(r"[^\w\s.,!?;:\-—«»\"'()]")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
    lowercase: bool = True,
) -> str:
    if not isinstance(text, str):
        return ""

    if remove_urls:
        text = _URL_RE.sub(" ", text)
    if remove_mentions:
        text = _MENTION_RE.sub(" ", text)

    text = _HASHTAG_RE.sub(r"\1" if keep_hashtag_text else " ", text)
    text = _NOISE_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()

    if lowercase:
        text = text.lower()

    return text


def lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """Lemmatize Russian text (requires pymorphy2)."""
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


def preprocess_for_transformer(
    texts: List[str],
    clean: bool = True,
    lemmatize_texts: bool = False,
) -> List[str]:
    """
    Light preprocessing for transformer input.
    Heavy normalization (lemmatization) is usually not needed for BERT models
    because the tokenizer handles subwords, but it's available as an option.
    """
    result = []
    for t in texts:
        if clean:
            t = clean_text(t)
        if lemmatize_texts:
            t = lemmatize(t)
        result.append(t)
    return result


def preprocess_batch(
    batch: dict,
    text_column: str = "text",
    clean: bool = True,
    lemmatize_texts: bool = False,
) -> dict:
    """HuggingFace Dataset.map()-compatible batch preprocessor."""
    batch[text_column] = preprocess_for_transformer(
        batch[text_column], clean=clean, lemmatize_texts=lemmatize_texts
    )
    return batch
