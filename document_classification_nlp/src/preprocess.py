"""
preprocess.py
-------------
Text cleaning utilities shared by train.py and predict.py.

The cleaning step here is intentionally simple and dependency-free
(no NLTK / spaCy download needed) so the project runs on any machine:

    1. Lowercase the text
    2. Strip URLs, HTML tags and non-alphabetic characters
    3. Collapse repeated whitespace

Stopword removal and TF-IDF weighting are handled later by
scikit-learn's TfidfVectorizer(stop_words="english"), which needs no
external corpus download.
"""

import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and strip noise from a single document."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def clean_corpus(texts):
    """Apply clean_text to an iterable of documents."""
    return [clean_text(t) for t in texts]
