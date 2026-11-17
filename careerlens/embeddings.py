"""Embeddings: local, free, no API key needed.

Two backends behind one interface (``fit`` / ``encode`` / ``name``):

- ``SentenceTransformerEmbedder`` — dense embeddings from
  ``all-MiniLM-L6-v2``. Used when the ``sentence-transformers`` package
  is installed and the model can be loaded.
- ``TfidfEmbedder`` — a from-scratch TF-IDF embedder built on numpy
  only (unigrams + bigrams, sublinear TF, smoothed IDF, L2-normalised).
  Always available; the default when sentence-transformers is missing.

``get_embedder()`` prefers sentence-transformers and falls back to
TF-IDF automatically, so the demo runs out of the box.
"""

import math
import re

import numpy as np

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "at", "from", "as", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "we", "our", "she",
    "her", "his", "he", "they", "their", "you", "your", "i", "me", "my",
    "not", "no", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "so", "if",
    "but", "than", "then", "when", "while", "which", "who", "whom",
    "what", "how", "why", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "too",
    "very", "just", "also", "into", "over", "under", "between", "through",
    "about", "up", "down", "out", "off", "again", "further", "once",
    "here", "there", "where", "am", "an", "using", "used", "use",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.#]*")


def tokenize(text):
    """Lowercase alphanumeric tokens, stopwords removed."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _ngrams(tokens, n=2):
    grams = list(tokens)
    for i in range(len(tokens) - n + 1):
        grams.append(tokens[i] + " " + tokens[i + 1])
    return grams


class TfidfEmbedder:
    """TF-IDF embeddings over unigrams + bigrams, numpy only."""

    name = "tfidf"

    def __init__(self):
        self.vocab = {}
        self.idf = None

    def fit(self, texts):
        """Build the vocabulary and IDF weights from the corpus texts."""
        doc_freq = {}
        for text in texts:
            seen = set(_ngrams(tokenize(text)))
            for gram in seen:
                doc_freq[gram] = doc_freq.get(gram, 0) + 1
        n_docs = max(1, len(texts))
        self.vocab = {gram: i for i, gram in enumerate(sorted(doc_freq))}
        self.idf = np.zeros(len(self.vocab), dtype=np.float64)
        for gram, idx in self.vocab.items():
            # Smoothed IDF: always > 0, rare terms weigh more.
            self.idf[idx] = math.log((n_docs + 1) / (doc_freq[gram] + 1)) + 1.0
        return self

    def encode(self, texts):
        """Return an (n, vocab) L2-normalised TF-IDF matrix."""
        if self.idf is None:
            raise ValueError("TfidfEmbedder.fit() must be called first")
        rows = np.zeros((len(texts), len(self.vocab)), dtype=np.float64)
        for r, text in enumerate(texts):
            counts = {}
            for gram in _ngrams(tokenize(text)):
                idx = self.vocab.get(gram)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1
            for idx, count in counts.items():
                # Sublinear term frequency: 1 + log(tf)
                rows[r, idx] = (1.0 + math.log(count)) * self.idf[idx]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return rows / norms


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers (optional dependency)."""

    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: F401
        # Imported lazily: raises ImportError when the package is absent.
        self._model = SentenceTransformer(model_name)

    def fit(self, texts):
        # Dense models need no fitting; kept for interface parity.
        return self

    def encode(self, texts):
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True),
            dtype=np.float64,
        )


def get_embedder():
    """Best available embedder: sentence-transformers, else TF-IDF."""
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return TfidfEmbedder()
