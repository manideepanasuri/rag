import math
import re
from typing import Any

DEFAULT_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}


def stem(w: str) -> str:
    """Lightweight rule-based English stemmer."""
    w = w.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("ing"):
        w = w[:-3]
    elif w.endswith("ed"):
        w = w[:-2]
    elif w.endswith("es"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2]:
        w = w[:-1]
    if w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str, remove_stopwords: bool = True, use_stem: bool = True) -> list[str]:
    """Lowercase, extract alphanumeric words, filter stopwords, and stem."""
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9_]+\b", text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOP_WORDS and len(t) > 1]
    if use_stem:
        tokens = [stem(t) for t in tokens]
    return tokens


class BM25Index:
    """
    Okapi BM25 implementation for lexical keyword ranking over document chunks.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict[str, Any]] = []
        self.doc_len: list[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_term_freqs: list[dict[str, int]] = []

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Build the index from chunk dicts containing 'text' and optional 'title'."""
        self.chunks = list(chunks)
        self.doc_len = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        total_tokens = 0
        for chunk in self.chunks:
            title = chunk.get("title", "")
            text = chunk.get("text", "")
            combined = f"{title} {text}"
            tokens = tokenize(combined)
            length = len(tokens)
            self.doc_len.append(length)
            total_tokens += length

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for term in tf:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        n_docs = len(self.chunks)
        self.avg_doc_len = (total_tokens / n_docs) if n_docs > 0 else 0.0

        # Calculate Okapi IDF with +1 smoothing to guarantee positive values
        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(((n_docs - freq + 0.5) / (freq + 0.5)) + 1.0)

    def score_chunk(self, query_tokens: list[str], chunk_idx: int) -> float:
        """Compute the raw BM25 score for a chunk against query tokens."""
        if not self.chunks or chunk_idx >= len(self.chunks):
            return 0.0

        score = 0.0
        tf_dict = self.doc_term_freqs[chunk_idx]
        d_len = self.doc_len[chunk_idx]
        denom_part = self.k1 * (1 - self.b + self.b * (d_len / (self.avg_doc_len or 1.0)))

        for t in query_tokens:
            if t not in tf_dict:
                continue
            tf = tf_dict[t]
            idf = self.idf.get(t, 0.0)
            score += idf * (tf * (self.k1 + 1.0)) / (tf + denom_part)

        return score

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Search for matching chunks.
        Returns list of (normalized_score, chunk_dict) sorted descending.
        """
        if not self.chunks:
            return []

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        max_possible = sum(self.idf.get(t, 0.0) * (self.k1 + 1.0) for t in q_tokens)
        if max_possible <= 0.0:
            max_possible = 1.0

        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, chunk in enumerate(self.chunks):
            raw_score = self.score_chunk(q_tokens, idx)
            normalized = min(1.0, max(0.0, raw_score / max_possible))
            scored.append((normalized, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
