import re
from dataclasses import dataclass
from typing import Any

from app.engine.bm25 import DEFAULT_STOP_WORDS, tokenize

QUERY_STOP_WORDS = DEFAULT_STOP_WORDS | {
    "how", "what", "when", "where", "why", "who", "which", "can", "could",
    "would", "should", "do", "does", "did", "is", "are", "was", "were",
    "tell", "show", "give", "please", "want", "need", "like", "know",
    "much", "many", "solstice", "system", "platform"
}


@dataclass
class CoverageResult:
    is_covered: bool
    reason: str = ""
    abstention_message: str = "I don't have an answer for that."


class QueryCoverageGate:
    """
    Anti-hallucination guardrail:
    Ensures that queries referencing entities/topics not documented in the corpus
    are rejected deterministically with 'I don't have an answer for that.'
    """

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
        self.corpus_vocabulary: set[str] = set()

    def update_corpus_vocabulary(self, chunks: list[dict[str, Any]]) -> None:
        """Collect all known tokens across all documentation chunks."""
        vocab: set[str] = set()
        for chunk in chunks:
            text = f"{chunk.get('title', '')} {chunk.get('text', '')}"
            tokens = tokenize(text, remove_stopwords=False)
            vocab.update(tokens)
        self.corpus_vocabulary = vocab

    def extract_salient_terms(self, query: str) -> list[str]:
        """Extract key non-stopword query tokens."""
        tokens = tokenize(query, remove_stopwords=False)
        salient = [t for t in tokens if t not in QUERY_STOP_WORDS and len(t) > 1]
        return salient

    def check_query_terms(self, query: str, is_semantic: bool = False) -> CoverageResult:
        """
        Check if any critical query entity/noun is completely absent from documentation.
        For example: 'Salesforce', 'HubSpot', 'Workday'.
        """
        if not self.corpus_vocabulary:
            return CoverageResult(is_covered=False, reason="Corpus is empty")

        salient_terms = self.extract_salient_terms(query)
        if not salient_terms:
            return CoverageResult(is_covered=False, reason="No salient query terms found")

        # Check for capitalized words in query (excluding first word of sentence)
        # This catches unknown external third-party tools/entities like 'Salesforce'
        words = query.strip().split()
        for idx, word in enumerate(words):
            clean_word = re.sub(r"[^a-zA-Z0-9]", "", word)
            if not clean_word:
                continue
            # If word is capitalized and not first word, or is all caps
            if (idx > 0 and clean_word[0].isupper()) or clean_word.isupper():
                lower_term = clean_word.lower()
                if lower_term not in QUERY_STOP_WORDS and lower_term not in self.corpus_vocabulary:
                    return CoverageResult(
                        is_covered=False,
                        reason=f"Unknown entity '{clean_word}' not found in documentation",
                    )

        # In semantic mode, synonyms and conceptual paraphrasing are evaluated by vector similarity
        if is_semantic:
            return CoverageResult(is_covered=True, reason="Semantic vocabulary check passed")

        # In offline BM25 mode, check that at least one salient term exists in the corpus
        known_count = sum(1 for t in salient_terms if t in self.corpus_vocabulary)
        if known_count == 0:
            return CoverageResult(
                is_covered=False,
                reason="None of the query terms exist in the documentation",
            )

        return CoverageResult(is_covered=True, reason="Vocabulary check passed")

    def check_retrieval_confidence(
        self,
        top_score: float,
        threshold: float | None = None,
    ) -> CoverageResult:
        """Verify that the retrieval confidence exceeds the minimum threshold."""
        effective_threshold = threshold if threshold is not None else self.confidence_threshold
        if top_score < effective_threshold:
            return CoverageResult(
                is_covered=False,
                reason=f"Retrieval score {top_score:.3f} below threshold {effective_threshold:.3f}",
            )
        return CoverageResult(is_covered=True, reason="Confidence threshold met")
