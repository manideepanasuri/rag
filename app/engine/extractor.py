import re
from typing import Any

from app.engine.bm25 import tokenize
from app.engine.chunker import split_into_sentences


def stem_simple(token: str) -> str:
    """Simple suffix stripping for matching singular/plural and verb tenses."""
    token = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)]
    return token


def extract_answer_pinpoint(question: str, chunk_text: str) -> str:
    """
    Sentence-level extractive pinpointing.
    Finds the most relevant 1-2 sentences in chunk_text that answer the question.
    """
    sentences = split_into_sentences(chunk_text)
    if not sentences:
        return chunk_text.strip()
    if len(sentences) == 1:
        return sentences[0]

    q_tokens = tokenize(question)
    q_stems = {stem_simple(t) for t in q_tokens}

    scored_sentences: list[tuple[float, int, str]] = []

    has_num_intent = any(w in question.lower() for w in ["how long", "how many", "limit", "rate", "maximum", "max", "number"])

    for idx, sentence in enumerate(sentences):
        s_tokens = tokenize(sentence, remove_stopwords=False)
        s_stems = [stem_simple(t) for t in s_tokens]

        # Exact token matches and stem matches
        exact_matches = sum(1 for t in s_tokens if t in q_tokens)
        stem_matches = sum(1 for s in s_stems if s in q_stems)
        match_score = exact_matches * 1.5 + stem_matches * 1.0

        # Numeric / parameter bonus if question asks about numbers/limits/duration
        if has_num_intent:
            if re.search(r"\b\d+([,\.]\d+)?\b", sentence) or any(w in sentence.lower() for w in ["minute", "minutes", "hour", "hours", "day", "days", "rows", "requests", "capped"]):
                match_score += 2.5

        # Colon intro bonus (e.g., "Rate limits: ...", "Important limit: ...", "Note: ...")
        if ":" in sentence:
            match_score += 1.0

        # Density normalization
        length_penalty = len(s_tokens) ** 0.3 if s_tokens else 1.0
        final_score = match_score / length_penalty

        scored_sentences.append((final_score, idx, sentence))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_score, best_idx, best_sentence = scored_sentences[0]

    # If the subsequent sentence immediately continues the instruction and is short, include it
    result_sentences = [best_sentence]
    if best_idx + 1 < len(sentences):
        next_sentence = sentences[best_idx + 1]
        # If total length is reasonable (< 220 chars) and next sentence is related
        if len(best_sentence) + len(next_sentence) <= 220 and best_score > 1.0:
            result_sentences.append(next_sentence)

    return " ".join(result_sentences).strip()
