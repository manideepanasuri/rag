import re
from typing import Any


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly based on punctuation boundaries."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence ending punctuation followed by whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    doc_id: str,
    title: str,
    content: str,
    target_size: int = 400,
) -> list[dict[str, Any]]:
    """
    Split a document into boundary-aware chunks.
    Respects paragraphs and sentences without truncating or dropping remainders.
    """
    content = content.strip()
    if not content:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
    if not paragraphs:
        paragraphs = [content]

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)
        if not sentences:
            continue

        for sentence in sentences:
            sent_len = len(sentence)
            # If current buffer has content and adding this sentence exceeds target_size
            if current_sentences and (current_length + sent_len + 1 > target_size):
                chunk_text = " ".join(current_sentences)
                idx = len(chunks)
                chunks.append({
                    "chunk_id": f"{doc_id}-c{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": idx,
                    "text": chunk_text,
                })
                current_sentences = [sentence]
                current_length = sent_len
            else:
                current_sentences.append(sentence)
                current_length += sent_len + (1 if len(current_sentences) > 1 else 0)

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        idx = len(chunks)
        chunks.append({
            "chunk_id": f"{doc_id}-c{idx}",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": idx,
            "text": chunk_text,
        })

    return chunks


def chunk_all_documents(
    docs: list[dict[str, Any]],
    target_size: int = 400,
) -> list[dict[str, Any]]:
    """Chunk a list of document dicts (or models) into flat chunks."""
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = doc.get("id") or doc.get("doc_id")
        title = doc.get("title", "")
        content = doc.get("content", "")
        chunks = chunk_document(doc_id, title, content, target_size=target_size)
        all_chunks.extend(chunks)
    return all_chunks
