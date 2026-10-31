"""
Small utility module provided as an optional starting point for the
docs Q&A assistant case study.

Feel free to use, modify, replace, or ignore this entirely -- it is not
guaranteed to be complete or bug-free.
"""
import json
import re

CHUNK_SIZE = 400


def load_docs(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("docs", [])


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split text into fixed-size chunks."""
    chunks = []
    num_chunks = len(text) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        chunks.append(text[start:start + chunk_size])
    return chunks


def chunk_all_docs(docs, chunk_size=CHUNK_SIZE):
    """Return a flat list of {doc_id, title, chunk_index, text} dicts."""
    all_chunks = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_text(doc["content"], chunk_size)):
            all_chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_index": idx,
                "text": chunk,
            })
    return all_chunks


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


def keyword_overlap_score(query_tokens, chunk_tokens):
    shared = set(query_tokens) & set(chunk_tokens)
    if not chunk_tokens:
        return 0.0
    return len(shared) / len(set(query_tokens) | set(chunk_tokens))


def find_best_chunks(question, chunks, top_n=3):
    q_tokens = normalize_text(question)
    scored = []
    for chunk in chunks:
        score = keyword_overlap_score(q_tokens, normalize_text(chunk["text"]))
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_n]
