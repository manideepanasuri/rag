import pytest
from app.engine.chunker import chunk_document, chunk_all_documents, split_into_sentences
from app.engine.bm25 import BM25Index, tokenize
from app.engine.coverage import QueryCoverageGate
from app.engine.extractor import extract_answer_pinpoint
from app.engine.llm_client import GeminiClient
from app.engine.vector_store import QdrantVectorStore, combine_hybrid_scores


def test_chunker_basic():
    doc_id = "D-TEST"
    title = "Test Article"
    content = (
        "This is sentence one. This is sentence two. This is sentence three. "
        "This is sentence four. This is sentence five."
    )
    chunks = chunk_document(doc_id, title, content, target_size=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c["doc_id"] == doc_id
        assert c["title"] == title
        assert "chunk_id" in c
        assert "chunk_index" in c
        assert "text" in c


def test_chunker_retains_all_text():
    content = "Small document."
    chunks = chunk_document("D-SHORT", "Short", content, target_size=400)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Small document."


def test_chunker_empty():
    assert chunk_document("D-EMPTY", "Empty", "") == []
    assert chunk_document("D-EMPTY", "Empty", "   \n\n  ") == []


def test_bm25_search():
    chunks = [
        {"chunk_id": "c1", "doc_id": "D-01", "title": "Password Reset", "text": "Click forgot password to reset your credentials."},
        {"chunk_id": "c2", "doc_id": "D-02", "title": "Billing FAQ", "text": "Billing cycles renew every month on the subscription date."},
        {"chunk_id": "c3", "doc_id": "D-03", "title": "Export Limits", "text": "CSV exports are limited to 10,000 rows max."}
    ]
    index = BM25Index()
    index.index_chunks(chunks)

    results = index.search("How do I reset my password?")
    assert len(results) > 0
    top_score, top_chunk = results[0]
    assert top_chunk["doc_id"] == "D-01"
    assert top_score > 0.0

    results = index.search("CSV export rows")
    assert len(results) > 0
    top_score, top_chunk = results[0]
    assert top_chunk["doc_id"] == "D-03"


def test_bm25_empty():
    index = BM25Index()
    assert index.search("anything") == []


def test_coverage_gate():
    chunks = [
        {"chunk_id": "c1", "doc_id": "D-01", "title": "Password Reset", "text": "Click forgot password to reset your email and password."},
        {"chunk_id": "c2", "doc_id": "D-06", "title": "API Rate Limits", "text": "The Solstice API is available on Growth plans."}
    ]
    gate = QueryCoverageGate(confidence_threshold=0.35)
    gate.update_corpus_vocabulary(chunks)

    # Valid question covered by docs
    res_valid = gate.check_query_terms("How do I reset my password?")
    assert res_valid.is_covered is True

    # Question with unknown entity "Salesforce"
    res_negative = gate.check_query_terms("Can I connect Solstice to Salesforce?")
    assert res_negative.is_covered is False
    assert "Salesforce" in res_negative.reason
    assert res_negative.abstention_message == "I don't have an answer for that."

    # Confidence check
    res_low = gate.check_retrieval_confidence(0.20)
    assert res_low.is_covered is False
    res_high = gate.check_retrieval_confidence(0.55)
    assert res_high.is_covered is True


def test_pinpointer():
    chunk_text = (
        "If you don't see the email within a few minutes, check your spam folder. "
        "Note: password reset links expire after 15 minutes. "
        "If yours has expired, request a brand new reset link."
    )
    question = "How long do I have to use a password reset link before it expires?"
    answer = extract_answer_pinpoint(question, chunk_text)
    assert "15 minutes" in answer

    chunk_export = (
        "Any report can be exported to CSV. "
        "Important limit: CSV exports are capped at 10,000 rows. "
        "If your report has more rows, apply additional filters."
    )
    ans_export = extract_answer_pinpoint("What's the maximum number of rows I can export to CSV?", chunk_export)
    assert "10,000 rows" in ans_export


def test_llm_client_offline():
    client = GeminiClient()
    # When API key is empty or none, returns None safely
    assert client.get_embedding("hello", "") is None
    assert client.generate_grounded_answer("question", "context", "") is None


def test_vector_store_in_memory():
    store = QdrantVectorStore(storage_path=":memory:", dim=4)
    chunks = [
        {"chunk_id": "c1", "doc_id": "D-01", "title": "T1", "chunk_index": 0, "text": "txt1"},
        {"chunk_id": "c2", "doc_id": "D-02", "title": "T2", "chunk_index": 0, "text": "txt2"},
    ]
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 2

    # Query matching c1
    hits = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    score, payload = hits[0]
    assert payload["chunk_id"] == "c1"
    assert score > 0.9

    # Test hybrid combiner
    bm25_res = [(0.5, chunks[1])]
    combined = combine_hybrid_scores(bm25_res, hits, alpha=0.6)
    assert len(combined) == 2
    # c1 has vector score ~1.0 * 0.6 = 0.6; c2 has bm25 score 0.5 * 0.4 = 0.2
    assert combined[0][1]["chunk_id"] == "c1"

    # Delete doc c1
    store.delete_doc_chunks("D-01")
    assert store.count() == 1
