import os
import pytest
from app.db.models import Document
from app.db.seeder import seed_docs_if_empty
from app.db.session import Base, SessionLocal, engine
from app.engine.coverage import QueryCoverageGate
from app.engine.service import DocsQAService
from app.engine.vector_store import QdrantVectorStore


class MockSemanticClient:
    """
    Mock embedding client for semantic unit testing without requiring external API calls.
    Maps concept spaces to deterministic 768-dim normalized vectors.
    """

    CONCEPT_VECTORS = {
        # Credentials & Password reset concepts (D-01)
        "password": 0, "credentials": 0, "authenticate": 0, "login": 0, "reset": 0, "access": 0,
        # Team management concepts (D-02)
        "invite": 1, "team": 1, "members": 1, "coworkers": 1, "colleague": 1, "provision": 1,
        # CSV Export concepts (D-03)
        "export": 2, "csv": 2, "reports": 2, "download": 2, "rows": 2, "spreadsheet": 2,
        # Billing & Refund concepts (D-04)
        "billing": 3, "renewal": 3, "invoices": 3, "refund": 3, "refunds": 3, "annual": 3, "subscription": 3, "money": 3,
        # SSO & SAML concepts (D-05)
        "sso": 4, "saml": 4, "enterprise": 4, "identity": 4, "provider": 4,
        # API Rate Limit concepts (D-06)
        "api": 5, "rate": 5, "limits": 5, "growth": 5, "requests": 5, "quota": 5,
    }

    def get_embedding(self, text: str, api_key: str | None = None) -> list[float]:
        vec = [0.0] * 768
        lower = text.lower()
        matched = False
        for word, cluster_idx in self.CONCEPT_VECTORS.items():
            if word in lower:
                # Distribute signal across the cluster dimension
                base_idx = cluster_idx * 50
                for offset in range(50):
                    vec[base_idx + offset] += 1.0
                matched = True

        # Normalize vector
        if matched:
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                vec = [x / norm for x in vec]
        else:
            # Baseline neutral vector
            vec[700] = 1.0
        return vec

    def generate_grounded_answer(self, question: str, context: str, api_key: str | None = None) -> str | None:
        return f"Grounded Answer based on: {context[:50]}..."


@pytest.fixture
def semantic_qa_service():
    """Initializes DocsQAService with mock semantic embeddings and in-memory Qdrant."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_docs_if_empty(db)

    # Initialize vector store in memory
    vec_store = QdrantVectorStore(storage_path=":memory:")
    mock_llm = MockSemanticClient()

    service = DocsQAService(
        vector_store=vec_store,
        llm_client=mock_llm,
        confidence_threshold=0.30,
    )

    # Load and index chunks with semantic vectors
    docs = db.query(Document).all()
    doc_dicts = [{"id": d.id, "title": d.title, "content": d.content} for d in docs]
    from app.engine.chunker import chunk_all_documents
    service.chunks = chunk_all_documents(doc_dicts)
    service.bm25_index.index_chunks(service.chunks)
    service.coverage_gate.update_corpus_vocabulary(service.chunks)

    # Embed chunks using the mock client
    embeddings = [
        mock_llm.get_embedding(f"{c['title']}: {c['text']}")
        for c in service.chunks
    ]
    service.vector_store.upsert_chunks(service.chunks, embeddings)

    db.close()
    return service


def test_semantic_credentials_rephrasing(semantic_qa_service):
    """Test Q&A on credential recovery using synonyms with zero keyword overlap with 'password'."""
    q = "How can I regain access if I forgot my credentials?"
    # Pass mock key so semantic search is enabled in ask()
    from app.config import settings
    orig_key = settings.gemini_api_key
    settings.gemini_api_key = "test_mock_key"
    try:
        res = semantic_qa_service.ask(q)
        assert res["status"] == "answered"
        assert res["doc_id"] == "D-01"
        assert res["gemini_enabled"] is True
        assert res["confidence_score"] >= 0.30
    finally:
        settings.gemini_api_key = orig_key


def test_semantic_annual_subscription_refund(semantic_qa_service):
    """Test semantic matching on billing and refunds with paraphrased terms."""
    q = "Can I get a refund if I cancel my annual subscription early?"
    from app.config import settings
    orig_key = settings.gemini_api_key
    settings.gemini_api_key = "test_mock_key"
    try:
        res = semantic_qa_service.ask(q)
        assert res["status"] == "answered"
        assert res["doc_id"] == "D-04"
        assert "refund" in res["answer"].lower() or "30 days" in res["answer"].lower()
    finally:
        settings.gemini_api_key = orig_key


def test_semantic_team_colleague_provisioning(semantic_qa_service):
    """Test semantic matching on inviting colleagues/coworkers."""
    q = "Where can an administrator add new coworkers and team colleagues?"
    from app.config import settings
    orig_key = settings.gemini_api_key
    settings.gemini_api_key = "test_mock_key"
    try:
        res = semantic_qa_service.ask(q)
        assert res["status"] == "answered"
        assert res["doc_id"] == "D-02"
    finally:
        settings.gemini_api_key = orig_key


def test_semantic_abstention_on_salesforce(semantic_qa_service):
    """Verify that even with semantic embeddings active, Salesforce is strictly rejected."""
    q = "Can I connect Solstice to Salesforce?"
    from app.config import settings
    orig_key = settings.gemini_api_key
    settings.gemini_api_key = "test_mock_key"
    try:
        res = semantic_qa_service.ask(q)
        assert res["status"] == "not_covered"
        assert res["doc_id"] is None
        assert res["answer"] == "I don't have an answer for that."
    finally:
        settings.gemini_api_key = orig_key


def test_semantic_abstention_on_unknown_hubspot(semantic_qa_service):
    """Verify that another third-party CRM (HubSpot) also triggers strict abstention."""
    q = "Is there an integration available for HubSpot?"
    from app.config import settings
    orig_key = settings.gemini_api_key
    settings.gemini_api_key = "test_mock_key"
    try:
        res = semantic_qa_service.ask(q)
        assert res["status"] == "not_covered"
        assert res["doc_id"] is None
        assert res["answer"] == "I don't have an answer for that."
    finally:
        settings.gemini_api_key = orig_key


def test_live_gemini_semantic_embeddings():
    """Live test calling Google Gemini embedding API when key is configured."""
    from app.config import settings
    from app.engine.llm_client import GeminiClient

    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("No GEMINI_API_KEY configured in .env or environment")

    client = GeminiClient()
    emb = client.get_embedding("What are the API rate limits?", api_key=api_key)
    assert emb is not None, "Failed to generate embedding with Gemini API"
    assert len(emb) == 768, f"Expected 768 dim vector, got {len(emb)}"
    assert any(x != 0.0 for x in emb), "Vector should not be all zeros"


def test_live_gemini_end_to_end_semantic_retrieval():
    """End-to-end test using live Gemini embeddings to answer a rephrased query with zero keyword overlap."""
    from app.config import settings
    from app.engine.service import DocsQAService

    if not settings.gemini_api_key:
        pytest.skip("No GEMINI_API_KEY configured in .env or environment")

    service = DocsQAService()
    db = SessionLocal()
    service.initialize(db)

    # Rephrased query with no exact keyword overlap with doc title or body
    query = "Can I get a refund if I cancel my annual subscription early?"
    result = service.ask(query)
    db.close()

    assert result["status"] == "answered"
    assert result["doc_id"] == "D-04"
    assert result["gemini_enabled"] is True
    assert "refund" in result["answer"].lower() or "30 days" in result["answer"].lower()
