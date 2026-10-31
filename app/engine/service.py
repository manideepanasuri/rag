import logging
from typing import Any
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Chunk, Document
from app.engine.bm25 import BM25Index
from app.engine.chunker import chunk_all_documents, chunk_document
from app.engine.coverage import QueryCoverageGate
from app.engine.extractor import extract_answer_pinpoint
from app.engine.llm_client import GeminiClient
from app.engine.vector_store import QdrantVectorStore, combine_hybrid_scores

logger = logging.getLogger(__name__)


class DocsQAService:
    """
    Unified DocsQA engine orchestrating chunking, BM25 keyword search,
    Qdrant vector retrieval, coverage gating, and answer extraction.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        llm_client: GeminiClient | None = None,
        confidence_threshold: float | None = None,
    ):
        self.threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.confidence_threshold
        )
        self.bm25_index = BM25Index()
        self.coverage_gate = QueryCoverageGate(confidence_threshold=self.threshold)
        self.llm_client = llm_client or GeminiClient()
        self.vector_store = vector_store or QdrantVectorStore(
            storage_path=settings.qdrant_storage_path,
            url=settings.qdrant_url,
        )
        self.chunks: list[dict[str, Any]] = []
        self.docs: list[dict[str, Any]] = []

    def initialize(self, db: Session) -> None:
        """Load documents from DB, chunk them, build BM25 and Qdrant indices."""
        docs = db.query(Document).order_by(Document.id).all()
        self.docs = [
            {"id": d.id, "doc_id": d.id, "title": d.title, "content": d.content}
            for d in docs
        ]

        self.chunks = chunk_all_documents(self.docs)
        self.bm25_index.index_chunks(self.chunks)
        self.coverage_gate.update_corpus_vocabulary(self.chunks)

        # Sync chunks to DB Chunk table
        db.query(Chunk).delete()
        for c in self.chunks:
            chunk_rec = Chunk(
                id=c["chunk_id"],
                doc_id=c["doc_id"],
                chunk_index=c["chunk_index"],
                text=c["text"],
            )
            db.add(chunk_rec)
        db.commit()

        # Seed Qdrant embeddings if Gemini API key is available
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if api_key:
            logger.info("✨ [GEMINI STATUS] ACTIVE: GEMINI_API_KEY detected. Indexing chunks with gemini-embedding-001 into Qdrant...")
            embeddings: list[list[float]] = []
            for c in self.chunks:
                emb = self.llm_client.get_embedding(
                    f"{c['title']}: {c['text']}", api_key
                )
                if emb:
                    embeddings.append(emb)
                else:
                    embeddings.append([0.0] * 768)
            self.vector_store.upsert_chunks(self.chunks, embeddings)
            logger.info("✨ [GEMINI STATUS] Successfully indexed %d chunks into Qdrant vector store", len(self.chunks))
        else:
            logger.info("⚙️ [GEMINI STATUS] INACTIVE: GEMINI_API_KEY not set. Using pure deterministic BM25 lexical search (100%% offline)")

    def get_all_docs_context(self) -> str:
        """Assemble all documents in the knowledge base into structured context for the LLM."""
        if self.docs:
            blocks = [
                f"[{d['doc_id']}: {d['title']}]\n{d['content'].strip()}"
                for d in self.docs
            ]
            return "\n\n".join(blocks)
        if self.chunks:
            grouped: dict[str, list[str]] = {}
            titles: dict[str, str] = {}
            for c in self.chunks:
                did = c.get("doc_id", "")
                titles[did] = c.get("title", "")
                grouped.setdefault(did, []).append(c.get("text", "").strip())
            blocks = []
            for did, texts in grouped.items():
                blocks.append(f"[{did}: {titles.get(did, '')}]\n" + "\n\n".join(texts))
            return "\n\n".join(blocks)
        return ""

    def ask(
        self,
        question: str,
        use_llm: bool = False,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Ask a question against documentation.
        Enforces abstention guardrail: if out-of-scope or below confidence threshold,
        returns 'I don't have an answer for that.'
        """
        clean_q = question.strip()
        effective_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        gemini_active = bool(effective_key)

        if not clean_q:
            return {
                "status": "not_covered",
                "answer": "I don't have an answer for that.",
                "doc_id": None,
                "doc_title": None,
                "confidence_score": 0.0,
                "supporting_excerpt": None,
                "mode": "deterministic",
                "llm_model": None,
                "gemini_enabled": gemini_active,
            }

        logger.info(
            "🔍 [QUERY] '%s' | Gemini Semantic Embeddings: %s | Mode Requested: %s",
            clean_q,
            "ACTIVE (gemini-embedding-001)" if gemini_active else "INACTIVE (BM25 only)",
            "LLM Synthesis" if (use_llm and gemini_active) else "Deterministic Pinpoint",
        )

        # Step 1: Query coverage gate (check for unknown entities/topics)
        cov_res = self.coverage_gate.check_query_terms(clean_q, is_semantic=gemini_active)
        if not cov_res.is_covered:
            logger.info("Coverage gate rejected query '%s': %s", clean_q, cov_res.reason)
            return {
                "status": "not_covered",
                "answer": cov_res.abstention_message,
                "doc_id": None,
                "doc_title": None,
                "confidence_score": 0.0,
                "supporting_excerpt": None,
                "mode": "deterministic",
                "llm_model": None,
                "gemini_enabled": gemini_active,
            }

        # Step 2: Retrieval
        if use_llm and effective_key:
            # Pure Semantic Vector Retrieval for LLM (No BM25)
            logger.info("⚡ [RETRIEVAL] LLM Mode: Pure Semantic Dense Vector Search via Qdrant (0%% BM25)")
            q_emb = self.llm_client.get_embedding(clean_q, effective_key)
            if q_emb:
                ranked = self.vector_store.search(q_emb, top_k=3)
            else:
                ranked = []
        else:
            # Deterministic Extractive / Offline Mode (BM25 + Hybrid)
            bm25_hits = self.bm25_index.search(clean_q, top_k=3)
            vector_hits: list[tuple[float, dict[str, Any]]] = []
            if effective_key:
                q_emb = self.llm_client.get_embedding(clean_q, effective_key)
                if q_emb:
                    vector_hits = self.vector_store.search(q_emb, top_k=3)
            ranked = combine_hybrid_scores(bm25_hits, vector_hits, alpha=0.6, top_k=3) if vector_hits else bm25_hits

        if not ranked:
            return {
                "status": "not_covered",
                "answer": "I don't have an answer for that.",
                "doc_id": None,
                "doc_title": None,
                "confidence_score": 0.0,
                "supporting_excerpt": None,
                "mode": "deterministic",
                "llm_model": None,
                "gemini_enabled": gemini_active,
            }

        top_score, top_chunk = ranked[0]

        # Step 5: Check retrieval confidence threshold
        active_threshold = (
            confidence_threshold if confidence_threshold is not None else self.threshold
        )
        conf_res = self.coverage_gate.check_retrieval_confidence(top_score, active_threshold)
        if not conf_res.is_covered:
            logger.info("Score %.3f rejected below threshold %.3f", top_score, active_threshold)
            return {
                "status": "not_covered",
                "answer": conf_res.abstention_message,
                "doc_id": None,
                "doc_title": None,
                "confidence_score": round(top_score, 4),
                "supporting_excerpt": None,
                "mode": "deterministic",
                "llm_model": None,
                "gemini_enabled": gemini_active,
            }

        # Step 6: Assemble all documents in knowledge base into context for the LLM
        primary_doc_id = top_chunk["doc_id"]
        all_docs_context = self.get_all_docs_context()
        if not all_docs_context:
            all_docs_context = "\n\n".join(
                f"[{c.get('doc_id')}: {c.get('title')}]\n{c.get('text', '').strip()}"
                for _, c in ranked[:3]
            )

        # Step 7: Generate answer (Deterministic extraction or Grounded LLM)
        mode = "deterministic"
        llm_model_name = None
        answer = extract_answer_pinpoint(clean_q, top_chunk["text"])

        effective_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if use_llm:
            if effective_key:
                logger.info("📚 [CONTEXT] Sending ALL %d documents in knowledge base to LLM context", len(self.docs))
                grounded, model_used = self.llm_client.generate_grounded_answer_with_model(
                    clean_q,
                    all_docs_context,
                    effective_key,
                    temperature=settings.llm_temperature,
                )
                if grounded:
                    if "i don't have an answer for that" in grounded.lower():
                        logger.warning("Gemini LLM returned abstention on matched doc %s; falling back to extractive answer", primary_doc_id)
                    else:
                        answer = grounded
                        mode = "llm"
                        llm_model_name = model_used
                else:
                    logger.warning("Gemini Flash failed to generate answer; keeping extractive answer")
            else:
                logger.warning("use_llm=True requested but GEMINI_API_KEY is not configured")

        return {
            "status": "answered",
            "answer": answer,
            "doc_id": top_chunk["doc_id"],
            "doc_title": top_chunk["title"],
            "confidence_score": round(top_score, 4),
            "supporting_excerpt": all_docs_context if mode == "llm" else top_chunk["text"],
            "mode": mode,
            "llm_model": llm_model_name,
            "gemini_enabled": gemini_active,
        }

    def add_or_update_doc(
        self,
        db: Session,
        doc_id: str,
        title: str,
        content: str,
    ) -> Document:
        """Create or update a document in DB and refresh all search indices."""
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.title = title
            doc.content = content
        else:
            doc = Document(id=doc_id, title=title, content=content)
            db.add(doc)
        db.commit()
        db.refresh(doc)

        # Re-initialize indices
        self.initialize(db)
        return doc

    def delete_doc(self, db: Session, doc_id: str) -> bool:
        """Delete a document and refresh search indices."""
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return False

        self.vector_store.delete_doc_chunks(doc_id)
        db.delete(doc)
        db.commit()

        # Re-initialize indices
        self.initialize(db)
        return True

    def list_docs(self, db: Session) -> list[Document]:
        return db.query(Document).order_by(Document.id).all()
