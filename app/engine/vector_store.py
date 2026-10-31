import os
import uuid
import logging
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "docs_chunks"
VECTOR_DIM = 768


class QdrantVectorStore:
    """
    Qdrant vector store adapter supporting local embedded storage and cloud clusters.
    """

    def __init__(
        self,
        storage_path: str | None = None,
        url: str | None = None,
        collection_name: str = DEFAULT_COLLECTION,
        dim: int = VECTOR_DIM,
    ):
        self.collection_name = collection_name
        self.dim = dim

        if url:
            self.client = QdrantClient(url=url)
        elif storage_path and storage_path != ":memory:":
            try:
                os.makedirs(storage_path, exist_ok=True)
                self.client = QdrantClient(path=storage_path)
            except Exception as e:
                logger.warning(
                    "Storage path '%s' is locked by another process (%s). Falling back to in-memory Qdrant client.",
                    storage_path,
                    e,
                )
                self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(location=":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                )
        except Exception as e:
            logger.warning("Failed to verify/create Qdrant collection: %s", e)

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunk points with their embeddings and metadata payload."""
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            payload = {
                "chunk_id": chunk_id,
                "doc_id": chunk["doc_id"],
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "text": chunk.get("text", ""),
            }
            points.append(PointStruct(id=pt_id, vector=embedding, payload=payload))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 3,
    ) -> list[tuple[float, dict[str, Any]]]:
        """Search nearest chunks by cosine similarity."""
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            scored: list[tuple[float, dict[str, Any]]] = []
            for pt in results.points:
                # Cosine similarity in Qdrant is between -1 and 1, or 0 and 1
                score = max(0.0, float(pt.score))
                payload = dict(pt.payload or {})
                scored.append((score, payload))
            return scored
        except Exception as e:
            logger.warning("Vector search query failed: %s", e)
            return []

    def delete_doc_chunks(self, doc_id: str) -> None:
        """Remove all chunks associated with a doc_id."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )
        except Exception as e:
            logger.warning("Failed to delete chunks for doc %s: %s", doc_id, e)

    def count(self) -> int:
        """Return total chunks in the collection."""
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception:
            return 0

    def close(self) -> None:
        """Close client connection."""
        try:
            self.client.close()
        except Exception:
            pass


def combine_hybrid_scores(
    bm25_results: list[tuple[float, dict[str, Any]]],
    vector_results: list[tuple[float, dict[str, Any]]],
    alpha: float = 0.6,
    top_k: int = 3,
) -> list[tuple[float, dict[str, Any]]]:
    """
    Blend dense semantic scores and sparse BM25 scores.
    If vector_results is empty, fall back 100% to BM25 scores.
    """
    if not vector_results:
        return bm25_results[:top_k]

    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}

    for b_score, chunk in bm25_results:
        cid = chunk["chunk_id"]
        chunks[cid] = chunk
        scores[cid] = (1.0 - alpha) * b_score

    for v_score, chunk in vector_results:
        cid = chunk["chunk_id"]
        if cid not in chunks:
            chunks[cid] = chunk
        scores[cid] = scores.get(cid, 0.0) + (alpha * v_score)

    blended = [(scores[cid], chunks[cid]) for cid in scores]
    blended.sort(key=lambda x: x[0], reverse=True)
    return blended[:top_k]
