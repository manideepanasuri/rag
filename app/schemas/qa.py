from datetime import datetime
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="Plain-English support question from platform admin")
    use_llm: bool = Field(False, description="Whether to synthesize answer using Gemini LLM if key is present")
    confidence_threshold: float | None = Field(None, description="Optional override for retrieval confidence threshold")


class AskResponse(BaseModel):
    status: str = Field(..., description="'answered' or 'not_covered'")
    answer: str = Field(..., description="Direct answer or 'I don't have an answer for that.'")
    doc_id: str | None = Field(None, description="Matched document ID or null if not covered")
    doc_title: str | None = Field(None, description="Matched document title or null")
    confidence_score: float = Field(..., description="Hybrid or BM25 retrieval confidence score")
    supporting_excerpt: str | None = Field(None, description="Exact document context excerpt or null")
    mode: str = Field(..., description="'deterministic' or 'llm'")
    gemini_enabled: bool = Field(False, description="Whether Gemini embeddings and semantic search were used")


class DocCreate(BaseModel):
    doc_id: str = Field(..., description="Unique document ID (e.g., 'D-07')")
    title: str = Field(..., description="Document article title")
    content: str = Field(..., description="Full text content of documentation article")


class DocResponse(BaseModel):
    doc_id: str
    title: str
    content: str
    created_at: str | None = None
    updated_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    indexed_documents: int
    indexed_chunks: int
    gemini_embeddings_enabled: bool
