from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document
from app.db.session import get_db
from app.engine.service import DocsQAService
from app.schemas.qa import (
    AskRequest,
    AskResponse,
    DocCreate,
    DocResponse,
    HealthResponse,
)

router = APIRouter(prefix="/api/v1")

_service_instance: DocsQAService | None = None


def get_qa_service() -> DocsQAService:
    global _service_instance
    if _service_instance is None:
        _service_instance = DocsQAService()
    return _service_instance


@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_db),
    qa_service: DocsQAService = Depends(get_qa_service),
):
    """Check service health and database connectivity."""
    try:
        doc_count = db.query(Document).count()
        db_status = "connected"
    except Exception as e:
        doc_count = 0
        db_status = f"unhealthy: {e}"

    return HealthResponse(
        status="ok",
        database=db_status,
        indexed_documents=doc_count,
        indexed_chunks=len(qa_service.chunks),
        gemini_embeddings_enabled=bool(settings.gemini_api_key),
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(
    req: AskRequest,
    qa_service: DocsQAService = Depends(get_qa_service),
):
    """
    Query the Solstice documentation.
    Returns the answer, source doc ID, confidence score, and supporting excerpt.
    Enforces abstention: returns 'I don't have an answer for that.' when not covered.
    """
    result = qa_service.ask(
        question=req.question,
        use_llm=req.use_llm,
        confidence_threshold=req.confidence_threshold,
    )
    return AskResponse(**result)


@router.get("/docs", response_model=list[DocResponse])
def list_documents(
    db: Session = Depends(get_db),
    qa_service: DocsQAService = Depends(get_qa_service),
):
    """List all stored documentation articles."""
    docs = qa_service.list_docs(db)
    return [
        DocResponse(
            doc_id=d.id,
            title=d.title,
            content=d.content,
            created_at=d.created_at.isoformat() if d.created_at else None,
            updated_at=d.updated_at.isoformat() if d.updated_at else None,
        )
        for d in docs
    ]


@router.post("/docs", response_model=DocResponse, status_code=status.HTTP_201_CREATED)
def create_or_update_document(
    doc_in: DocCreate,
    db: Session = Depends(get_db),
    qa_service: DocsQAService = Depends(get_qa_service),
):
    """Create or update a documentation article and immediately re-index search indices."""
    doc = qa_service.add_or_update_doc(
        db=db,
        doc_id=doc_in.doc_id,
        title=doc_in.title,
        content=doc_in.content,
    )
    return DocResponse(
        doc_id=doc.id,
        title=doc.title,
        content=doc.content,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


@router.delete("/docs/{doc_id}")
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    qa_service: DocsQAService = Depends(get_qa_service),
):
    """Delete a documentation article and purge its chunks from indices."""
    success = qa_service.delete_doc(db=db, doc_id=doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return {"status": "deleted", "doc_id": doc_id}
