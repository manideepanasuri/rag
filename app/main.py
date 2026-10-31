from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import get_qa_service, router
from app.db.seeder import seed_docs_if_empty
from app.db.session import Base, SessionLocal, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables, seed documents, initialize QA engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_docs_if_empty(db)
        qa_service = get_qa_service()
        qa_service.initialize(db)
    finally:
        db.close()
    yield
    # Shutdown logic
    qa_service = get_qa_service()
    if qa_service and hasattr(qa_service, "vector_store"):
        qa_service.vector_store.close()


app = FastAPI(
    title="Solstice Docs Q&A Assistant API",
    description="Factual, hallucination-free Q&A assistant for Solstice Learning product documentation.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Solstice Docs Q&A Assistant API is running.",
        "documentation": "/docs",
        "health": "/api/v1/health",
    }
