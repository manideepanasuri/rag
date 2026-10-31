# TASKS.md — Solstice Docs Q&A Assistant Implementation Roadmap

This document outlines the sequential, atomic engineering tasks required to build the Docs Q&A Assistant as specified in `AGENTS.md` and `SOLUTION_DESIGN.md`.

Pony-tail principles applied:
- Smallest complete code that solves Elena's problem.
- Zero extraneous dependencies: standard library HTTP for LLM calls, embedded Qdrant/SQLite for local execution.
- Deterministic abstention guardrail: return exact string `"I don't have an answer for that."` when evidence is missing.

---

## Phase 1: Environment, Dependencies & Data Models

- [x] **Task 1.1: Environment Setup & Minimal Dependencies**
  - **Target Files**: `requirements.txt`, `.env.example`, `.gitignore`
  - **Implementation**:
    - Create `requirements.txt` with pinned dependencies: `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `sqlalchemy`, `qdrant-client`, `streamlit`, `pytest`, `httpx`.
    - Create `.env.example` with template keys: `DATABASE_URL=sqlite:///./docs_qa.db`, `GEMINI_API_KEY=`, `OPENAI_API_KEY=`, `BACKEND_URL=http://localhost:8000`, `PORT=8000`.
    - Create `.gitignore` ignoring `.venv`, `__pycache__`, `*.db`, `qdrant_storage/`, `.env`.
  - **Done Criteria**: Run `pip install -r requirements.txt` in a virtual environment without dependency conflicts.

- [x] **Task 1.2: App Configuration & Database Schema**
  - **Target Files**: `app/__init__.py`, `app/config.py`, `app/db/__init__.py`, `app/db/session.py`, `app/db/models.py`
  - **Implementation**:
    - `app/config.py`: Define `Settings` using `pydantic_settings.BaseSettings` with environment fallbacks (`database_url`, `gemini_api_key`, `qdrant_storage_path="./qdrant_storage"`, `confidence_threshold=0.40`).
    - `app/db/session.py`: SQLAlchemy database engine, `sessionmaker`, and `get_db` generator.
    - `app/db/models.py`:
      - `Document`: `id` (string PK, e.g. "D-01"), `title` (string), `content` (text), `created_at` (datetime), `updated_at` (datetime).
      - `Chunk`: `id` (string PK, e.g. "D-01-c0"), `doc_id` (FK to documents.id), `chunk_index` (int), `text` (text).
  - **Done Criteria**: Run `python3 -c "from app.db.session import engine, Base; from app.db.models import Document, Chunk; Base.metadata.create_all(bind=engine); print('DB initialized')"` and confirm tables are created without error.

- [x] **Task 1.3: Document Auto-Seeder**
  - **Target Files**: `app/db/seeder.py`
  - **Implementation**:
    - Create `seed_docs_if_empty(db: Session)` function.
    - Check if `Document` table has records. If count is 0, read `starter_kit/data/docs.json` and insert all 6 initial documents.
  - **Done Criteria**: Run `python3 -c "from app.db.session import SessionLocal; from app.db.seeder import seed_docs_if_empty; from app.db.models import Document; db = SessionLocal(); seed_docs_if_empty(db); assert db.query(Document).count() == 6; print('Seeding passed')"` and confirm 6 records inserted.

---

## Phase 2: Core Chunker & Deterministic BM25 Engine

- [x] **Task 2.1: Boundary-Aware Text Chunker**
  - **Target Files**: `app/engine/__init__.py`, `app/engine/chunker.py`, `tests/__init__.py`, `tests/test_engine.py`
  - **Implementation**:
    - Implement `chunk_document(doc_id: str, title: str, content: str, target_size: int = 400) -> list[dict]`.
    - Split content along paragraph boundaries (`\n\n+`) and sentence boundaries (`(?<=[.!?])\s+`) rather than character-count chopping.
    - Group sentences up to target size (~300-500 chars). Retain remainders without truncating.
    - Return list of chunk dicts: `{"chunk_id": f"{doc_id}-c{idx}", "doc_id": doc_id, "title": title, "chunk_index": idx, "text": chunk_text}`.
    - Add unit tests in `tests/test_engine.py` verifying no dropped text and correct chunk boundaries.
  - **Done Criteria**: Run `pytest tests/test_engine.py -k test_chunker` and pass.

- [x] **Task 2.2: Deterministic BM25 Search Engine**
  - **Target Files**: `app/engine/bm25.py`, `tests/test_engine.py`
  - **Implementation**:
    - Implement lightweight Okapi BM25 (`BM25Index`) in pure Python (no heavyweight external dependencies).
    - Tokenization helper: lowercase, strip punctuation, filter common stop words.
    - Compute IDF across indexed chunks, term frequencies, document length normalization ($k_1 = 1.5, b = 0.75$).
    - Implement `search(query: str, top_k: int = 3) -> list[tuple[float, dict]]` returning chunk dicts sorted by BM25 score descending.
    - Add unit tests in `tests/test_engine.py` testing BM25 token matching on sample sentences.
  - **Done Criteria**: Run `pytest tests/test_engine.py -k test_bm25` and pass.

---

## Phase 3: Gemini Embeddings & Qdrant Vector Store

- [x] **Task 3.1: Zero-Dependency Gemini REST Client**
  - **Target Files**: `app/engine/llm_client.py`, `tests/test_engine.py`
  - **Implementation**:
    - Implement `GeminiClient` using standard library `urllib.request` (zero SDK dependencies).
    - `get_embedding(text: str, api_key: str) -> list[float] | None`: calls Gemini `text-embedding-004` (dim=768).
    - `generate_grounded_answer(question: str, context: str, api_key: str) -> str | None`: calls `gemini-1.5-flash` with a strict anti-hallucination system prompt ("Answer only from the context. If the context does not answer the question, return 'I don't have an answer for that.'").
    - Gracefully catch timeouts (`urllib.error.URLError`), HTTP errors, or missing API keys, returning `None` to allow fallback.
  - **Done Criteria**: Run `python3 -c "from app.engine.llm_client import GeminiClient; c = GeminiClient(); print('Client initialized successfully')"` without errors.

- [x] **Task 3.2: Qdrant Vector DB & Hybrid Scoring Adapter**
  - **Target Files**: `app/engine/vector_store.py`, `tests/test_engine.py`
  - **Implementation**:
    - Implement `QdrantVectorStore` wrapping `qdrant-client`.
    - Local disk mode: initialize `QdrantClient(path=settings.qdrant_storage_path)`. If `settings.qdrant_url` is provided, connect via URL.
    - Ensure collection `docs_chunks` exists (vector size 768, distance Cosine).
    - Method `upsert_chunks(chunks: list[dict], embeddings: list[list[float]])`.
    - Method `search(query_vector: list[float], top_k: int = 3) -> list[tuple[float, dict]]`.
    - Implement hybrid score combiner:
      - If Gemini vector is available: `score = 0.6 * cosine_sim + 0.4 * normalized_bm25`.
      - If Gemini vector is unavailable (no key or offline): graceful fallback to 100% BM25 score.
  - **Done Criteria**: Run `pytest tests/test_engine.py -k test_vector_store` and pass.

---

## Phase 4: Anti-Hallucination Coverage Gate & Pinpointing

- [x] **Task 4.1: Query Term Coverage Gate**
  - **Target Files**: `app/engine/coverage.py`, `tests/test_engine.py`
  - **Implementation**:
    - Elena's core guardrail: If question references entities never mentioned anywhere in documentation, abstain immediately.
    - Implement `QueryCoverageGate`:
      - Extract key content words / entities from query (stripping questions words like "how", "what", "is", "can", "i", "do", "for", "the").
      - Cross-reference key words against the corpus vocabulary.
      - If a critical query word (e.g., "salesforce" in Q-5) is completely absent from all documentation, return `CoverageResult(is_covered=False, reason="Missing query entity")`.
      - Check minimum retrieval confidence threshold ($< 0.40$). If below threshold, return `CoverageResult(is_covered=False, reason="Low confidence")`.
    - Abstention return string must be exact: `"I don't have an answer for that."`.
  - **Done Criteria**: Run `pytest tests/test_engine.py -k test_coverage_gate` ensuring Q-5 ("Salesforce") is rejected while Q-1..Q-4 pass.

- [x] **Task 4.2: Sentence-Level Answer Pinpointer**
  - **Target Files**: `app/engine/extractor.py`, `tests/test_engine.py`
  - **Implementation**:
    - Implement `extract_answer_pinpoint(question: str, chunk_text: str) -> str`:
      - Break chunk into individual sentences.
      - Score each sentence against query salient terms using token overlap and proximity.
      - Return the highest-scoring 1-2 sentences as the concise pinpointed answer (e.g., for Q-2: "Note: password reset links expire after 15 minutes.").
    - Provide fallback support: if pinpointed extract is found, use it; if `use_llm=True` and LLM available, generate synthesized sentence using retrieved chunk as grounding context.
  - **Done Criteria**: Run `pytest tests/test_engine.py -k test_pinpointer` and verify sentence extraction on Q-2 returns the 15-minute expiration statement.

---

## Phase 5: DocsQA Engine Service & FastAPI REST API

- [x] **Task 5.1: Unified DocsQA Engine Service**
  - **Target Files**: `app/engine/service.py`
  - **Implementation**:
    - Implement `DocsQAService`:
      - `initialize(db: Session)`: Loads docs, chunks docs, builds BM25 index, seeds/updates Qdrant vectors if Gemini is enabled.
      - `ask(question: str, use_llm: bool = False, confidence_threshold: float = None) -> dict`:
        1. Run `QueryCoverageGate`. If not covered, return `status="not_covered"`, `answer="I don't have an answer for that."`, `doc_id=None`.
        2. Perform hybrid retrieval (Gemini dense + BM25 lexical).
        3. Check top score against confidence threshold. If low, return `status="not_covered"`.
        4. Pinpoint answer sentence from top chunk (or synthesize via LLM if requested and enabled).
        5. Return `status="answered"`, pinpointed `answer`, `doc_id`, `doc_title`, `confidence_score`, and `supporting_excerpt`.
      - `add_or_update_doc(db: Session, doc_id: str, title: str, content: str)`: updates DB and re-indexes chunk/BM25/vector indices dynamically.
      - `delete_doc(db: Session, doc_id: str)`: removes from DB and purges indices.
  - **Done Criteria**: Run `python3 -c "from app.db.session import SessionLocal; from app.db.seeder import seed_docs_if_empty; from app.engine.service import DocsQAService; db = SessionLocal(); seed_docs_if_empty(db); s = DocsQAService(db); s.initialize(); res = s.ask('How do I reset my password?'); assert res['doc_id'] == 'D-01'; print('Service verified')"` successfully.

- [x] **Task 5.2: Pydantic Schemas & FastAPI REST Endpoints**
  - **Target Files**: `app/schemas/__init__.py`, `app/schemas/qa.py`, `app/api/__init__.py`, `app/api/routes.py`, `app/main.py`, `tests/test_api.py`
  - **Implementation**:
    - `app/schemas/qa.py`:
      - `AskRequest(question: str, use_llm: bool = False, confidence_threshold: float | None = None)`.
      - `AskResponse(status: str, answer: str, doc_id: str | None, doc_title: str | None, confidence_score: float, supporting_excerpt: str | None, mode: str)`.
      - `DocCreate(doc_id: str, title: str, content: str)`, `DocResponse(doc_id: str, title: str, content: str)`.
    - `app/api/routes.py`:
      - `GET /api/v1/health`: health status, db status, indexed document count.
      - `POST /api/v1/ask`: accepts `AskRequest`, calls `DocsQAService.ask()`, returns `AskResponse`.
      - `GET /api/v1/docs`: lists all documents in DB.
      - `POST /api/v1/docs`: creates/updates document and rebuilds index.
      - `DELETE /api/v1/docs/{doc_id}`: deletes document and rebuilds index.
    - `app/main.py`: FastAPI app instantiation, CORS middleware, lifespan handler (seeding DB and initializing engine service on boot), OpenAPI docs at `/docs`.
  - **Done Criteria**: Run `pytest tests/test_api.py` verifying `/api/v1/health` returns status ok and `/api/v1/ask` returns correct answers.

---

## Phase 6: Client Interfaces (CLI & Streamlit)

- [x] **Task 6.1: CLI Client & Automated Benchmark Runner**
  - **Target Files**: `cli.py`
  - **Implementation**:
    - Build standalone CLI tool using `argparse`:
      - `python cli.py ask "<question>"`: calls engine or backend, prints clean human-readable answer, source doc, confidence score, and supporting excerpt.
      - `python cli.py ask "<question>" --json`: outputs structured JSON.
      - `python cli.py interactive`: interactive REPL session for testing arbitrary questions.
      - `python cli.py eval`: automated evaluation against `starter_kit/data/example_questions.json`:
        - Iterates over all 5 questions.
        - Compares retrieved `doc_id` to `expected_doc_id`.
        - For Q-5, asserts answer is `"I don't have an answer for that."` and `doc_id` is None.
        - Prints results summary, latency per question, and overall pass rate.
        - Exits with return code 0 if 100% pass (5/5), code 1 if any failure.
  - **Done Criteria**: Run `python cli.py eval` and confirm 5/5 PASSED output with exit code 0.

- [x] **Task 6.2: Streamlit Interactive Web Application**
  - **Target Files**: `streamlit_app.py`
  - **Implementation**:
    - Streamlit web interface running on port 8501:
      - **View 1: Support Q&A Assistant**:
        - Plain-English question input.
        - Toggle switch: "Deterministic Extraction" vs "LLM Synthesis".
        - Confidence score indicator badge (green for high confidence, red for not covered).
        - Source document badge (e.g. `D-01: Resetting Your Password`).
        - Supporting excerpt accordion/expander showing the exact context passage.
      - **View 2: Benchmark Evaluation Suite**:
        - "Run 5 Benchmark Tests" button.
        - Executes all 5 test questions from `starter_kit/data/example_questions.json`.
        - Displays pass/fail status, expected vs actual doc ID, latency, and overall pass percentage.
      - **View 3: Knowledge Base Management**:
        - List indexed articles with document ID and title.
        - Form to add a new article and immediately test live search re-indexing.
  - **Done Criteria**: Run `python -m py_compile streamlit_app.py` and verify syntax without error.

---

## Phase 7: Verification, Containerization & README

- [x] **Task 7.1: Comprehensive Test Suite & Benchmark Validation**
  - **Target Files**: `tests/test_benchmark.py`
  - **Implementation**:
    - Pytest automated test file `test_benchmark.py`:
      - Test Q-1: `How do I reset my password?` -> returns `D-01`.
      - Test Q-2: `How long do I have to use a password reset link before it expires?` -> returns `D-01` with `15 minutes` in answer.
      - Test Q-3: `What's the maximum number of rows I can export to CSV?` -> returns `D-03` with `10,000` in answer.
      - Test Q-4: `What are the API rate limits for the Growth plan?` -> returns `D-06` with `60 requests per minute` in answer.
      - Test Q-5: `Can I connect Solstice to Salesforce?` -> returns `status="not_covered"`, `answer="I don't have an answer for that."`, `doc_id=None`.
  - **Done Criteria**: Run `pytest tests/test_benchmark.py` with 100% passing tests.

- [x] **Task 7.2: Dockerfile & Docker Compose Setup**
  - **Target Files**: `Dockerfile`, `docker-compose.yml`, `.dockerignore`
  - **Implementation**:
    - `Dockerfile`: Clean python:3.11-slim or python:3.12-slim base image, install requirements, copy app code.
    - `docker-compose.yml`: Spins up FastAPI backend on port 8000 and Streamlit frontend on port 8501 with environment variables wired.
    - `.dockerignore`: exclude `.git`, `__pycache__`, `.venv`, `*.db`, `qdrant_storage/`.
  - **Done Criteria**: Validate docker compose configuration via syntax check or dry run.

- [x] **Task 7.3: Complete README Documentation**
  - **Target Files**: `README.md`
  - **Implementation**:
    - Quick start setup instructions (virtualenv creation, requirements install).
    - Instructions to start FastAPI backend (`uvicorn app.main:app --port 8000`).
    - Instructions to launch Streamlit UI (`streamlit run streamlit_app.py`).
    - Instructions to run CLI and evaluation (`python cli.py eval`, `pytest`).
    - Architecture summary and environment variable reference (`.env.example`).
    - Section on Known Limitations & Future Improvements (addressing Elena's long-term LMS support needs as required by the case study brief).
  - **Done Criteria**: Review `README.md` against case study deliverables checklist.
