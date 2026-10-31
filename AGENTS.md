# AGENTS.md — System Architecture & Implementation Guide

This document captures the agreed-upon technical decisions, architectural specifications, and implementation roadmap for the Solstice Docs Q&A Assistant. It serves as the single source of truth for all autonomous agents and developers working on this codebase.

---

## 1. Context & Business Problem

- **Customer**: Solstice Learning (online course platform for corporate training teams).
- **Stakeholder**: Elena Torres (Head of Support).
- **Core Need**: Admins frequently ask repetitive questions (e.g., password resets, export limits, billing cycles, SSO). These are documented in product docs, but admins prefer typing plain-English questions.
- **Critical Iron Rule**: If the documentation genuinely does not cover a question, the assistant must return:
  > `"I don't have an answer for that."`
  It must **never** hallucinate, guess, or extrapolate beyond the provided text.

---

## 2. Architecture Overview (Client-Server)

The solution is structured as a decoupled, production-ready client-server architecture:

```
[ Streamlit Web UI ] (Port 8501)
         │
         ├── HTTP (JSON) ──► [ FastAPI Backend ] (Port 8000)
         │                          │
[ CLI Client / Eval ]               ├──► [ SQLAlchemy ORM ] ──► [ SQLite / PostgreSQL ]
                                    │
                                    ├──► [ Qdrant Vector DB ] ──► [ Local Embedded / AWS Cluster ]
                                    │
                                    ├──► [ BM25 + Coverage Gate Engine ]
                                    │
                                    └──► [ Optional LLM Adapter (OpenAI / Gemini) ]
```

---

## 3. Core Engine Decisions

### 3.1 Dual-Mode Answer Generation & Hybrid Retrieval
1. **Hybrid Retrieval (Gemini Semantic Embeddings + BM25)**:
   - When `GEMINI_API_KEY` is configured, uses Google Gemini's free `text-embedding-004` API to generate document and query vector embeddings.
   - Computes cosine similarity blended with BM25 keyword scoring for maximum recall across rephrased queries.
   - If `GEMINI_API_KEY` is not set, gracefully falls back to deterministic BM25 scoring with zero configuration or crashes.
2. **Deterministic & Extractive Pinpointing**:
   - 100% offline baseline, zero-dependency on external AI models, zero hallucination risk.
   - Combines retrieval with a sentence-level extractive pinpointing module.
   - Returns concise, direct answers accompanied by source doc ID, document title, confidence score, and supporting excerpt.
3. **Optional LLM Synthesis Mode**:
   - Zero external library requirement (uses Python standard library HTTP client).
   - Activated only when requested (`use_llm=True`).
   - Uses a strict grounding prompt injecting only the retrieved excerpt. If the context does not answer the question, it enforces the exact fallback: `"I don't have an answer for that."`.

### 3.2 Chunking & Retrieval Pipeline
- **Boundary-Aware Chunking**: Replaces the naive character chunker with paragraph and sentence boundary chunking. Does not truncate or drop remainders.
- **BM25 Scoring**: Inverse document frequency + term frequency normalization across document chunks.
- **Query Term Coverage Gate (Anti-Hallucination Guardrail)**:
  - Extracts key query nouns and salient terms.
  - Checks if critical query terms exist in the corpus or retrieved chunk.
  - If coverage is below threshold (e.g., query asks about "Salesforce", which is completely absent from all docs), or if BM25 confidence is below threshold, retrieval immediately rejects the query and returns status `NOT_COVERED` with `"I don't have an answer for that."`.

---

## 4. Data Storage & Management

- **Relational Database**: SQLAlchemy with SQLite default (`sqlite:///./docs_qa.db`), switchable to PostgreSQL via `DATABASE_URL`.
- **Vector Database**: Qdrant (`qdrant-client`):
  - Local mode: runs embedded on disk (`./qdrant_storage`) with zero background daemon required.
  - AWS mode: connects to standalone Qdrant container or cloud cluster via `QDRANT_URL`.
- **Auto-Seeding**: On backend startup, checks if the database is populated. If empty, automatically seeds from `starter_kit/data/docs.json` and embeds chunks via Gemini into Qdrant.
- **Dynamic CRUD API**:
  - `GET /api/v1/docs`: List all indexed documents.
  - `POST /api/v1/docs`: Add or update a document, automatically triggering chunking and search index rebuild.
  - `DELETE /api/v1/docs/{doc_id}`: Remove a document and purge associated chunks.

---

## 5. API Specification (FastAPI)

- **Interactive Docs**: Swagger UI at `/docs`, ReDoc at `/redoc`.
- **Endpoints**:
  - `GET /api/v1/health`: Returns service health status and database connectivity.
  - `POST /api/v1/ask`:
    - Request: `{"question": string, "use_llm": boolean (default false), "confidence_threshold": float (optional)}`
    - Response:
      ```json
      {
        "status": "answered" | "not_covered",
        "answer": string,
        "doc_id": string | null,
        "doc_title": string | null,
        "confidence_score": float,
        "supporting_excerpt": string | null,
        "mode": "deterministic" | "llm"
      }
      ```
  - `GET /api/v1/docs`: Retrieve all stored documentation articles.
  - `POST /api/v1/docs`: Create/update documentation article.
  - `DELETE /api/v1/docs/{doc_id}`: Delete an article.

---

## 6. Client Interfaces

### 6.1 Streamlit Frontend (`streamlit_app.py`)
- Port: `8501`.
- **Q&A Chat View**:
  - Input field for admin queries.
  - Visual display of answers, confidence badges, source doc metadata, and highlighted context excerpts.
  - Toggle switch for deterministic vs LLM synthesis.
- **Benchmark Evaluation View**:
  - One-click runner for the 5 benchmark questions in `starter_kit/data/example_questions.json`.
  - Visual pass/fail badges, expected vs retrieved doc IDs, latency, and overall pass rate.
- **Document Management View**:
  - View existing documents.
  - Add new documentation articles directly to test live re-indexing.

### 6.2 CLI Client (`cli.py`)
- Standalone execution or client mode (can communicate with FastAPI server or run engine locally).
- Subcommands:
  - `python cli.py ask "How do I reset my password?"`: Single query output.
  - `python cli.py ask "Can I connect Solstice to Salesforce?" --json`: JSON output.
  - `python cli.py interactive`: Interactive REPL.
  - `python cli.py eval`: Runs automated batch test against `starter_kit/data/example_questions.json` and prints summary metrics.

---

## 7. AWS Hosting & Cloud Readiness

The project is structured for easy deployment to AWS:

1. **AWS ECS Fargate or AWS App Runner**:
   - `Dockerfile`: Multi-stage or clean Python environment container.
   - `docker-compose.yml`: Spins up FastAPI backend + Streamlit frontend.
2. **AWS EC2 / Lightsail**:
   - Clone repo, create virtualenv, install `requirements.txt`, run uvicorn and streamlit via systemd or supervisord.
3. **Database**:
   - SQLite for low-overhead instances.
   - Amazon RDS PostgreSQL enabled via `DATABASE_URL=postgresql://user:password@host:5432/dbname`.
4. **Configuration**:
   - Fully driven by `.env` / environment variables (`PORT`, `DATABASE_URL`, `BACKEND_URL`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).

---

## 8. Directory & File Layout

```
case-study-5-docs-qa-assistant/
├── AGENTS.md                  # This file
├── SOLUTION_DESIGN.md         # 1-2 page design document for case study submission
├── README.md                  # Setup, running, test instructions, and limitations
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Local & cloud orchestration
├── requirements.txt           # Python dependencies
├── .env.example               # Template environment configuration
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI application entrypoint
│   ├── config.py              # Settings & environment variables
│   ├── db/
│   │   ├── __init__.py
│   │   ├── session.py         # SQLAlchemy engine & session factory
│   │   ├── models.py          # Document & Chunk database models
│   │   └── seeder.py          # Auto-seeder from starter data
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── chunker.py         # Boundary-aware text chunking
│   │   ├── bm25.py            # BM25 retrieval index
│   │   ├── coverage.py        # Query Term Coverage anti-hallucination gate
│   │   ├── extractor.py       # Sentence-level answer pinpointing
│   │   ├── llm_client.py      # Standard library REST client (Gemini/OpenAI)
│   │   └── service.py         # Unified DocsQA engine service
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── qa.py              # Pydantic request/response schemas
│   └── api/
│       ├── __init__.py
│       └── routes.py          # FastAPI routes (/ask, /docs, /health)
├── streamlit_app.py           # Streamlit Web UI client
├── cli.py                     # CLI query tool & batch evaluator
├── tests/
│   ├── __init__.py
│   ├── test_engine.py         # Unit tests for chunker, bm25, coverage gate
│   ├── test_api.py            # Integration tests for FastAPI endpoints
│   └── test_benchmark.py      # Validation tests against example_questions.json
└── starter_kit/               # Original starter files & data
```

---

## 9. Evaluation Criteria & Verification Checklist

Before completing work, ensure:
1. `cli.py eval` runs against all 5 questions in `starter_kit/data/example_questions.json` and scores **100% pass**:
   - `Q-1`: maps to `D-01`
   - `Q-2`: maps to `D-01`
   - `Q-3`: maps to `D-03`
   - `Q-4`: maps to `D-06`
   - `Q-5`: rejected with `"I don't have an answer for that."`
2. FastAPI runs with `/docs` interactive documentation.
3. Streamlit app runs and connects to the backend.
4. Unit tests pass with `python -m pytest` or `python -m unittest`.
5. `SOLUTION_DESIGN.md` clearly documents assumptions, trade-offs, and future improvements.
