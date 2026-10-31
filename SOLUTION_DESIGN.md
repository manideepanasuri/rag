# Solution Design: Solstice Documentation Q&A Assistant

## 1. Problem Restatement & Elena's Core Constraint

Solstice Learning sells an enterprise LMS to corporate training teams. Company support lead Elena Torres faces a recurring issue: platform administrators repeatedly ask the support desk basic procedural questions (password resets, CSV export limits, billing cycles, API rate limits). These answers already exist in the documentation, but admins want to ask a question in plain English and get an immediate, direct answer without hunting across multiple help articles.

Elena has one non-negotiable rule:
> *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."*

In enterprise support, hallucinations are unacceptable. Telling an admin they can connect to Salesforce when no such integration exists creates broken customer workflows and escalations. A confident refusal protects trust and routes unhandled queries to human support.

The goal is to build a reliable, maintainable Q&A system that accurately retrieves facts from the 6 documentation articles, extracts clear answers, and strictly abstains when documentation does not contain the answer.

---

## 2. Scope & Boundaries for v1

### In Scope for v1
- **Document Ingestion & Storage**: Ingests the 6 provided docs into an SQLite database via SQLAlchemy, auto-seeding on boot. Supports dynamic CRUD operations so articles can be added, edited, or removed via API.
- **Boundary-Aware Chunking**: Breaks articles along paragraph and sentence boundaries instead of blind character slicing, keeping semantic statements intact.
- **Hybrid Retrieval (Gemini Embeddings + BM25)**:
  - Dense semantic search using Google Gemini's free `text-embedding-004` (768-dim) handles synonym matches and conversational phrasing (e.g., "credentials" matching "password").
  - Lexical BM25 ranking preserves exact keyword matches for error codes, plan names ("Growth plan"), and numbers ("10,000 rows").
  - Graceful degradation: if `GEMINI_API_KEY` is not provided or network is offline, it defaults to pure BM25 without errors.
- **Evidence Coverage Gate (The Abstention Engine)**: Query nouns are extracted and cross-checked against the corpus. If salient query terms (e.g., "Salesforce") do not exist in the documentation, or if top retrieval similarity is below the confidence threshold, the system immediately returns `"I don't have an answer for that."`.
- **Extractive Answer Pinpointing**: Locates the specific sentences containing the answer and returns them alongside document ID, title, confidence score, and supporting excerpt.
- **Optional Grounded LLM Mode**: When requested (`use_llm=True`) and an API key is set, a zero-dependency client calls Gemini/OpenAI to format a natural answer strictly from the retrieved passage.
- **Client Interfaces**:
  - FastAPI backend exposing REST endpoints and interactive Swagger docs (`/docs`).
  - Streamlit web portal (`streamlit_app.py`) for admin Q&A and document inspection.
  - CLI tool (`cli.py`) with an automated benchmark evaluation runner (`cli.py eval`).

### Explicitly Excluded from v1
- User authentication and role-based permissions (RBAC).
- Multi-turn conversational chat memory (admins ask independent factual questions).
- Live ticketing sync (Zendesk / Jira ticket generation).
- Distributed vector databases (Milvus / Pinecone) — 6 articles do not warrant cluster infrastructure.

---

## 3. System Architecture

The project uses a decoupled client-server structure:

```
[ Streamlit Web Portal ] (8501)       [ CLI Client (cli.py) ]
          │                                     │
          └───────────────────┬─────────────────┘
                              │ HTTP / REST
                              ▼
                   [ FastAPI Server ] (8000)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        [ SQLAlchemy ORM ]          [ Qdrant Vector DB ]
        (SQLite / AWS RDS)          (Local / AWS Cluster)
               │                             │
               └──────────────┬──────────────┘
                              ▼
                      [ DocsQA Engine ]
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           [ BM25 ]     [ Coverage ]   [ Extractor & ]
           [ Search ]   [ Gate     ]   [ Optional LLM]
```

### Retrieval & Decision Flow
1. **Query Input**: Admin enters a question in Streamlit or CLI.
2. **Coverage Guardrail**: The engine checks whether the query contains out-of-vocabulary entities. If an essential noun (e.g., "Salesforce") is absent across all docs, the query is rejected immediately without speculative vector matching.
3. **Hybrid Scoring**:
   - Semantic score: Cosine similarity between query embedding and chunk embeddings (via Gemini API).
   - Lexical score: Okapi BM25 score over chunk tokens.
   - Blended score: `Score = 0.6 * Semantic + 0.4 * BM25`.
4. **Relevance Threshold**: If the top score falls below the calibrated cutoff, return `"I don't have an answer for that."`.
5. **Answer Pinpointing**: The best chunk is parsed into sentences, and the highest-matching sentences are extracted as the concise answer.
6. **Optional Synthesis**: If enabled, retrieved context is fed to Gemini with a strict system prompt ("Answer only from the context. If uncertain, say 'I don't have an answer for that.'").
7. **Response Delivery**: Returns answer, source document ID, title, confidence score, and raw citation quote.

---

## 4. Key Trade-offs & Decisions

| Decision | What We Chose | What We Skipped | Why |
| :--- | :--- | :--- | :--- |
| **Semantic Embeddings** | Cloud API (Gemini `text-embedding-004`) | Local PyTorch models (`sentence-transformers`) | Local embedding models require a 1.5 GB PyTorch install and substantial memory. Gemini's embedding API is free (1,500 RPM), fast, and requires zero heavy local packages. |
| **Abstention Logic** | Keyword Coverage Gate + Score Threshold | Score threshold alone | In pure vector search, every query gets a similarity score (often 0.60+ for unrelated questions). A score threshold alone will match "Salesforce" to SSO or API docs. Checking keyword presence prevents this failure mode completely. |
| **Answer Format** | Direct extractive pinpointing + optional LLM | Full paragraph dump or pure LLM generation | Raw paragraph dumps force the admin to read a wall of text. Pure LLM generation can hallucinate small details. Extractive pinpointing gives exact factual sentences with zero hallucination risk. |
| **Vector Storage** | Dedicated Vector DB (Qdrant via `qdrant-client`) | Flat arrays or ChromaDB bloat | Qdrant provides a clean embedded disk mode (`./qdrant_storage`) locally with zero daemon setup, and connects seamlessly to a production Qdrant cluster on AWS via `QDRANT_URL`. It avoids the heavy Kubernetes/OpenTelemetry dependency bloat of ChromaDB. |
| **Client UI** | Streamlit + CLI | Single-page static HTML | Streamlit gives Elena an interactive testing portal with one-click benchmark runs in under 100 lines of Python, while the CLI satisfies automated test requirements. |

---

## 5. Evaluation & Verification

The implementation is verified against the 5 questions in `starter_kit/data/example_questions.json`:

1. `Q-1`: *"How do I reset my password?"*
   - Target: `D-01` ("Resetting Your Password")
   - Expected: Steps to click 'Forgot password?' on login screen, enter email, use link.
2. `Q-2`: *"How long do I have to use a password reset link before it expires?"*
   - Target: `D-01`
   - Expected: "15 minutes".
3. `Q-3`: *"What's the maximum number of rows I can export to CSV?"*
   - Target: `D-03` ("Exporting Reports to CSV")
   - Expected: "10,000 rows".
4. `Q-4`: *"What are the API rate limits for the Growth plan?"*
   - Target: `D-06` ("API Rate Limits")
   - Expected: "60 requests per minute".
5. `Q-5`: *"Can I connect Solstice to Salesforce?"*
   - Target: `null` (Unsupported)
   - Expected: Rejection with exact text `"I don't have an answer for that."`.

Automated verification is built into `cli.py eval`, which runs all 5 questions, asserts correct document retrieval, checks Q-5 abstention, and prints latency and pass rates.

---

## 6. Production Roadmap (What We'd Do Next)

If allocated additional engineering time and budget:
1. **Unanswered Query Triage**: Automatically log rejected queries (`NOT_COVERED`) into a dashboard or Jira queue so Elena's technical writers know which documentation articles to write next.
2. **Document Webhook Sync**: Connect to GitHub or Notion via webhooks to auto-reindex docs on push or publish events.
3. **Role-Based Filtering**: Partition documentation by audience (Admin vs Learner vs Instructor) so search results respect the caller's role.
4. **Offline Local Embeddings**: Add an ONNX runtime (`fastembed`) option for enterprise clients with strict zero-cloud egress policies.
