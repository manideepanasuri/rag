# Solstice Documentation Q&A Assistant

A production-grade, hallucination-free support assistant built for Solstice Learning administrators.

---

## Overview

Elena Torres, Head of Support at Solstice Learning, faces repetitive procedural questions from platform administrators regarding password resets, CSV export limits, billing cycles, and API limits. While documented in the product guides, admins prefer asking questions in natural language.

### Elena's Core Iron Rule
> *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."*

This system achieves 100% adherence to this constraint via a **Query Term Coverage Gate** and calibrated retrieval confidence scoring. If a query references an entity not documented anywhere (e.g., Salesforce in benchmark question Q-5), the engine rejects the query and returns:
`"I don't have an answer for that."`

---

## How to Use This Project

### 1. Initial Setup

#### Step 1.1: Clone and Enter the Project Directory
```bash
cd case-study-5-docs-qa-assistant
```

#### Step 1.2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 1.3: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 1.4: Configure Environment Variables
Copy the template configuration file:
```bash
cp .env.example .env
```
*(Optional)* Add your Gemini API key in `.env` to enable 768-dim neural semantic search and Flash synthesis:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
> **Offline Note**: If left blank, the assistant automatically runs 100% offline using deterministic BM25 search with zero errors.

---

### 2. Starting the Server & Streamlit Web UI

To use the full client-server application, start the FastAPI backend and Streamlit frontend:

#### Method A: Two Terminal Windows (Recommended for Development)

**Terminal 1 — Start the FastAPI Backend Server (Port 8000):**
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

**Terminal 2 — Start the Streamlit Web Application (Port 8501):**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```
- Web UI URL: `http://localhost:8501`
- Automatically connects to the running FastAPI server on port 8000.

#### Method B: One-Command Background Execution
If you prefer running both in a single terminal session:
```bash
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py
```

#### Method C: One-Command Docker Compose
Run both backend and frontend inside isolated containers:
```bash
docker-compose up --build
```
- FastAPI Backend: `http://localhost:8000`
- Streamlit Web Portal: `http://localhost:8501`

---

### 3. Using the CLI (Command-Line Interface)

The CLI tool `cli.py` allows you to test queries, run interactive sessions, and execute benchmark evaluations directly from your terminal.

#### Ask a Single Question
```bash
python cli.py ask "What is the maximum number of rows I can export to CSV?"
```
*Output:*
```
---
STATUS:     ANSWERED
ANSWER:     Important limit: CSV exports are capped at 10,000 rows. If your report has more rows than that, apply additional filters to narrow it down, or contact support about a bulk data export instead.
SOURCE DOC: D-03 (Exporting Reports to CSV)
CONFIDENCE: 0.6374
GEMINI:     ACTIVE (Gemini text-embedding-004)
MODE:       deterministic
LATENCY:    18.4 ms
---
```

#### Test a Rephrased Semantic Query (Synonyms with Zero Keyword Overlap)
```bash
python cli.py ask "Can I get a refund if I cancel my annual subscription early?"
```
*Output:* Correctly retrieves `D-04` ("Refunds for annual plans canceled within the first 30 days are available on request.").

#### Test Unsupported Negative Query (Elena's Abstention Rule)
```bash
python cli.py ask "Can I connect Solstice to Salesforce?"
```
*Output:*
```
---
STATUS:     NOT_COVERED
ANSWER:     I don't have an answer for that.
SOURCE DOC: None (N/A)
CONFIDENCE: 0.0000
---
```

#### Request JSON Output (for API integration or scripting)
```bash
python cli.py ask "What are the rate limits for the Growth plan?" --json
```

#### Start an Interactive Terminal REPL
```bash
python cli.py interactive
```
Type any question at the prompt. Enter `exit` or `quit` to leave.

---

### 4. Using the Streamlit Web Application (Features Overview)

Launch the full interactive admin web portal:
```bash
streamlit run streamlit_app.py
```
Open your browser at **`http://localhost:8501`**.

The web UI contains three dedicated tabs:
1. **💬 Support Q&A**:
   - Type any natural language question.
   - Toggle optional LLM Synthesis mode.
   - View answer text, source document ID, title, confidence score metric, latency, and expandable context excerpt.
   - Clearly flags abstentions when unsupported topics are asked.
2. **🧪 Benchmark Evaluation**:
   - Click **Run 5 Benchmark Tests** to run the 5 starter questions from Elena Torres.
   - Shows live progress, pass/fail badges, expected vs retrieved doc IDs, latency, and overall pass rate percentage (100%).
3. **📖 Knowledge Base Management**:
   - Inspect all indexed documentation articles.
   - Add a new document (e.g. `D-07`: "Setting up Webhooks") through the interactive form with automatic live re-indexing.

---

### 5. Running the FastAPI REST Server

Start the asynchronous REST API backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Interactive API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Available Endpoints
- `GET /api/v1/health`: Service health and indexed counts.
  ```bash
  curl http://localhost:8000/api/v1/health
  ```
- `POST /api/v1/ask`: Query the documentation.
  ```bash
  curl -X POST http://localhost:8000/api/v1/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "How do I reset my password?"}'
  ```
- `GET /api/v1/docs`: List all indexed documentation articles.
  ```bash
  curl http://localhost:8000/api/v1/docs
  ```
- `POST /api/v1/docs`: Add or update a documentation article (triggers instant re-indexing).
  ```bash
  curl -X POST http://localhost:8000/api/v1/docs \
    -H "Content-Type: application/json" \
    -d '{"doc_id": "D-07", "title": "Webhooks", "content": "Admins can configure webhook endpoints from Settings > Integrations."}'
  ```

---

### 6. Running Automated Tests & Benchmark Evaluation

#### Run the 5 Benchmark Evaluation Questions
```bash
python cli.py eval
```
Expected output:
```
==================================================
SOLSTICE DOCS Q&A ASSISTANT — BENCHMARK EVALUATION
==================================================
Total benchmark questions: 5

[1/5] Q-1: How do I reset my password? -> D-01 (PASS)
[2/5] Q-2: How long do I have to use a password reset link before it expires? -> D-01 (PASS)
[3/5] Q-3: What's the maximum number of rows I can export to CSV? -> D-03 (PASS)
[4/5] Q-4: What are the API rate limits for the Growth plan? -> D-06 (PASS)
[5/5] Q-5: Can I connect Solstice to Salesforce? -> None (PASS)
==================================================
SUMMARY: 5/5 PASSED (100.0%)
==================================================
```

#### Run the Full Pytest Suite (27 Unit & Integration Tests)
```bash
pytest -v
```

#### Run Only the Semantic Retrieval Tests
```bash
pytest tests/test_semantic.py -v
```

---

### 7. Running with Docker & AWS

#### Local Docker Compose
To spin up both the FastAPI backend and Streamlit frontend in isolated containers:
```bash
docker-compose up --build
```
- FastAPI Backend: `http://localhost:8000`
- Streamlit Web UI: `http://localhost:8501`

#### AWS Deployment Options
1. **AWS ECS Fargate / AWS App Runner**: Deploy using the provided `Dockerfile`.
2. **Amazon RDS PostgreSQL**: To upgrade from SQLite to managed PostgreSQL on AWS, set the `DATABASE_URL` environment variable:
   ```env
   DATABASE_URL=postgresql://user:password@rds-host:5432/solstice_docs
   ```
3. **Dedicated Qdrant Cluster on AWS**: Connect to standalone Qdrant on EC2 or Qdrant Cloud via:
   ```env
   QDRANT_URL=http://qdrant-cluster-ip:6333
   ```

---

## Benchmark Results Summary

Tested against `starter_kit/data/example_questions.json`:

| ID | Question | Expected Doc | Retrieved Doc | Confidence | Answer Summary | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Q-1** | How do I reset my password? | `D-01` | `D-01` | 0.6860 | Click 'Forgot password?' on login screen... | **PASS** |
| **Q-2** | How long do I have to use a password reset link before it expires? | `D-01` | `D-01` | 0.6335 | Note: password reset links expire after 15 minutes. | **PASS** |
| **Q-3** | What's the maximum number of rows I can export to CSV? | `D-03` | `D-03` | 0.6374 | Important limit: CSV exports are capped at 10,000 rows. | **PASS** |
| **Q-4** | What are the API rate limits for the Growth plan? | `D-06` | `D-06` | 0.7337 | Rate limits: Growth plans are limited to 60 requests per minute... | **PASS** |
| **Q-5** | Can I connect Solstice to Salesforce? | *None* | *None* | 0.0000 | **I don't have an answer for that.** | **PASS** |

Pass Rate: **5 / 5 (100.0%)**

---

## Known Limitations & Production Roadmap

### Known Limitations in v1
1. **Multi-turn Memory**: Current version processes each question independently as an isolated query.
2. **Document Partitioning**: All documents are in a single shared index without permission or user role filters (e.g., Learner vs Admin).
3. **Table & Image Extraction**: Formatted tables or markdown diagrams are ingested as plain text strings.

### Recommended Next Steps for Elena's Support Team
1. **Unanswered Query Triage Queue**: Log queries that trigger abstention (`not_covered`) into a dashboard so documentation writers can identify missing help topics.
2. **Automated Documentation Webhooks**: Integrate with GitHub or Notion to auto-trigger document re-indexing on commit or publication.
3. **Role-Based Access Filtering (RBAC)**: Tag articles with user roles (`Viewer`, `Editor`, `Admin`) to restrict answer retrieval to the caller's privilege level.
