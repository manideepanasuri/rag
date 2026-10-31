import json
import os
import time
import urllib.request
import streamlit as st

# Sync Streamlit secrets to environment and settings if present
try:
    if hasattr(st, "secrets"):
        for key in ["GEMINI_API_KEY", "OPENAI_API_KEY", "CONFIDENCE_THRESHOLD", "BACKEND_URL", "DATABASE_URL"]:
            if key in st.secrets and st.secrets[key]:
                val = str(st.secrets[key])
                os.environ[key] = val
except Exception:
    pass

from app.config import settings

# Ensure settings picks up any newly synced keys
if os.getenv("GEMINI_API_KEY") and not settings.gemini_api_key:
    settings.gemini_api_key = os.getenv("GEMINI_API_KEY")

from app.db.models import Document
from app.db.seeder import seed_docs_if_empty
from app.db.session import Base, SessionLocal, engine
from app.engine.service import DocsQAService

st.set_page_config(
    page_title="Solstice Docs Q&A Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_service():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_docs_if_empty(db)
    service = DocsQAService()
    service.initialize(db)
    return service, db


def ask_api_or_service(question: str, use_llm: bool = False):
    # Try backend HTTP first
    try:
        url = f"{settings.backend_url}/api/v1/ask"
        payload = json.dumps({"question": question, "use_llm": use_llm}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        # Fall back to cached local service
        service, _ = get_service()
        return service.ask(question, use_llm=use_llm)


def fetch_docs():
    try:
        url = f"{settings.backend_url}/api/v1/docs"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8")), True
    except Exception:
        service, db = get_service()
        docs = service.list_docs(db)
        return [{"doc_id": d.id, "title": d.title, "content": d.content} for d in docs], False


def post_doc(doc_id: str, title: str, content: str):
    try:
        url = f"{settings.backend_url}/api/v1/docs"
        payload = json.dumps({"doc_id": doc_id, "title": title, "content": content}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        service, db = get_service()
        service.add_or_update_doc(db=db, doc_id=doc_id, title=title, content=content)
        return {"status": "ok"}


st.title("📚 Solstice Learning — Documentation Q&A")
st.caption("Factual, hallucination-free support assistant for Solstice Learning platform administrators.")

with st.sidebar:
    st.header("⚙️ Engine Status")
    gemini_key_present = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
    if gemini_key_present:
        st.success("✨ Gemini Semantic Vectors: ACTIVE")
        st.caption("Model: gemini-embedding-001 (768-dim)")
    else:
        st.warning("⚙️ Gemini: NOT CONFIGURED")
        st.caption("Running pure deterministic BM25 search (Offline)")

    # Test backend connection
    is_backend_online = False
    try:
        req = urllib.request.Request(f"{settings.backend_url}/api/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as r:
            is_backend_online = (r.status == 200)
    except Exception:
        is_backend_online = False

    if is_backend_online:
        st.success(f"🟢 Backend API: CONNECTED ({settings.backend_url})")
    else:
        st.info("ℹ️ Backend API: STANDALONE MODE (running engine in-process)")

tab_qa, tab_eval, tab_kb = st.tabs([
    "💬 Support Q&A",
    "🧪 Benchmark Evaluation (5 Cases)",
    "📖 Knowledge Base Management",
])

# ----------------- TAB 1: Support Q&A -----------------
with tab_qa:
    st.subheader("Ask a Product Question")
    st.write(
        "Enter any question about passwords, CSV exports, billing, API limits, or integrations. "
        "If the docs do not cover it, the assistant strictly abstains."
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input(
            "Admin Question",
            value="How long do I have to use a password reset link before it expires?",
            label_visibility="collapsed",
            placeholder="Type your question here...",
        )
    with col2:
        use_llm = st.toggle("LLM Synthesis", value=False, help="Requires GEMINI_API_KEY")

    if st.button("Submit Question", type="primary"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching documentation..."):
                start_time = time.perf_counter()
                res = ask_api_or_service(user_question, use_llm=use_llm)
                latency = (time.perf_counter() - start_time) * 1000

            st.divider()
            status = res.get("status")
            confidence = res.get("confidence_score", 0.0)
            answer = res.get("answer", "")
            doc_id = res.get("doc_id")
            doc_title = res.get("doc_title")
            excerpt = res.get("supporting_excerpt")
            mode = res.get("mode", "deterministic")

            if status == "answered":
                st.success("✅ Documented Answer Found")
                st.markdown(f"### Answer\n> **{answer}**")

                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Source Document", f"{doc_id}: {doc_title}")
                col_b.metric("Confidence Score", f"{confidence:.3f}")
                col_c.metric("Latency", f"{latency:.1f} ms")
                col_d.metric("Engine", "Gemini 768d" if res.get("gemini_enabled") else "Offline BM25")

                if excerpt:
                    with st.expander("📄 Supporting Documentation Excerpt", expanded=True):
                        st.info(excerpt)
            else:
                st.error("🚫 Abstention Triggered (Out of Scope)")
                st.markdown(f"### Answer\n> **{answer}**")
                st.info(
                    "Elena's Rule Enforced: This topic is not covered in Solstice documentation. "
                    "The assistant strictly abstains rather than hallucinating."
                )
                col_x, col_y = st.columns(2)
                col_x.metric("Latency", f"{latency:.1f} ms")
                col_y.metric("Engine", "Gemini 768d" if res.get("gemini_enabled") else "Offline BM25")

# ----------------- TAB 2: Benchmark Evaluation -----------------
with tab_eval:
    st.subheader("Elena's 5 Benchmark Questions")
    st.write(
        "Runs automated verification against `starter_kit/data/example_questions.json`. "
        "Validates that Q-1 through Q-4 retrieve the exact target documents, and Q-5 strictly abstains."
    )

    if st.button("Run 5 Benchmark Tests", type="primary"):
        bench_path = "starter_kit/data/example_questions.json"
        if not os.path.exists(bench_path):
            bench_path = os.path.join(
                os.path.dirname(__file__), "starter_kit", "data", "example_questions.json"
            )

        with open(bench_path, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        questions = bench_data.get("example_questions", [])
        passed_count = 0
        results_rows = []

        progress_bar = st.progress(0)
        for idx, q in enumerate(questions):
            q_id = q["question_id"]
            q_text = q["question"]
            expected_doc = q.get("expected_doc_id")

            t0 = time.perf_counter()
            res = ask_api_or_service(q_text, use_llm=False)
            dt = (time.perf_counter() - t0) * 1000

            actual_doc = res.get("doc_id")
            status = res.get("status")
            ans = res.get("answer")

            passed = False
            if expected_doc is not None:
                passed = (actual_doc == expected_doc and status == "answered")
            else:
                passed = (actual_doc is None and status == "not_covered" and ans == "I don't have an answer for that.")

            if passed:
                passed_count += 1

            results_rows.append({
                "ID": q_id,
                "Question": q_text,
                "Expected": expected_doc or "None (Abstain)",
                "Retrieved": actual_doc or "None (Abstain)",
                "Verdict": "PASS ✅" if passed else "FAIL ❌",
                "Confidence": f"{res.get('confidence_score', 0.0):.3f}",
                "Latency (ms)": f"{dt:.1f}",
                "Extracted Answer": ans,
            })
            progress_bar.progress((idx + 1) / len(questions))

        pass_rate = (passed_count / len(questions)) * 100
        if passed_count == len(questions):
            st.success(f"🎉 100% Pass Rate: {passed_count}/{len(questions)} passed successfully!")
        else:
            st.error(f"⚠️ Pass Rate: {passed_count}/{len(questions)} ({pass_rate:.1f}%)")

        for r in results_rows:
            with st.container():
                st.markdown(f"**[{r['ID']}] {r['Question']}** — {r['Verdict']}")
                st.caption(f"Expected: {r['Expected']} | Retrieved: {r['Retrieved']} | Confidence: {r['Confidence']} | Latency: {r['Latency (ms)']} ms")
                st.markdown(f"> *{r['Extracted Answer']}*")
                st.divider()

# ----------------- TAB 3: Knowledge Base Management -----------------
with tab_kb:
    st.subheader("Manage Documentation Articles")
    docs, is_live_api = fetch_docs()

    if is_live_api:
        st.success(f"🟢 Syncing with FastAPI REST Server ({settings.backend_url})")
    else:
        st.info("ℹ️ Running in Standalone In-Process Mode (FastAPI server not detected on port 8000)")

    st.write(f"Currently indexing **{len(docs)} documents**.")

    for d in docs:
        d_id = d.get("doc_id", "")
        d_title = d.get("title", "")
        d_content = d.get("content", "")
        with st.expander(f"📄 {d_id}: {d_title}"):
            st.write(d_content)

    st.divider()
    st.subheader("Add / Update Documentation Article")
    with st.form("add_doc_form"):
        new_doc_id = st.text_input("Document ID", value="D-07", placeholder="e.g. D-07")
        new_doc_title = st.text_input("Title", placeholder="e.g. Setting up Webhooks")
        new_doc_content = st.text_area(
            "Content",
            placeholder="Type or paste the documentation article text here...",
            height=120,
        )
        submitted = st.form_submit_button("Index Document")
        if submitted:
            if not new_doc_id.strip() or not new_doc_title.strip() or not new_doc_content.strip():
                st.error("All fields are required.")
            else:
                post_doc(
                    doc_id=new_doc_id.strip(),
                    title=new_doc_title.strip(),
                    content=new_doc_content.strip(),
                )
                st.success(f"Successfully indexed document {new_doc_id}! Search indices updated dynamically.")
                st.rerun()
