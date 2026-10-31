import json
import os
import pytest
from app.db.seeder import seed_docs_if_empty
from app.db.session import Base, SessionLocal, engine
from app.engine.service import DocsQAService


@pytest.fixture(scope="module")
def qa_service():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_docs_if_empty(db)
    service = DocsQAService()
    service.initialize(db)
    yield service
    db.close()


def test_q1_password_reset(qa_service):
    res = qa_service.ask("How do I reset my password?")
    assert res["status"] == "answered"
    assert res["doc_id"] == "D-01"
    assert "reset link" in res["answer"].lower() or "forgot password" in res["answer"].lower()


def test_q2_password_expiration(qa_service):
    res = qa_service.ask("How long do I have to use a password reset link before it expires?")
    assert res["status"] == "answered"
    assert res["doc_id"] == "D-01"
    assert "15 minutes" in res["answer"]


def test_q3_csv_export_limit(qa_service):
    res = qa_service.ask("What's the maximum number of rows I can export to CSV?")
    assert res["status"] == "answered"
    assert res["doc_id"] == "D-03"
    assert "10,000" in res["answer"] or "10,000 rows" in res["answer"]


def test_q4_api_growth_rate_limit(qa_service):
    res = qa_service.ask("What are the API rate limits for the Growth plan?")
    assert res["status"] == "answered"
    assert res["doc_id"] == "D-06"
    assert "60 requests per minute" in res["answer"]


def test_q5_salesforce_abstention(qa_service):
    res = qa_service.ask("Can I connect Solstice to Salesforce?")
    assert res["status"] == "not_covered"
    assert res["doc_id"] is None
    assert res["answer"] == "I don't have an answer for that."


def test_all_benchmark_dataset(qa_service):
    bench_path = "starter_kit/data/example_questions.json"
    if not os.path.exists(bench_path):
        bench_path = os.path.join(
            os.path.dirname(__file__), "..", "starter_kit", "data", "example_questions.json"
        )
    with open(bench_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for q in data["example_questions"]:
        res = qa_service.ask(q["question"])
        expected = q["expected_doc_id"]
        if expected is not None:
            assert res["doc_id"] == expected, f"Failed for {q['question_id']}"
            assert res["status"] == "answered"
        else:
            assert res["doc_id"] is None, f"Failed abstention for {q['question_id']}"
            assert res["status"] == "not_covered"
            assert res["answer"] == "I don't have an answer for that."
