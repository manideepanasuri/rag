import argparse
import json
import os
import sys
import time
import urllib.request
from typing import Any

from app.config import settings
from app.db.seeder import seed_docs_if_empty
from app.db.session import Base, SessionLocal, engine
from app.engine.service import DocsQAService


def get_service_or_backend():
    """
    Checks if FastAPI backend is responding on settings.backend_url.
    If not, returns an in-process initialized DocsQAService.
    """
    try:
        req = urllib.request.Request(
            f"{settings.backend_url}/api/v1/health",
            headers={"User-Agent": "CLI"},
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return "backend"
    except Exception:
        pass

    # Fallback to in-process service
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    seed_docs_if_empty(db)
    service = DocsQAService()
    service.initialize(db)
    return service


def query_qa(
    question: str,
    use_llm: bool = False,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    handler = get_service_or_backend()
    if handler == "backend":
        url = f"{settings.backend_url}/api/v1/ask"
        payload = {
            "question": question,
            "use_llm": use_llm,
            "confidence_threshold": confidence_threshold,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=35.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    else:
        return handler.ask(
            question=question,
            use_llm=use_llm,
            confidence_threshold=confidence_threshold,
        )


def cmd_ask(args):
    start = time.perf_counter()
    result = query_qa(
        question=args.question,
        use_llm=args.use_llm,
        confidence_threshold=args.threshold,
    )
    elapsed = (time.perf_counter() - start) * 1000

    if args.json:
        result["latency_ms"] = round(elapsed, 2)
        print(json.dumps(result, indent=2))
        return

    print("---")
    print(f"STATUS:     {result['status'].upper()}")
    print(f"ANSWER:     {result['answer']}")
    print(f"SOURCE DOC: {result.get('doc_id') or 'None'} ({result.get('doc_title') or 'N/A'})")
    print(f"CONFIDENCE: {result['confidence_score']:.4f}")
    gemini_str = "ACTIVE (Gemini gemini-embedding-001)" if result.get("gemini_enabled") else "INACTIVE (Offline BM25)"
    print(f"GEMINI:     {gemini_str}")
    model_name = result.get("llm_model")
    mode_str = f"llm (Synthesized by Gemini Flash: {model_name or 'gemini-flash-lite-latest'})" if result.get("mode") == "llm" else "deterministic (Extracted from Documentation)"
    print(f"MODE:       {mode_str}")
    print(f"LATENCY:    {elapsed:.1f} ms")
    if result.get("supporting_excerpt"):
        print(f"EXCERPT:    {result['supporting_excerpt']}")
    print("---")


def cmd_interactive(args):
    print("SOLSTICE DOCS Q&A ASSISTANT — INTERACTIVE REPL")
    print("Type your questions below. Enter 'exit' or 'quit' to end.\n")

    while True:
        try:
            prompt = input("Question > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            result = query_qa(question=prompt, use_llm=args.use_llm)
            gemini_str = "ACTIVE" if result.get("gemini_enabled") else "INACTIVE (BM25)"
            print("\nANSWER:")
            print(f"  {result['answer']}")
            print(f"SOURCE:     {result.get('doc_id') or 'None'} - {result.get('doc_title') or 'N/A'}")
            print(f"CONFIDENCE: {result['confidence_score']:.4f}")
            print(f"GEMINI:     {gemini_str}")
            print(f"STATUS:     {result['status'].upper()}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def cmd_eval(args):
    eval_path = args.file or "starter_kit/data/example_questions.json"
    if not os.path.exists(eval_path):
        eval_path = os.path.join(
            os.path.dirname(__file__), "starter_kit", "data", "example_questions.json"
        )

    with open(eval_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    questions = bench_data.get("example_questions", [])
    total = len(questions)
    passed = 0

    print("==================================================")
    print("SOLSTICE DOCS Q&A ASSISTANT — BENCHMARK EVALUATION")
    print("==================================================")
    print(f"Total benchmark questions: {total}\n")

    for idx, q in enumerate(questions, 1):
        q_id = q["question_id"]
        q_text = q["question"]
        expected_doc = q.get("expected_doc_id")

        start = time.perf_counter()
        res = query_qa(question=q_text, use_llm=args.use_llm)
        elapsed = (time.perf_counter() - start) * 1000

        actual_doc = res.get("doc_id")
        status = res.get("status")
        answer = res.get("answer", "")

        is_correct = False
        if expected_doc is not None:
            if actual_doc == expected_doc and status == "answered":
                is_correct = True
        else:
            if actual_doc is None and status == "not_covered" and answer == "I don't have an answer for that.":
                is_correct = True

        if is_correct:
            passed += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        print(f"[{idx}/{total}] {q_id}: {q_text}")
        print(f"  VERDICT:      {verdict}")
        print(f"  EXPECTED DOC: {expected_doc}")
        print(f"  RETRIEVED:    {actual_doc}")
        print(f"  CONFIDENCE:   {res.get('confidence_score', 0.0):.4f}")
        print(f"  LATENCY:      {elapsed:.1f} ms")
        print(f"  ANSWER:       {answer}")
        print()

    print("==================================================")
    pass_rate = (passed / total) * 100 if total > 0 else 0
    print(f"SUMMARY: {passed}/{total} PASSED ({pass_rate:.1f}%)")
    print("==================================================")

    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Solstice Documentation Q&A CLI Client and Benchmark Evaluator"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ask command
    ask_p = subparsers.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("question", help="Question to ask")
    ask_p.add_argument("--json", action="store_true", help="Output as JSON")
    ask_p.add_argument("--use-llm", action="store_true", help="Use LLM synthesis if available")
    ask_p.add_argument("--threshold", type=float, default=None, help="Confidence threshold")

    # Interactive command
    inter_p = subparsers.add_parser("interactive", help="Start interactive Q&A session")
    inter_p.add_argument("--use-llm", action="store_true", help="Use LLM synthesis if available")

    # Eval command
    eval_p = subparsers.add_parser("eval", help="Run automated benchmark evaluation")
    eval_p.add_argument("--file", help="Path to example_questions.json")
    eval_p.add_argument("--use-llm", action="store_true", help="Use LLM synthesis if available")

    args = parser.parse_args()
    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
