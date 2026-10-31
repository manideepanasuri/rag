# Starter Kit — Docs Q&A Assistant

## Contents

- `data/docs.json` — 6 synthetic documentation articles covering common product topics.
- `data/example_questions.json` — 5 example questions to test against, including one that isn't covered by any doc.
- `docs_qa_utils.py` — a small, optional Python utility module with basic chunking and keyword-matching helpers, provided as a rough starting point.

## A few notes

- You're free to use, modify, replace, or ignore `docs_qa_utils.py` entirely — it's not guaranteed to be complete, optimal, or bug-free.
- Not every question in `example_questions.json` has a good answer in the docs — take a look at how `expected_doc_id` is set for each one.
