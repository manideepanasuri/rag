# FDE Take-Home Case Study: Docs Q&A Assistant

## Before you start

This exercise mirrors a real Forward Deployed Engineer (FDE) engagement: a customer brings you a genuine but partially-specified problem, and you're expected to (1) clarify/assume what's needed, (2) propose a solution, (3) build a working version of it, and (4) present and defend your approach to the customer (in this case, our panel).

**What we're evaluating:** how you handle ambiguity, the soundness of your design judgment, your ability to build a working solution under real constraints, how clearly you communicate your reasoning, and how you respond when your approach is challenged. We are **not** grading for a "perfect" or exhaustive solution — a well-reasoned, appropriately-scoped solution with clearly stated trade-offs is exactly what we want to see.

## The customer

**Solstice Learning** sells an online-course platform to corporate training teams. Their Head of Support, Elena Torres, reached out with the following ask:

> "New admins ask us the same handful of questions over and over — password resets, export limits, billing cycles, that kind of thing — and it's all already in our docs, they just don't want to dig through pages to find it. I want something where an admin can type a plain-English question and get a real answer pulled from our documentation.
>
> The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up. What would you build for us?"

**Note:** Elena's ask is intentionally incomplete. You will need to decide things like: how to search across documents to find the right passage, how confident an answer needs to be before you show it versus saying you don't know, and whether (and how) to use an LLM at all versus a simpler search approach. **Document whatever assumptions you make.**

## What we're providing you

- A seed dataset of 6 realistic (synthetic) documentation articles, plus 5 example questions to test against.
- A minimal starter repository scaffold with basic chunking and keyword-matching helpers, so you're not starting from a completely blank page.
- *(See `starter_kit/` alongside this brief.)*

## What you have full freedom on

- **Language/stack** — use whatever you're most productive in.
- **Approach** — a well-executed keyword/retrieval-based search is a completely valid answer here; using an LLM is optional (see below), not required.
- **Scope** — you decide what's in v1 versus "future work." A smaller, well-reasoned scope beats an ambitious, half-finished one.

## Time expectation

We expect this to take **roughly 3-4 focused hours**, with a full week of calendar time to fit it in. If you're well past 4-5 hours, **stop** — use your README to describe what you'd tackle next.

## Required deliverables

1. **`SOLUTION_DESIGN.md`** (1-2 pages): the problem restated in your own words, your assumptions, your proposed approach, the trade-offs you considered, and what you'd do with more time/data/budget.
2. **Working implementation** — should run end-to-end against the provided questions and produce a clear answer (or an honest "I don't know") for each.
3. **`README.md`** — how to run it, known limitations.

## Optional bonus track (not required, never penalized if skipped)

You're welcome to use an LLM to generate the final answer text from the retrieved passage(s), rather than just returning raw excerpts — as long as the answer stays grounded in what the docs actually say. This is a **plus signal**, not a gate; a well-built non-AI retrieval solution is a completely acceptable submission.

## What happens after you submit

We'll review your submission ahead of a **30-45 minute live session**: a walkthrough of your design doc and a short demo, followed by Q&A, including "what if..." questions and the occasional new constraint introduced mid-conversation.

## Submission

Send us a link to your repository (or a zip archive) containing your code, `SOLUTION_DESIGN.md`, and `README.md`.
