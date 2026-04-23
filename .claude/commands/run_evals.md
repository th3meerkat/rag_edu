---
description: Run the retrieval-only RAG evaluation (both engines, HTML report)
argument-hint: [--no-judge]
---

Run the retrieval eval suite: both engines over the golden set at `backend/evals/datasets/principito_golden.jsonl`, ground-truth metrics (Hit@k, Recall@k, MRR) + optional LLM-as-judge context relevance via local Gemma 2, HTML report written to `backend/evals/reports/`.

`run_evals.sh` self-bootstraps everything:
- Starts the Ollama daemon in background if it is not already alive (and kills it on exit if it did so).
- Verifies `gemma2:2b` is installed — if not, exits with the `ollama pull` command.
- Auto-ingests missing Chroma collections (drops the manifest first so it re-runs fresh).
- Preloads Gemma 2 into the daemon before metrics start, unloads it at the end (option B: daemon stays alive, model RAM freed).

**The only external prereqs are**: ChromaDB on `localhost:8001`, reranker on `localhost:8002`, `OPENAI_API_KEY` in `backend/.env`, and the `ollama` CLI installed (unless using `--no-judge`).

Run with the Bash tool in the **foreground** (the user wants to see progress logs and summary):

```
./run_evals.sh $ARGUMENTS
```

After it finishes, report to the user:
- Path to the HTML report (`backend/evals/reports/eval_<timestamp>.html`).
- The summary table printed at the end (mean metrics per engine).
- Any obvious gaps (engine with noticeably lower Hit@k or MRR).
