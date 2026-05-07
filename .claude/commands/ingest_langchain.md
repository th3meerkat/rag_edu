---
description: Run the Langchain RAG ingestion (load PDFs, chunk, embed, store in ChromaDB)
---

Execute the Langchain RAG ingestion pipeline.

**Precondition**: ChromaDB must be running (`docker compose up -d chromadb`) and `OPENAI_API_KEY` must be set in `backend/.env`.

Run with the Bash tool (foreground, so the user sees the output):

```
cd backend && uv run python -m tools.ingest
```

Add `--reset` to drop the collection and manifest before reingesting (used when the chunk schema changes).

This will:
1. Scan `backend/app/data/` for PDFs and skip any already listed in `backend/app/data/ingested.json`
2. Split new PDFs into token-based chunks (aligned with the embedding model's tokenizer)
3. Generate embeddings with OpenAI `text-embedding-3-small` in batches
4. Upsert them into the `langchain_rag` Chroma collection with deterministic IDs (`source:page:chunk_idx`) so re-ingesting is idempotent
5. Record the processed filenames in `ingested.json`

Report the output to the user (totals of PDFs, chunks, and final embedding count).
