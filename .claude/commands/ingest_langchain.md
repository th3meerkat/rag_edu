---
description: Run the Langchain RAG ingestion (load PDFs, chunk, embed, store in ChromaDB)
---

Execute the Langchain RAG ingestion pipeline.

**Precondition**: ChromaDB must be running (`docker compose up -d chromadb`) and `OPENAI_API_KEY` must be set in `backend/.env`.

Run with the Bash tool (foreground, so the user sees the output):

```
cd backend && uv run python -c "from app.services.langchain_rag.service import LangchainSrv; LangchainSrv().run_ingestion()"
```

This will:
1. Scan `backend/app/data/` for PDFs and skip any already listed in `backend/app/data/ingested.txt`
2. Split new PDFs into overlapping chunks (size=1000, overlap=200)
3. Generate embeddings with OpenAI `text-embedding-3-small`
4. Append them to the `langchain_rag` collection in ChromaDB (with `source` and `page` metadata)
5. Record the newly processed filenames in `ingested.txt` — the collection is never dropped

Report the output to the user (totals of PDFs, chunks, and final embedding count).
