---
description: Run the backend regression test suite with coverage
---

Execute the backend pytest regression suite.

**Preconditions**: ChromaDB running on `localhost:8001`, reranker on `localhost:8002`, and `OPENAI_API_KEY` set in `backend/.env`.

Run with the Bash tool (foreground, so the user sees the output):

```
backend/tools/run_tests.sh
```

Extra pytest flags can be appended, e.g. `backend/tools/run_tests.sh -k test_retrieve -v`.

Report to the user:
- Tests passed / failed / skipped
- Coverage percentage per module
- Any failures with the relevant traceback snippet
