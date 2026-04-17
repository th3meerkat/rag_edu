---
description: Launch backend (FastAPI) and/or frontend (Vite) dev servers
argument-hint: [--be | --fe]
---

Run `./start.sh $ARGUMENTS` from the project root using the Bash tool with `run_in_background: true`.

Flags (pass the user's `$ARGUMENTS` verbatim):
- (none) → start both
- `--be` → start only the backend (FastAPI on http://localhost:8000)
- `--fe` → start only the frontend (Vite on http://localhost:5173)

The full script would otherwise start:
- Backend on http://localhost:8000 (FastAPI + uvicorn with `--reload`)
- Frontend on http://localhost:5173 (Vite dev server)

After launching, report the background task ID and the URL(s) of the server(s) actually started. The user can stop them by killing the background task (which triggers the script's SIGTERM trap).

Do NOT run the script in the foreground — it is a long-running process and would block the conversation.
