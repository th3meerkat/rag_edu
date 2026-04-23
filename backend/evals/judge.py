"""LLM-as-judge for Context Relevance — local via Ollama + Gemma 2.

Binary rubric: per (query, chunk) pair the judge returns 1 if the chunk is
relevant to answer the query, else 0. We aggregate to the mean per query.

The judge must be cheap AND as deterministic as possible:
  - `temperature=0` and a fixed seed for reproducibility.
  - Plain-text structured response (prefix `VEREDICTO: 0|1`) so parsing survives
    small model quirks.
"""
from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

import ollama

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gemma2:2b"
JUDGE_SEED = 42

_PROMPT_TEMPLATE = """\
Eres un evaluador imparcial de sistemas de búsqueda. Debes decidir si el \
FRAGMENTO recuperado contiene información útil y directamente relacionada \
para responder la PREGUNTA del usuario.

Criterios:
- Responde 1 si el FRAGMENTO aporta información relevante para responder la PREGUNTA.
- Responde 0 si es irrelevante, tangencial o no aporta a la respuesta.

PREGUNTA:
{query}

FRAGMENTO:
{chunk}

Responde en una sola línea con el formato exacto:
VEREDICTO: 0
o
VEREDICTO: 1
"""

_VEREDICTO_RE = re.compile(r"VEREDICTO\s*:\s*([01])", re.IGNORECASE)


@dataclass
class JudgeUnavailable(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def preflight() -> None:
    """Fail fast with an actionable message if the Ollama daemon or the model
    aren't ready. Called once at the start of the run."""
    try:
        models = ollama.list().get("models", [])
    except Exception as e:  # connection refused → daemon not running
        raise JudgeUnavailable(
            f"Ollama daemon no responde en 127.0.0.1:11434 ({e}). "
            "Arrancalo con `ollama serve` o abriendo la app de macOS."
        ) from e

    names = {m.get("model") or m.get("name") for m in models}
    if not any(JUDGE_MODEL in (n or "") for n in names):
        raise JudgeUnavailable(
            f"El modelo '{JUDGE_MODEL}' no está instalado. "
            f"Instalalo con: `ollama pull {JUDGE_MODEL}`"
        )


def parse_veredicto(text: str) -> int | None:
    """Parse the 0/1 verdict from a judge response; None if unparseable."""
    m = _VEREDICTO_RE.search(text)
    if m is None:
        return None
    return int(m.group(1))


def load_model() -> None:
    """Explicitly load the model in the daemon before metrics start, so the
    load time (~1-2s) is not counted as part of the per-query judge latency.

    `keep_alive="-1"` pins the model in memory until we explicitly unload with
    `stop_model()` at the end of the run."""
    ollama.generate(
        model=JUDGE_MODEL,
        prompt="",
        options={"num_predict": 1},
        keep_alive=-1,
    )


def score_chunk(query: str, chunk_text: str) -> int | None:
    """Ask Gemma 2 whether the chunk is relevant to the query. Returns 0/1
    or None if the response couldn't be parsed."""
    resp = ollama.generate(
        model=JUDGE_MODEL,
        prompt=_PROMPT_TEMPLATE.format(query=query, chunk=chunk_text),
        options={"temperature": 0.0, "seed": JUDGE_SEED, "num_predict": 16},
        keep_alive=-1,
    )
    return parse_veredicto(resp.get("response", ""))


def stop_model() -> None:
    """Release the model's RAM after the eval run (option B in the plan).

    The daemon stays alive. We shell out to `ollama stop` because the Python
    client doesn't expose an unload method in this version."""
    try:
        subprocess.run(
            ["ollama", "stop", JUDGE_MODEL],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except Exception as e:
        logger.warning("Could not stop Ollama model '%s': %s", JUDGE_MODEL, e)
