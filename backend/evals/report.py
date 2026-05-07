"""Self-contained HTML report with plotly. One file per run in evals/reports/."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import TOP_K_FINAL, TOP_K_PER_QUERY


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _bar_metrics(agg: pd.DataFrame, metric_cols: list[str], title: str) -> go.Figure:
    """One grouped bar chart comparing engines across metrics."""
    fig = go.Figure()
    for engine in agg.index:
        fig.add_bar(name=engine, x=metric_cols, y=[agg.loc[engine, c] for c in metric_cols])
    fig.update_layout(title=title, barmode="group", yaxis_range=[0, 1])
    return fig


def _bar_latency(mean_df: pd.DataFrame, median_df: pd.DataFrame) -> go.Figure:
    """Grouped bars for per-stage latency (mean and median side-by-side)."""
    stages = ["prepare_s", "retrieve_fuse_s", "rerank_s", "total_s"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Media (s)", "Mediana (s)"))
    for engine in mean_df.index:
        fig.add_bar(
            name=f"{engine} mean", x=stages,
            y=[mean_df.loc[engine, s] for s in stages],
            legendgroup=engine, row=1, col=1,
        )
        fig.add_bar(
            name=f"{engine} median", x=stages,
            y=[median_df.loc[engine, s] for s in stages],
            legendgroup=engine, showlegend=False, row=1, col=2,
        )
    fig.update_layout(title="Latencia por etapa", barmode="group")
    return fig


def build(
    metrics_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    judge_df: pd.DataFrame | None,
    out_dir: Path,
    timestamp: str,
) -> Path:
    """Write a standalone HTML file and return its path.

    `metrics_df` : rows = (engine, query_id), cols = hit@k / recall@k / mrr (retrieve and rerank).
    `latency_df` : rows = (engine, query_id), cols = prepare_s / retrieve_fuse_s / rerank_s / total_s.
    `judge_df`   : rows = (engine, query_id), col = `context_relevance_mean` (None if judge skipped).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"eval_{timestamp}.html"

    # --- Aggregations ---
    m_mean = metrics_df.groupby("engine").mean(numeric_only=True)
    lat_mean = latency_df.groupby("engine").mean(numeric_only=True)
    lat_median = latency_df.groupby("engine").median(numeric_only=True)

    # --- Figures ---
    retr_cols = [f"hit@{TOP_K_PER_QUERY}_retrieve", f"recall@{TOP_K_PER_QUERY}_retrieve",
                 f"mrr@{TOP_K_PER_QUERY}_retrieve"]
    rer_cols = [f"hit@{TOP_K_FINAL}_rerank", f"recall@{TOP_K_FINAL}_rerank",
                f"mrr@{TOP_K_FINAL}_rerank"]
    fig_retr = _bar_metrics(m_mean, retr_cols, "Métricas post-retrieve (ground truth)")
    fig_rer = _bar_metrics(m_mean, rer_cols, "Métricas post-rerank (ground truth)")
    fig_lat = _bar_latency(lat_mean, lat_median)

    figs_html = "\n".join(
        f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        for i, f in enumerate([fig_retr, fig_rer, fig_lat])
    )

    # --- Tables ---
    summary = m_mean.copy()
    if judge_df is not None:
        j = judge_df.groupby("engine")["context_relevance_mean"].mean()
        summary["context_relevance"] = j
    summary_html = summary.round(4).to_html(border=0, classes="tbl")

    per_query_html = (
        metrics_df.join(latency_df, rsuffix="_lat")
        .round(4)
        .to_html(border=0, classes="tbl")
    )

    sha = _git_sha()
    html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Eval · {timestamp}</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  h1, h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.25rem; }}
  .tbl {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; margin-bottom: 1rem; }}
  .tbl th, .tbl td {{ border-bottom: 1px solid #eee; padding: 4px 8px; text-align: right; }}
  .tbl th:first-child, .tbl td:first-child {{ text-align: left; font-family: monospace; }}
  code {{ background: #f4f4f4; padding: 0 4px; }}

  /* Floating glossary button */
  #glossary-btn {{
    position: fixed; top: 50%; right: 0; transform: translateY(-50%);
    background: #2563eb; color: white; border: none;
    padding: 12px 10px; border-radius: 8px 0 0 8px;
    cursor: pointer; font-size: 0.85rem; font-weight: 600;
    writing-mode: vertical-rl; letter-spacing: 0.05em;
    box-shadow: -2px 0 6px rgba(0,0,0,0.15); z-index: 1000;
  }}
  #glossary-btn:hover {{ background: #1d4ed8; }}

  /* Slide-out drawer */
  #glossary-drawer {{
    position: fixed; top: 0; right: 0; height: 100vh; width: min(420px, 90vw);
    background: white; box-shadow: -4px 0 16px rgba(0,0,0,0.18);
    transform: translateX(100%); transition: transform 0.28s ease;
    z-index: 1001; overflow-y: auto; padding: 1.5rem 1.5rem 3rem;
  }}
  #glossary-drawer.open {{ transform: translateX(0); }}
  #glossary-drawer h2 {{ margin-top: 0; }}
  #glossary-close {{
    position: absolute; top: 0.75rem; right: 1rem;
    background: none; border: none; font-size: 1.5rem;
    cursor: pointer; color: #666; line-height: 1;
  }}
  #glossary-close:hover {{ color: #000; }}

  /* Backdrop */
  #glossary-backdrop {{
    position: fixed; inset: 0; background: rgba(0,0,0,0.25);
    opacity: 0; pointer-events: none; transition: opacity 0.2s;
    z-index: 1000;
  }}
  #glossary-backdrop.open {{ opacity: 1; pointer-events: auto; }}

  dl dt {{ margin-top: 0.8rem; }}
  dl dd {{ margin: 0.2rem 0 0 1rem; color: #333; font-size: 0.92rem; }}
</style>
</head>
<body>
<h1>Retrieval eval · {timestamp}</h1>
<p><code>git={sha}</code> · queries={len(metrics_df.index.unique('query_id'))} · engines={", ".join(sorted(metrics_df.index.unique('engine')))}</p>

<h2>Resumen por engine</h2>
{summary_html}

<h2>Gráficos</h2>
{figs_html}

<h2>Detalle por query</h2>
{per_query_html}

<button id="glossary-btn" onclick="toggleGlossary()">📖 Glosario de métricas</button>
<div id="glossary-backdrop" onclick="toggleGlossary()"></div>
<aside id="glossary-drawer" aria-hidden="true">
  <button id="glossary-close" onclick="toggleGlossary()" aria-label="Cerrar">×</button>
  <h2>Glosario de métricas</h2>
  <p>Todas las métricas se calculan <strong>por query</strong> contra el <em>ground truth</em> (conjunto de <code>doc_id</code> relevantes del <code>golden set</code>) y luego se <strong>promedian sobre todas las queries</strong> por engine (lo que se muestra en la tabla "Resumen por engine"). El sufijo <code>_retrieve</code> evalúa la salida del retriever (top-{TOP_K_PER_QUERY}); <code>_rerank</code> evalúa la salida final tras el reranker (top-{TOP_K_FINAL}).</p>
  <dl>
    <dt><strong>Hit@k</strong></dt>
    <dd><em>Por query</em>: 1 si al menos un documento relevante aparece en el top-k, 0 si no. Mide <em>cobertura binaria</em>: ¿la respuesta está ahí?<br/>
    <em>Ejemplo (query)</em>: relevantes = {{d3}}, top-5 = [d1, d3, d5, d7, d9] → Hit@5 = <code>1</code>.<br/>
    <em>Agregado</em>: el valor en la tabla es la media sobre queries → equivale a la <strong>fracción de queries</strong> con al menos un relevante en el top-k. Ej: <code>Hit@5 = 0.7</code> ⇒ 70% de las queries tuvieron éxito.</dd>

    <dt><strong>Recall@k</strong></dt>
    <dd><em>Por query</em>: fracción de documentos relevantes recuperados dentro del top-k: <code>|relevantes ∩ top-k| / |relevantes|</code>. Mide <em>exhaustividad</em>.<br/>
    <em>Ejemplo (query)</em>: relevantes = {{d1, d2, d3}}, top-5 = [d1, d3, d5, d7, d9] → Recall@5 = 2/3 ≈ <code>0.67</code>.<br/>
    <em>Agregado</em>: media de los Recall@k por query.</dd>

    <dt><strong>MRR@k</strong> (Mean Reciprocal Rank)</dt>
    <dd><em>Por query</em>: <code>1 / rank</code> del primer documento relevante dentro del top-k (0 si no aparece). Premia que lo relevante esté <em>alto</em> en el ranking.<br/>
    <em>Ejemplo (query)</em>: relevantes = {{d3}}, top-5 = [d5, d3, d7, d1, d9] → rank = 2 → RR = <code>0.5</code>.<br/>
    <em>Agregado</em>: media de los RR por query (de ahí el nombre <em>Mean</em> Reciprocal Rank).</dd>

    <dt><strong>context_relevance</strong> (LLM-as-judge, opcional)</dt>
    <dd><em>Por query</em>: fracción de chunks recuperados que un LLM local (Gemma 2) considera relevantes para responder la pregunta. No depende del ground truth — útil cuando el golden set es pequeño o incompleto.<br/>
    <em>Ejemplo (query)</em>: 5 chunks recuperados, el juez marca 3 como relevantes → <code>0.6</code>.<br/>
    <em>Agregado</em>: media sobre queries.</dd>
  </dl>
</aside>
<script>
  function toggleGlossary() {{
    const d = document.getElementById('glossary-drawer');
    const b = document.getElementById('glossary-backdrop');
    const open = d.classList.toggle('open');
    b.classList.toggle('open', open);
    d.setAttribute('aria-hidden', open ? 'false' : 'true');
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') {{
      const d = document.getElementById('glossary-drawer');
      if (d.classList.contains('open')) toggleGlossary();
    }}
  }});
</script>
</body>
</html>"""
    out.write_text(html, encoding="utf-8")
    return out
