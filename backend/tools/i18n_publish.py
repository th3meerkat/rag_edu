#!/usr/bin/env python3
"""Apply 'ready-for-public-Internet' patches to the tutorial HTMLs.

Idempotent — re-runnable any time. Each patch is gated by an "is this already
applied?" check so running the script twice is a no-op.

Patches applied per tutorial:
  1. <head> Open Graph + Twitter Card meta tags (per-tutorial og:image, title,
     description) so LinkedIn shows a rich preview.
  2. <head> favicon links (16/32 PNGs + apple-touch-icon).
  3. CSS additions: print stylesheet, mobile touch-target bump for the toggle
     pill, cross-tutorial sub-menu styling.
  4. <body> a small "tutorial siblings" sub-menu fixed top-left with the four
     tutorial entries, current one highlighted.
  5. <footer> attribution: author, publication date, GitHub link, versions
     pinned (LangChain / LlamaIndex / Python).

Site config lives in the SITE dict below — adjust there once and re-run.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INFO_DIR = REPO_ROOT / "tutorials"
I18N_DIR = INFO_DIR / "i18n"

SITE = {
    "base_url": "https://th3meerkat.github.io/rag_edu",
    "github_url": "https://github.com/th3meerkat/rag_edu",
    "author": "Diego",
    "published": "2026-05-07",
    "versions": "LangChain 0.3.x · LlamaIndex 0.12.x · Python 3.13",
}

# Per-tutorial publication metadata.
TUTORIALS = {
    "rag.html": {
        "slug": "rag",
        "og_image": "imgs/og_rag.png",
        "og_title_es": "Cómo funciona un RAG por dentro",
        "og_title_en": "How a RAG actually works",
        "og_desc_es": "Conceptos de RAG: pipeline canónico, chunking, embeddings, retrieval, reranking, generación, memoria, evaluación.",
        "og_desc_en": "RAG concepts: canonical pipeline, chunking, embeddings, retrieval, reranking, generation, memory, evaluation.",
    },
    "langchain.html": {
        "slug": "langchain",
        "og_image": "imgs/og_langchain.png",
        "og_title_es": "Aprender LangChain leyendo un RAG real",
        "og_title_en": "Learn LangChain by reading a real RAG",
        "og_desc_es": "Recorrido por un servicio RAG real construido con LangChain — LCEL, Runnable, RunnableWithMessageHistory, decisiones grises explicadas.",
        "og_desc_en": "A walk through a real RAG service built with LangChain — LCEL, Runnable, RunnableWithMessageHistory, gray-area decisions explained.",
    },
    "llamaindex.html": {
        "slug": "llamaindex",
        "og_image": "imgs/og_llamaindex.png",
        "og_title_es": "Aprender LlamaIndex leyendo un RAG real",
        "og_title_en": "Learn LlamaIndex by reading a real RAG",
        "og_desc_es": "Recorrido por un servicio RAG real construido con LlamaIndex — VectorStoreIndex, CustomQueryEngine, retriever custom sobre Chroma.",
        "og_desc_en": "A walk through a real RAG service built with LlamaIndex — VectorStoreIndex, CustomQueryEngine, custom retriever over Chroma.",
    },
    "rag_faq.html": {
        "slug": "rag_faq",
        "og_image": "imgs/og_rag_faq.png",
        "og_title_es": "Troubleshooting y FAQ de un RAG",
        "og_title_en": "RAG troubleshooting & FAQ",
        "og_desc_es": "Pitfalls clásicos, FAQ, y un mapa de síntomas → causas posibles para cuando un RAG en producción se porta raro.",
        "og_desc_en": "Classic pitfalls, FAQ, and a symptom-to-likely-cause map for when a production RAG starts behaving oddly.",
    },
}

SIBLINGS_NAV_HTML = """<nav class="siblings" aria-label="Tutorial siblings">
  <a href="rag.html"        data-slug="rag"        ><span data-i18n="siblings.rag"></span></a>
  <a href="langchain.html"  data-slug="langchain"  ><span data-i18n="siblings.langchain"></span></a>
  <a href="llamaindex.html" data-slug="llamaindex" ><span data-i18n="siblings.llamaindex"></span></a>
  <a href="rag_faq.html"    data-slug="rag_faq"    ><span data-i18n="siblings.rag_faq"></span></a>
</nav>
"""

SIBLINGS_LABELS = {
    "siblings.rag":        {"es": "Conceptos",        "en": "Concepts"},
    "siblings.langchain":  {"es": "LangChain",        "en": "LangChain"},
    "siblings.llamaindex": {"es": "LlamaIndex",       "en": "LlamaIndex"},
    "siblings.rag_faq":    {"es": "Troubleshooting",  "en": "Troubleshooting"},
}

EXTRA_CSS = """
  /* ======= Sibling-tutorials extras :: managed by i18n_publish ======= */
  /* ======= Sibling-tutorials nav ====================================== */
  .siblings {
    position: fixed;
    top: 18px;
    left: 18px;
    z-index: 1200;
    display: inline-flex;
    flex-direction: column;
    gap: 2px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    font-size: 12px;
    font-weight: 600;
    user-select: none;
  }
  .siblings a {
    padding: 6px 14px;
    border-radius: 10px;
    color: var(--text-muted);
    text-decoration: none;
    transition: background 0.15s ease, color 0.15s ease;
    letter-spacing: 0.04em;
    white-space: nowrap;
  }
  .siblings a:hover { color: var(--text); background: var(--accent-soft); }
  .siblings a[data-slug="__CURRENT_SLUG__"] {
    background: var(--accent);
    color: #fff;
  }
  @media (max-width: 720px) {
    .siblings {
      top: 64px;          /* below the toggle pill */
      left: 8px;
      font-size: 11px;
    }
    .siblings a { padding: 5px 10px; }
  }

  /* ======= Mobile touch target for toggle pill ======================== */
  @media (max-width: 720px) {
    .lang-toggle button {
      padding: 8px 14px;
      min-height: 36px;
    }
  }

  /* ======= Print stylesheet =========================================== */
  @media print {
    .lang-toggle, .siblings, .lang-toggle-mobile,
    .breadcrumb, .back-link, .zoom-btn, .modal-overlay,
    .cross-cta { display: none !important; }
    body { font-size: 12pt; line-height: 1.5; }
    .container { max-width: none; padding: 0; }
    a { color: inherit; text-decoration: underline; }
    figure, table, pre { page-break-inside: avoid; }
    h2, h3 { page-break-after: avoid; }
  }
  /* ======= /Sibling-tutorials extras ================================== */"""

FOOTER_BLOCK_TEMPLATE = """<footer>
  <span data-i18n="{body_key}"></span>
  <p class="footer-meta">
    <span data-i18n="footer.author"></span>
    · <a href="{github_url}" target="_blank" rel="noopener">GitHub</a>
    · <span data-i18n="footer.versions"></span>
  </p>
</footer>"""

FOOTER_CSS = """
  footer .footer-meta {
    margin-top: 8px;
    font-size: 12px;
    color: var(--text-muted);
    opacity: 0.85;
  }
  footer .footer-meta a {
    color: var(--text-muted);
    text-decoration: none;
    border-bottom: 1px dotted var(--text-muted);
  }
  footer .footer-meta a:hover { color: var(--accent); border-bottom-color: var(--accent); }"""


def build_meta_block(t: dict, base_url: str) -> str:
    page_url = f"{base_url}/{[k for k, v in TUTORIALS.items() if v['slug'] == t['slug']][0]}"
    return f"""<!-- Open Graph / Twitter Cards (auto-managed by tools/i18n_publish.py) -->
<meta property="og:type" content="article">
<meta property="og:site_name" content="rag_edu">
<meta property="og:url" content="{page_url}">
<meta property="og:title" content="{t['og_title_en']}">
<meta property="og:description" content="{t['og_desc_en']}">
<meta property="og:image" content="{base_url}/{t['og_image']}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{t['og_title_en']}">
<meta name="twitter:description" content="{t['og_desc_en']}">
<meta name="twitter:image" content="{base_url}/{t['og_image']}">
<link rel="canonical" href="{page_url}">
<!-- /Open Graph -->
"""


FAVICON_BLOCK = """<!-- Favicons (auto-managed) -->
<link rel="icon" type="image/png" sizes="16x16" href="imgs/favicon-16.png">
<link rel="icon" type="image/png" sizes="32x32" href="imgs/favicon-32.png">
<link rel="apple-touch-icon" sizes="180x180" href="imgs/apple-touch-icon.png">
<!-- /Favicons -->
"""


META_BLOCK_RE = re.compile(
    r"<!-- Open Graph / Twitter Cards \(auto-managed[\s\S]*?<!-- /Open Graph -->\s*",
)
FAVICON_BLOCK_RE = re.compile(
    r"<!-- Favicons \(auto-managed\)[\s\S]*?<!-- /Favicons -->\s*",
)
EXTRA_CSS_BLOCK_RE = re.compile(
    r"  /\* ======= Sibling-tutorials extras :: managed by i18n_publish ======= \*/[\s\S]*?  /\* ======= /Sibling-tutorials extras ================================== \*/\s*",
)
FOOTER_CSS_RE = re.compile(
    r"  footer \.footer-meta \{.*?\}\s*",
    re.DOTALL,
)


def patch_html(path: Path, t: dict) -> tuple[bool, list[str]]:
    """Apply all per-tutorial publication patches. Returns (changed, log)."""
    src = path.read_text(encoding="utf-8")
    out = src
    log: list[str] = []

    # 1. <head> meta block — replace if present, insert before </head> otherwise.
    meta_block = build_meta_block(t, SITE["base_url"])
    if META_BLOCK_RE.search(out):
        out = META_BLOCK_RE.sub(meta_block, out, count=1)
        log.append("meta refreshed")
    else:
        out = out.replace("</head>", meta_block + "</head>", 1)
        log.append("meta inserted")

    # 2. Favicon links.
    if FAVICON_BLOCK_RE.search(out):
        out = FAVICON_BLOCK_RE.sub(FAVICON_BLOCK, out, count=1)
    else:
        out = out.replace("</head>", FAVICON_BLOCK + "</head>", 1)
        log.append("favicons inserted")

    # 3. Extra CSS — sibling nav, mobile touch target, print stylesheet.
    extra_css = EXTRA_CSS.replace("__CURRENT_SLUG__", t["slug"])
    if EXTRA_CSS_BLOCK_RE.search(out):
        out = EXTRA_CSS_BLOCK_RE.sub(extra_css, out, count=1)
    else:
        out = out.replace("</style>", extra_css + "\n</style>", 1)
        log.append("extra CSS inserted")

    # 4. Footer CSS .footer-meta.
    if FOOTER_CSS_RE.search(out):
        out = FOOTER_CSS_RE.sub(FOOTER_CSS, out, count=1)
    else:
        out = out.replace("</style>", FOOTER_CSS + "\n</style>", 1)

    # 5. Sibling nav: insert just inside <body>, before everything else.
    if 'class="siblings"' not in out:
        out = re.sub(
            r"(<body[^>]*>\s*)",
            r"\1" + SIBLINGS_NAV_HTML,
            out,
            count=1,
        )
        log.append("siblings nav inserted")

    # 6. Footer rewrite — preserve the existing body i18n key (varies per file)
    # and add the meta line. Match any single <span data-i18n="..."> inside <footer>.
    if 'class="footer-meta"' not in out:
        m = re.search(
            r'<footer>\s*<span data-i18n="([^"]+)"></span>\s*</footer>', out
        )
        if m:
            new_footer = FOOTER_BLOCK_TEMPLATE.format(
                github_url=SITE["github_url"], body_key=m.group(1)
            )
            out = out.replace(m.group(0), new_footer, 1)
            log.append("footer attribution added")

    if out != src:
        path.write_text(out, encoding="utf-8")
        return True, log
    return False, ["unchanged"]


def patch_json(json_path: Path, slug: str) -> None:
    """Inject sibling-nav i18n labels + footer.author / footer.versions keys."""
    d = json.loads(json_path.read_text(encoding="utf-8"))

    for k, v in SIBLINGS_LABELS.items():
        d[k] = v

    d["footer.author"] = {
        "es": f"Por {SITE['author']} · publicado el {SITE['published']}",
        "en": f"By {SITE['author']} · published {SITE['published']}",
    }
    d["footer.versions"] = {
        "es": f"Versiones probadas: {SITE['versions']}",
        "en": f"Tested versions: {SITE['versions']}",
    }

    json_path.write_text(
        json.dumps(d, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    print("== publish-patch all tutorials ==")
    for fname, meta in TUTORIALS.items():
        html_path = INFO_DIR / fname
        json_path = I18N_DIR / (meta["slug"] + ".json")
        if not html_path.exists() or not json_path.exists():
            print(f"  skip {fname} (missing file)")
            continue
        changed, log = patch_html(html_path, meta)
        patch_json(json_path, meta["slug"])
        print(f"  {fname}: {', '.join(log)}")

    # Re-sync inline JSON in every HTML so the embedded i18n picks up the new keys.
    print("\n== inline JSON re-sync ==")
    from tools.i18n_sync import embed
    for fname, meta in TUTORIALS.items():
        html_path = INFO_DIR / fname
        json_path = I18N_DIR / (meta["slug"] + ".json")
        if not html_path.exists() or not json_path.exists():
            continue
        n, status = embed(html_path, json_path)
        print(f"  {fname}: {status} ({n} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
