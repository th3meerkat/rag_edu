#!/usr/bin/env python3
"""Embed (or refresh) a tutorial's i18n JSON inline inside its HTML, as a
`<script type="application/json" id="i18n-data">` block read by `i18n.js`.

Why this exists: opening the tutorials directly via `file://` (Finder, IDE
preview) blocks `fetch()` in most browsers, so the loader can't read the
external JSON. Embedding the JSON inline removes the fetch entirely while
keeping the external file as the editable source of truth.

Workflow:
    1. Edit  tutorials/i18n/<name>.json   (single source of truth).
    2. Run   `uv run python -m tools.i18n_sync ../tutorials/<name>.html`
       (from backend/, or pass multiple paths).
    3. Open the HTML directly with no server.

The script is idempotent: it replaces an existing inline block if found,
otherwise inserts a new one immediately before the `<script src="i18n.js">`
tag.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


INLINE_BLOCK_RE = re.compile(
    r'<script type="application/json" id="i18n-data">.*?</script>\s*',
    re.DOTALL,
)


def embed(html_path: Path, json_path: Path) -> tuple[int, str]:
    """Embed `json_path` content inline in `html_path`. Returns (key_count, status)."""
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)

    raw_html = html_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Re-serialize compact (one line per key for legibility, but compact spacing).
    # Escape `</` to keep the script tag from terminating early.
    payload = json.dumps(data, ensure_ascii=False, indent=2).replace("</", "<\\/")

    block = (
        '<script type="application/json" id="i18n-data">\n'
        + payload
        + "\n</script>\n"
    )

    if INLINE_BLOCK_RE.search(raw_html):
        # Use a lambda so re.sub() does not interpret backslash escapes in the
        # replacement (which would convert `\n` inside the embedded JSON into
        # real newlines and break the parser).
        new_html = INLINE_BLOCK_RE.sub(lambda _m: block, raw_html, count=1)
        status = "refreshed"
    else:
        # Insert just before the i18n.js script tag.
        m = re.search(r'<script src="i18n.js"></script>', raw_html)
        if not m:
            raise RuntimeError(
                f"{html_path}: no <script src='i18n.js'> tag found — was this HTML processed by i18n_extract?"
            )
        insert_at = m.start()
        new_html = raw_html[:insert_at] + block + raw_html[insert_at:]
        status = "inserted"

    html_path.write_text(new_html, encoding="utf-8")
    return len(data), status


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", nargs="+", help="One or more tutorial HTML paths")
    args = parser.parse_args(argv)

    rc = 0
    for h in args.html:
        html_path = Path(h).resolve()
        json_path = html_path.parent / "i18n" / (html_path.stem + ".json")
        try:
            n, status = embed(html_path, json_path)
            print(f"  {status}: {html_path.name}  ({n} keys from {json_path.name})")
        except Exception as e:
            print(f"  FAIL: {html_path.name}  →  {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
