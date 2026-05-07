---
description: Extract bilingual (.lang-es/.lang-en) content from a tutorial HTML to a JSON i18n file and rewrite the HTML to use data-i18n references
---

Run the deterministic i18n extractor. Use when adding a new bilingual tutorial under `tutorials/`, or to re-derive a JSON when content has changed.

**Preconditions**: the source HTML uses the `.lang-es` / `.lang-en` pattern that `body.lang-en .lang-es { display:none }` keys off. The shared loader at `tutorials/i18n.js` must exist (already does).

Run with the Bash tool (foreground), passing the HTML path:

```
cd backend && uv run python -m tools.i18n_extract ../tutorials/<file>.html
```

The tool writes:

- `tutorials/i18n/<file>.json` — flat dict with `{key: {es, en}}` entries.
- Modified `tutorials/<file>.html` — bilingual elements collapsed to a single tag with `data-i18n="key"`. Bilingual `<img>` pairs (different `src` per language) become a single `<img>` with `data-i18n-attr-src="key"`. Bilingual zoom buttons collapse similarly. Dead `body.lang-X .lang-Y` CSS rules are removed. The toggle script is wired with `i18n.init(...)` and `i18n.setLang(...)` calls.
- Re-embeds the JSON inline as a `<script type="application/json" id="i18n-data">` so the page works on `file://` (no fetch).

Verify after running:

```
python3 -c "import json,re; h=open('tutorials/<file>.html').read(); d=json.load(open('tutorials/i18n/<file>.json')); ks=set(re.findall(r'data-i18n[a-z\-]*=\"([^\"]+)\"',h)); print('html_keys',len(ks),'json_keys',len(d),'missing',len(ks-set(d)),'extra',len(set(d)-ks))"
```

Both counts should match and missing/extra should be 0.

To refresh the inline JSON after editing the JSON file directly (no HTML changes), use the sync command:

```
cd backend && uv run python -m tools.i18n_sync ../tutorials/<file>.html
```

Edge cases worth eyeballing:

- Headers that originally had `<span class="lang-es">…</span><span class="lang-en">…</span>` collapse to a single `<span data-i18n="…">`. The `<h2 id="…">` keeps its id.
- Code blocks with bilingual comments are extracted as a single key whose value contains the entire `<pre><code>…</code></pre>` innerHTML, including the `<span class="tok-…">` syntax tokens.
- The wiring patches (`<script src="i18n.js">`, `i18n.init(...)`, `i18n.setLang(lang)`) are idempotent: re-running on an already-patched HTML is a no-op.

Report to the user:
- Number of keys extracted.
- Final HTML line count vs original.
- Any HTML/JSON key mismatch.
