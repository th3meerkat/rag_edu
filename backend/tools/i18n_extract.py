#!/usr/bin/env python3
"""Extract bilingual `.lang-es`/`.lang-en` content from a tutorial HTML into a
JSON i18n file, and rewrite the HTML to use `data-i18n` / `data-i18n-attr-*`
references read by `tutorials/i18n.js`.

Patterns handled:

  1. Adjacent siblings with `class="lang-es"` and `class="lang-en"`
     (e.g. two `<p>`, two `<table>`, two `<ol>`, two `<div class="callout">`,
     two `<figcaption>`, two `<text>` inside an SVG).
     → replaced with a single element using `data-i18n="key"` (innerHTML swap).

  2. Inline span pairs inside a header
     (e.g. `<h2 id="x"><span class="lang-es">A</span><span class="lang-en">B</span></h2>`).
     → collapsed to a single `<span data-i18n="key">`.

  3. `<img class="lang-es" src="A">` followed by `<img class="lang-en" src="B">`
     (different `src` per language).
     → single `<img>` with `data-i18n-attr-src="key"`.

  4. `<button class="zoom-btn lang-es" data-zoom-src="A">label-es</button>`
     followed by the matching `lang-en` button.
     → single `<button>` with `data-i18n="key.label"` and
       `data-i18n-attr-data-zoom-src="key.src"`.

Section context: the script walks the document in order and tracks the most
recent `<h2 id="...">`; keys are generated as
`section.<slug>.<kind>.<counter>` where `<slug>` is the id with `-`→`_`.
For elements outside any section (hero, toc, footer), keys use stable names
(`hero.title`, `meta.level`, `toc.item.<n>`, `footer.body`, etc.).

Side effects on the HTML, beyond the bilingual replacements:
  * `<script src="i18n.js"></script>` is inserted before the existing toggle
    script (idempotent).
  * The toggle's `setLang(lang)` gets a `if (window.i18n) window.i18n.setLang(lang);`
    call inserted before the `localStorage.setItem` line (idempotent).
  * After the initial `setLang(...)` invocation in the toggle IIFE, an
    `i18n.init(...)` call is inserted (idempotent).
  * The CSS rules `body.lang-en .lang-es { display: none !important; }`
    and `body.lang-es .lang-en { display: none !important; }` are removed.

Usage:
    uv run python -m tools.i18n_extract <html_path>

The script writes the JSON next to `<html_dir>/i18n/<basename>.json` and
rewrites the HTML in place.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag


# ---------- helpers ----------

def class_of(tag: Tag) -> set[str]:
    cls = tag.get("class")
    if cls is None:
        return set()
    if isinstance(cls, str):
        return set(cls.split())
    return set(cls)


def is_lang_es(tag: Tag) -> bool:
    return "lang-es" in class_of(tag)


def is_lang_en(tag: Tag) -> bool:
    return "lang-en" in class_of(tag)


def strip_lang_class(tag: Tag) -> None:
    """Remove `lang-es` / `lang-en` from the class attr; drop the attr if empty."""
    cls = class_of(tag) - {"lang-es", "lang-en"}
    if cls:
        tag["class"] = sorted(cls)
    elif tag.has_attr("class"):
        del tag["class"]


def slugify_id(id_str: str) -> str:
    return id_str.replace("-", "_") if id_str else "unknown"


def section_for(tag: Tag) -> Optional[Tag]:
    """The nearest preceding `<h2 id=...>` for a given element, or None."""
    cur = tag
    while cur is not None:
        for prev in cur.find_all_previous("h2"):
            if prev.get("id"):
                return prev
            return None
        # walk up to the parent if no preceding h2 found
        cur = cur.parent
    return None


def header_classification(tag: Tag) -> Optional[str]:
    """Return a stable key prefix for elements outside sections (hero/toc/footer/intro).

    Checks the element itself first (a `<nav class="toc">` paired with another
    is the TOC) and then its ancestors. Falls back to `intro` for elements that
    sit after the hero but before the first `<h2>` — typically the lead
    paragraphs and the first illustration.
    """
    # element itself
    if tag.name == "nav" and "toc" in class_of(tag):
        return "toc"
    if tag.name == "header" and "hero" in class_of(tag):
        return "hero"
    if tag.name == "footer":
        return "footer"

    # ancestors
    for anc in tag.parents:
        if not isinstance(anc, Tag):
            continue
        if anc.name == "header" and "hero" in class_of(anc):
            return "hero"
        if anc.name == "nav" and "toc" in class_of(anc):
            return "toc"
        if anc.name == "footer":
            return "footer"

    # If we got here, the element is between the hero and the first <h2 id=...>
    # (or it's truly orphaned). Treat post-hero / pre-section content as `intro`.
    if section_for(tag) is None:
        return "intro"

    return None


# ---------- key generation ----------

class KeyGen:
    def __init__(self):
        # counters[section_slug][kind] → int
        self.counters: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def next(self, section: str, kind: str) -> str:
        self.counters[section][kind] += 1
        n = self.counters[section][kind]
        return f"{section}.{kind}.{n}"

    def fixed(self, key: str) -> str:
        # No counter — caller asserts uniqueness.
        return key


def kind_for_pair(es_tag: Tag) -> str:
    """Coarse element-type classifier for key naming."""
    name = es_tag.name
    if name in ("h2",): return "heading"
    if name in ("h3",): return "subheading"
    if name in ("h4",): return "subsubheading"
    if name == "p":     return "body.p"
    if name == "ol" or name == "ul": return "list"
    if name == "li":    return "list_item"
    if name == "table": return "table"
    if name == "div" and "callout" in class_of(es_tag): return "callout"
    if name == "div" and "meta"    in class_of(es_tag): return "meta"
    if name == "div" and "pipeline-step" in class_of(es_tag): return "pipeline_step"
    if name == "figcaption": return "figcaption"
    if name == "figure":    return "figure"
    if name == "pre":       return "code"
    if name == "span":      return "span"
    if name == "text":      return "svg_text"
    if name == "img":       return "figure_src"
    if name == "button":    return "cta"
    if name == "th" or name == "td": return "cell"
    return name


# ---------- main extraction ----------

def extract(html: str) -> tuple[str, dict]:
    soup = BeautifulSoup(html, "html.parser")
    dictionary: dict[str, dict[str, str]] = {}
    keygen = KeyGen()

    # --- Step 1: pair-up adjacent .lang-es / .lang-en siblings of the same tag.
    #
    # We walk all .lang-es elements; for each, we look at next-sibling Tag (skipping
    # whitespace text nodes). If it matches with .lang-en and same tag name, pair.
    es_tags = list(soup.find_all(class_="lang-es"))
    processed_pairs: set[int] = set()  # ids of tags already replaced

    for es in es_tags:
        if id(es) in processed_pairs:
            continue
        if es.parent is None:  # already detached
            continue

        # find next sibling tag
        sib = es.next_sibling
        while sib is not None and isinstance(sib, NavigableString) and sib.strip() == "":
            sib = sib.next_sibling
        if not isinstance(sib, Tag):
            continue
        if not is_lang_en(sib):
            continue
        if sib.name != es.name:
            continue

        # generate key
        key_prefix = ""
        h2 = section_for(es)
        if h2:
            section = "section." + slugify_id(h2["id"])
        else:
            scope = header_classification(es)
            if scope is None:
                scope = "misc"
            section = scope

        kind = kind_for_pair(es)
        key = keygen.next(section, kind)

        # special handling: <img> with different src per language → attr swap
        if es.name == "img":
            es_src = es.get("src", "")
            en_src = sib.get("src", "")
            # build replacement: keep the EN img but add data-i18n-attr-src
            new_img = sib  # mutate the EN tag in place
            new_img["data-i18n-attr-src"] = key
            strip_lang_class(new_img)
            dictionary[key] = {"es": es_src, "en": en_src}
            es.decompose()
            processed_pairs.add(id(es))
            processed_pairs.add(id(sib))
            continue

        # special handling: <button class="zoom-btn"> with bilingual label and src
        if es.name == "button" and "zoom-btn" in class_of(es):
            label_key = key + ".label"
            src_key = key + ".src"
            es_label = es.decode_contents()
            en_label = sib.decode_contents()
            es_src = es.get("data-zoom-src", "")
            en_src = sib.get("data-zoom-src", "")
            # build single replacement
            new_btn = sib
            new_btn.clear()
            new_btn["data-i18n"] = label_key
            new_btn["data-i18n-attr-data-zoom-src"] = src_key
            new_btn["data-zoom-src"] = en_src  # default to EN
            strip_lang_class(new_btn)
            dictionary[label_key] = {"es": es_label, "en": en_label}
            dictionary[src_key] = {"es": es_src, "en": en_src}
            es.decompose()
            processed_pairs.add(id(es))
            processed_pairs.add(id(sib))
            continue

        # generic case: extract innerHTML, replace ES with single element bearing data-i18n
        es_html = es.decode_contents()
        en_html = sib.decode_contents()

        new_tag = sib  # mutate EN in place; remove ES
        new_tag.clear()
        new_tag["data-i18n"] = key
        strip_lang_class(new_tag)

        dictionary[key] = {"es": es_html, "en": en_html}
        es.decompose()
        processed_pairs.add(id(es))
        processed_pairs.add(id(sib))

    # --- Step 2: drop the dead CSS rules.
    css_blocks = soup.find_all("style")
    rules_to_drop = [
        r"\s*body\.lang-en\s+\.lang-es\s*\{\s*display:\s*none\s*!important;\s*\}\s*",
        r"\s*body\.lang-es\s+\.lang-en\s*\{\s*display:\s*none\s*!important;\s*\}\s*",
    ]
    for css in css_blocks:
        if css.string is None:
            continue
        text = css.string
        for pat in rules_to_drop:
            text = re.sub(pat, "", text, flags=re.MULTILINE)
        css.string.replace_with(text)

    # --- Step 3: serialize and apply post-processing on the raw string.
    result = str(soup)

    # Insert <script src="i18n.js"></script> before the first <script> tag
    # whose body references the localStorage key `tutorial_lang` (i.e. the
    # toggle script), if not already present.
    if 'src="i18n.js"' not in result:
        m = re.search(
            r"(<script>(?:(?!</script>).)*tutorial_lang)",
            result,
            re.DOTALL,
        )
        if m:
            insert_at = m.start()
            result = (
                result[:insert_at]
                + '<script src="i18n.js"></script>\n'
                + result[insert_at:]
            )

    return result, dictionary


# ---------- post-rewrite hooks ----------

def patch_toggle_script(html: str, json_url: str) -> str:
    """Insert i18n.init(...) and i18n.setLang(...) calls into the toggle IIFE.

    Idempotent: re-running on already-patched HTML is a no-op.
    """
    # 1. Inside the setLang function: add i18n.setLang(lang) before the
    # localStorage.setItem call. Tolerant of the line being on its own row or
    # joined to the previous statement.
    if "window.i18n.setLang(lang)" not in html:
        html = re.sub(
            r"(\n\s*try\s*\{\s*localStorage\.setItem\('tutorial_lang',\s*lang\);)",
            r"\n    if (window.i18n) window.i18n.setLang(lang);\1",
            html,
            count=1,
        )

    # 2. After the initial setLang(...) call near the end of the IIFE, insert
    # an i18n.init(...). Matches any single-arg setLang call (e.g. setLang(initialLang),
    # setLang(stored || 'en')) at the start of a line.
    if "window.i18n.init(" not in html:
        pattern = re.compile(
            r"(^\s*setLang\(\s*\w+(?:\s*\|\|\s*['\"]en['\"])?\s*\);)",
            re.MULTILINE,
        )
        # Use a callable to avoid backslash-escaping issues with the json_url.
        def _repl(m: "re.Match[str]") -> str:
            return (
                m.group(1)
                + "\n  if (window.i18n) window.i18n.init('"
                + json_url
                + "', initialLang);"
            )
        html = pattern.sub(_repl, html, count=1)

    return html


# ---------- CLI ----------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", help="Path to the tutorial HTML to extract from")
    parser.add_argument(
        "--json",
        help="Output JSON path. Default: <html_dir>/i18n/<basename>.json",
    )
    args = parser.parse_args(argv)

    html_path = Path(args.html).resolve()
    if not html_path.is_file():
        print(f"error: not a file: {html_path}", file=sys.stderr)
        return 2

    if args.json:
        json_path = Path(args.json).resolve()
    else:
        json_path = html_path.parent / "i18n" / (html_path.stem + ".json")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_url_relative = f"i18n/{html_path.stem}.json"

    raw = html_path.read_text(encoding="utf-8")
    new_html, dictionary = extract(raw)
    new_html = patch_toggle_script(new_html, json_url_relative)

    json_path.write_text(
        json.dumps(dictionary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(new_html, encoding="utf-8")

    print(f"  Wrote JSON: {json_path}  ({len(dictionary)} keys)")
    print(f"  Patched HTML: {html_path}")

    # Embed the JSON inline so the page works via `file://` (no fetch needed).
    from tools.i18n_sync import embed
    n, status = embed(html_path, json_path)
    print(f"  Inline embed: {status} ({n} keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
