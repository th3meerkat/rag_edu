// Shared i18n loader for the tutorials in this directory.
//
// Pattern:
//   1. Each tutorial HTML loads its own translations JSON via `i18n.init(url, lang)`.
//   2. Translatable elements use `data-i18n="key"`; their innerHTML is replaced
//      with `translations[key][lang]`.
//   3. Missing keys render as the literal `{key.path}` so they're visible in
//      the page during authoring without breaking the layout.
//   4. `i18n.setLang(lang)` re-applies the current translations dictionary.
//      The toggle pill's `setLang(...)` must call it after switching `<body>`
//      class / `<html lang>` / `document.title`.

(function () {
  let dict = null;

  async function load(jsonUrl) {
    if (dict) return dict;

    // Prefer the inline <script type="application/json" id="i18n-data"> if
    // present. This makes the page work when opened via `file://` (no fetch),
    // which is the common preview path for these tutorials. The inline data
    // is kept in sync with the external JSON file by `tools/i18n_sync.py`.
    const inline = document.getElementById('i18n-data');
    if (inline && inline.textContent && inline.textContent.trim().length > 0) {
      try {
        dict = JSON.parse(inline.textContent);
        return dict;
      } catch (e) {
        console.error('[i18n] inline data parse error', e);
      }
    }

    try {
      const res = await fetch(jsonUrl);
      if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
      dict = await res.json();
    } catch (e) {
      console.error('[i18n] Failed to load', jsonUrl, e);
      dict = {};
    }
    return dict;
  }

  // Apply translations to:
  //  - `data-i18n="key"`           → element.innerHTML = dict[key][lang]
  //  - `data-i18n-attr-<name>="k"` → element.setAttribute(name, dict[k][lang])
  //    where <name> is any html attribute (`src`, `alt`, `title`, `href`, etc.).
  function apply(lang) {
    if (!dict) return;

    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      const key = el.getAttribute('data-i18n');
      const entry = dict[key];
      if (entry && typeof entry[lang] === 'string') {
        el.innerHTML = entry[lang];
      } else {
        el.textContent = '{' + key + '}';
      }
    });

    // Walk attribute-swap directives. Each `data-i18n-attr-<name>` reads its
    // key and writes the translated value into the named attribute.
    document.querySelectorAll('*').forEach(function (el) {
      for (let i = 0; i < el.attributes.length; i++) {
        const attr = el.attributes[i];
        if (!attr.name.startsWith('data-i18n-attr-')) continue;
        const target = attr.name.substring('data-i18n-attr-'.length);
        const key = attr.value;
        const entry = dict[key];
        if (entry && typeof entry[lang] === 'string') {
          el.setAttribute(target, entry[lang]);
        }
      }
    });
  }

  window.i18n = {
    init: async function (jsonUrl, lang) {
      await load(jsonUrl);
      apply(lang);
    },
    setLang: function (lang) { apply(lang); }
  };
})();
