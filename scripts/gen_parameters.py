"""Generate ``docs/parameters.html`` from the live model parameter classes.

Reads ``BrainParams``, ``SensoryParams``, ``PlantParams`` (canonical Python
source of truth for defaults), pairs each field with the inline source-code
comment, optionally enriches each field from ``docs/parameters_schema.yaml``
(human-curated descriptions, anatomy, references, disorder tags), and emits a
single HTML document.

Schema YAML is OPTIONAL — if missing, every field still appears in the HTML
using its inline comment as the description. Fields present in code but
missing from the schema are flagged with a TODO marker so gaps are visible.

Run:
    python -X utf8 scripts/gen_parameters.py

Output:
    docs/parameters.html
"""

import datetime
import inspect
import os
import re
import sys

# Optional YAML enrichment — degrade gracefully if missing
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from oculomotor.params import BrainParams, SensoryParams, PlantParams
import oculomotor


# ── Parameter introspection ───────────────────────────────────────────────────

# Match a NamedTuple field declaration:  "    name: type = default  # comment"
_FIELD_RE = re.compile(
    r"^    (?P<name>\w+)\s*:\s*"          # field name and colon
    r"[^=]+="                              # type annotation up to '='
    r"\s*(?P<default_src>.*?)"             # default expression (lazy)
    r"(?:\s*#\s*(?P<comment>.*))?$"        # optional inline comment
)
# Continuation comment line:  "                              # more text"
_CONT_RE = re.compile(r"^\s*#\s*(?P<comment>.*)$")


def _extract_fields(cls):
    """Return [(name, default_value, comment_text), ...] preserving source order.

    ``default_value`` is read from a default instance of ``cls`` (the actual
    runtime value the simulator sees); ``comment_text`` is parsed from the
    source so docstrings stay close to the values they describe.
    """
    src_lines = inspect.getsource(cls).splitlines()
    defaults = cls()._asdict()

    fields = []
    current_name = None
    current_comment_parts = []

    def _flush():
        if current_name is not None:
            fields.append((current_name,
                           defaults[current_name],
                           " ".join(p.strip() for p in current_comment_parts).strip()))

    for raw in src_lines:
        m = _FIELD_RE.match(raw)
        if m:
            _flush()
            current_name = m["name"]
            current_comment_parts = [m["comment"] or ""]
        elif current_name is not None:
            cm = _CONT_RE.match(raw)
            if cm:
                current_comment_parts.append(cm["comment"])
            elif raw.strip() == "":
                # Blank line ends the comment block but keeps current field open
                # until the next definition line — comments after a blank line
                # are usually section headers, so stop accumulating.
                _flush()
                current_name = None
        # else: non-comment, non-field line (e.g. closing brace) — ignore
    _flush()
    return fields


def _format_default(value):
    """Render a default value compactly for HTML display."""
    try:
        # JAX/numpy arrays: stringify and trim
        import numpy as _np
        if hasattr(value, 'tolist'):
            value = value.tolist()
            if isinstance(value, list) and len(value) > 6:
                head = ", ".join(f"{v:g}" for v in value[:3])
                return f"[{head}, … ({len(value)} total)]"
            return str(value)
    except ImportError:
        pass
    if isinstance(value, list) and len(value) > 6:
        head = ", ".join(f"{v}" for v in value[:3])
        return f"[{head}, … ({len(value)} total)]"
    if isinstance(value, float):
        # Tidy float display — avoid 0.6000000000000001 noise
        if value == int(value):
            return f"{int(value)}"
        return f"{value:.4g}"
    return str(value)


# ── Schema loading ────────────────────────────────────────────────────────────

def _load_schema(path):
    if not _HAS_YAML or not os.path.isfile(path):
        return {}, "(none)"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, path


def _enrich(class_tag, name, default, fallback_comment, schema):
    """Combine code-side info with optional YAML enrichment.

    Schema keys are ``<class_tag>.<field>`` (e.g. ``brain.K_grav``).
    Presence of ``disorders:`` (even empty list) marks the field as
    LLM-exposed — the YAML-driven Patient builder picks it up.
    """
    key = f"{class_tag}.{name}"
    entry = schema.get(key, {})
    rng = entry.get("range")
    return {
        "name":         name,
        "default":      _format_default(default),
        "units":        entry.get("units", ""),
        "description":  entry.get("description") or fallback_comment or "(no description)",
        "anatomy":      entry.get("anatomy", ""),
        "references":   entry.get("references", []),
        "disorders":    entry.get("disorders", []),
        "range":        rng,
        "length":       entry.get("length"),
        "llm_exposed":  "disorders" in entry,
        "group":        entry.get("group", "ungrouped"),
        "needs_review": bool(entry.get("needs_review", False)),
        "enriched":     key in schema,
        "key":          key,
    }


def _format_range(rng):
    """Format range [lo, hi] for display.  None = unbounded."""
    if not rng:
        return ""
    lo = rng[0] if len(rng) > 0 else None
    hi = rng[1] if len(rng) > 1 else None
    lo_s = "−∞" if lo is None else f"{lo:g}"
    hi_s = "+∞" if hi is None else f"{hi:g}"
    return f"[{lo_s}, {hi_s}]"


# ── HTML emission ─────────────────────────────────────────────────────────────

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; min-height: 100vh; }
nav  { width: 230px; min-height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; flex-shrink: 0;
       overflow-y: auto; max-height: 100vh; }
nav h2  { font-size: 12px; padding: 0 16px 10px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav .nav-group { padding: 12px 16px 4px; color: #888; font-size: 10px;
                 text-transform: uppercase; letter-spacing: 0.06em; }
nav a   { display: block; padding: 7px 16px; color: #ccc; text-decoration: none;
          font-size: 12px; border-left: 3px solid transparent; }
nav a:hover { background: #2a2a4e; color: #fff; border-left-color: #4a90d9; }
main { flex: 1; padding: 32px; max-width: 1400px; }
h1   { font-size: 22px; margin-bottom: 4px; }
.meta { font-size: 12px; color: #888; margin-bottom: 24px; }
.section    { margin-bottom: 40px; }
.section h2 { font-size: 18px; margin-bottom: 6px; border-bottom: 2px solid #ddd;
              padding-bottom: 6px; }
.section .desc { font-size: 13px; color: #555; margin-bottom: 14px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
        border-radius: 6px; overflow: hidden; }
th, td { padding: 7px 9px; text-align: left; vertical-align: top; }
th     { background: #efeff5; font-weight: 600; font-size: 11px;
         color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
tr     { border-bottom: 1px solid #eee; }
tr:last-child { border-bottom: none; }
.param-name  { font-family: 'SF Mono', Consolas, monospace; font-weight: 600;
               color: #1a1a2e; }
.default     { font-family: 'SF Mono', Consolas, monospace;
               background: #f5f5fa; padding: 1px 6px; border-radius: 3px;
               color: #2c5d99; }
.units       { font-size: 11px; color: #888; }
.desc-cell   { color: #333; line-height: 1.45; }
.anatomy-cell { font-size: 11px; color: #666; }
.refs        { font-size: 11px; color: #888; font-style: italic;
               margin-top: 4px; }
.tag         { display: inline-block; padding: 1px 6px; border-radius: 10px;
               font-size: 10px; font-weight: 600; margin-right: 3px;
               margin-bottom: 2px; }
.tag-vn        { background: #fff3cd; color: #856404; }
.tag-cerebellar{ background: #d1ecf1; color: #0c5460; }
.tag-ino       { background: #d4edda; color: #155724; }
.tag-cn        { background: #f8d7da; color: #721c24; }
.tag-other     { background: #e2e3e5; color: #383d41; }
.todo        { display: inline-block; padding: 2px 7px; border-radius: 10px;
               font-size: 10px; font-weight: 600; background: #fff3cd;
               color: #b8870b; margin-left: 8px; }
.review      { display: inline-block; padding: 2px 7px; border-radius: 10px;
               font-size: 10px; font-weight: 600; background: #d1ecf1;
               color: #0c5460; margin-left: 8px; }
.llm-tag     { display: inline-block; padding: 2px 7px; border-radius: 10px;
               font-size: 10px; font-weight: 700; background: #d4edda;
               color: #155724; margin-left: 8px;
               cursor: help; }
.range-cell  { font-family: 'SF Mono', Consolas, monospace; font-size: 11px;
               color: #555; white-space: nowrap; }
.warning     { background: #fff3cd; border-left: 4px solid #f6c90e;
               padding: 10px 14px; border-radius: 4px; font-size: 12px;
               margin-bottom: 18px; color: #856404; }
.subsection  { margin-bottom: 28px; }
.subsection h3 { font-size: 14px; margin-bottom: 6px; color: #333;
                 border-bottom: 1px solid #ddd; padding-bottom: 4px; }
#search      { width: 100%; padding: 8px 12px; border: 1px solid #ccc;
               border-radius: 6px; font-size: 13px; margin-bottom: 18px;
               font-family: inherit; }
#search:focus { outline: none; border-color: #4a90d9;
                box-shadow: 0 0 0 2px rgba(74,144,217,.15); }
.match-count { font-size: 11px; color: #888; margin-bottom: 14px; }
.section.empty, .subsection.empty { display: none; }
"""


def _row_html(field):
    badges = []
    if not field["enriched"]:
        badges.append('<span class="todo">TODO: enrich schema</span>')
    elif field["needs_review"]:
        badges.append('<span class="review">needs review</span>')
    if field["llm_exposed"]:
        badges.append('<span class="llm-tag" title="Exposed in the LLM Patient schema">LLM</span>')
    badge = " ".join(badges)
    refs_html = (
        '<div class="refs">' + " · ".join(field["references"]) + '</div>'
        if field["references"] else ""
    )
    disorders_html = ""
    if field["disorders"]:
        items = []
        for d in field["disorders"]:
            tag_cls = f'tag-{d.get("tag", "other")}'
            label = d.get("name", "?")
            val = d.get("value", "")
            items.append(f'<span class="tag {tag_cls}">{label}: {val}</span>')
        disorders_html = " ".join(items)
    range_html = _format_range(field["range"])
    if field["length"]:
        range_html = (range_html + " " if range_html else "") + f"<small>(len {field['length']})</small>"
    desc = field["description"].rstrip()
    # Preserve multi-paragraph YAML descriptions: blank-line-separated chunks → <p>.
    paragraphs = [p.strip() for p in desc.split("\n\n") if p.strip()]
    desc_html = "".join(f"<p>{p.replace(chr(10), ' ')}</p>" for p in paragraphs) or desc
    return f"""
        <tr>
          <td class="param-name">{field["name"]}{badge}</td>
          <td><span class="default">{field["default"]}</span></td>
          <td class="units">{field["units"]}</td>
          <td class="range-cell">{range_html}</td>
          <td class="desc-cell">{desc_html}{refs_html}</td>
          <td class="anatomy-cell">{field["anatomy"]}</td>
          <td>{disorders_html}</td>
        </tr>"""


def _group_label(group_key):
    """Pretty-print a group key for HTML headers."""
    return group_key.replace("_", " ").title() if group_key else "Ungrouped"


def _subsection_html(class_label, group_key, fields):
    """Render one (class, group) subsection as a table."""
    rows = "\n".join(_row_html(f) for f in fields)
    sid = f"{class_label.lower()}-{group_key.lower().replace(' ', '-')}"
    return f"""
    <div class="subsection" id="{sid}">
      <h3>{_group_label(group_key)} <small style="font-size:11px;font-weight:400;color:#888">({len(fields)} field{'s' if len(fields) != 1 else ''})</small></h3>
      <table class="param-table">
        <thead><tr>
          <th style="width:160px">Parameter</th>
          <th style="width:90px">Default</th>
          <th style="width:60px">Units</th>
          <th style="width:110px">Valid range</th>
          <th>Description</th>
          <th style="width:170px">Anatomical locus</th>
          <th style="width:200px">Disorders / notable values</th>
        </tr></thead>
        <tbody>{rows}
        </tbody>
      </table>
    </div>"""


def _section_html(class_label, fields, intro=""):
    """Render a class section, internally grouped by `group:` key from schema."""
    sid = class_label.lower().replace(" ", "-")
    # Stable order: preserve first-seen group order from the field list.
    groups = []
    grouped = {}
    for f in fields:
        g = f.get("group", "ungrouped") or "ungrouped"
        if g not in grouped:
            groups.append(g)
            grouped[g] = []
        grouped[g].append(f)
    subsections = "\n".join(_subsection_html(class_label, g, grouped[g]) for g in groups)
    return f"""
  <section class="section" id="{sid}">
    <h2>{class_label} <small style="font-size:12px;font-weight:400;color:#888">({len(fields)} fields)</small></h2>
    <p class="desc">{intro}</p>
    {subsections}
  </section>"""


def _build_nav_groups(class_label, fields):
    """Build nav links for each group within a class."""
    seen = []
    grouped = {}
    for f in fields:
        g = f.get("group", "ungrouped") or "ungrouped"
        if g not in grouped:
            seen.append(g)
            grouped[g] = 0
        grouped[g] += 1
    sid_class = class_label.lower()
    parts = [f'    <a href="#{sid_class}">{class_label}</a>']
    parts += [f'    <a href="#{sid_class}-{g.lower().replace(" ", "-")}" '
              f'style="padding-left:32px;font-size:11px;color:#aaa">'
              f'{_group_label(g)} <span style="color:#666">({grouped[g]})</span></a>'
              for g in seen]
    return "\n".join(parts)


def _build_html(brain_fields, sensory_fields, plant_fields, schema_path,
                missing_keys, stale_keys):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ver = oculomotor.__version__

    nav = "\n".join([
        _build_nav_groups("BrainParams",   brain_fields),
        '    <div style="height:8px"></div>',
        _build_nav_groups("SensoryParams", sensory_fields),
        '    <div style="height:8px"></div>',
        _build_nav_groups("PlantParams",   plant_fields),
    ])

    warn = ""
    if missing_keys:
        warn += (
            f'<div class="warning"><b>{len(missing_keys)} fields lack schema entries</b> '
            'and are flagged TODO below. Add to <code>docs/parameters_schema.yaml</code> '
            'to provide rich descriptions, anatomy, and disorder tags.</div>'
        )
    if stale_keys:
        warn += (
            f'<div class="warning"><b>{len(stale_keys)} stale schema entries</b> '
            f'do not match any current field: <code>{", ".join(stale_keys[:6])}'
            f'{"…" if len(stale_keys) > 6 else ""}</code>. '
            'Remove from YAML or rename to match the current code.</div>'
        )

    search_js = """
<script>
(function(){
  var search = document.getElementById('search');
  var counter = document.getElementById('match-count');
  var allRows = Array.prototype.slice.call(document.querySelectorAll('tbody tr'));
  var allSubs = Array.prototype.slice.call(document.querySelectorAll('.subsection'));
  var allSecs = Array.prototype.slice.call(document.querySelectorAll('.section'));

  function applyFilter(){
    var q = search.value.toLowerCase().trim();
    var visible = 0;
    allRows.forEach(function(row){
      if (!q) { row.style.display = ''; visible++; return; }
      var text = row.textContent.toLowerCase();
      var match = text.indexOf(q) !== -1;
      row.style.display = match ? '' : 'none';
      if (match) visible++;
    });
    // Hide subsections / sections whose all rows are hidden
    allSubs.forEach(function(sub){
      var any = sub.querySelectorAll('tbody tr:not([style*="display: none"])').length;
      sub.classList.toggle('empty', any === 0 && !!q);
    });
    allSecs.forEach(function(sec){
      var any = sec.querySelectorAll('.subsection:not(.empty)').length;
      sec.classList.toggle('empty', any === 0 && !!q);
    });
    counter.textContent = q ? (visible + ' matching field' + (visible === 1 ? '' : 's')) : '';
  }
  search.addEventListener('input', applyFilter);
})();
</script>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OMSim — Parameter Reference</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
  <nav>
    <h2>Sections</h2>
{nav}
    <div style="border-top:1px solid #2a2a4e;margin:18px 12px 8px;"></div>
    <h2>Other pages</h2>
    <a href="benchmarks/">Benchmarks</a>
    <a href="clinical_benchmarks/">Clinical benchmarks</a>
    <a href="index.html">LLM simulator</a>
  </nav>
  <main>
    <h1>OMSim — Parameter Reference</h1>
    <p class="meta">
      Generated: <strong>{ts}</strong> &nbsp;|&nbsp;
      Version: <strong>{ver}</strong> &nbsp;|&nbsp;
      Schema: <strong>{schema_path}</strong>
    </p>
    <input id="search" type="search" placeholder="Search parameters by name, description, anatomy, disorder…" autofocus>
    <div id="match-count" class="match-count"></div>
    {warn}
    {_section_html("BrainParams", brain_fields,
                   "Central neural model parameters — VS, NI, SG, gravity, "
                   "vergence, accommodation, T-VOR, OCR, CN/MLF gains. "
                   "Read by every brain submodule's <code>step()</code>.")}
    {_section_html("SensoryParams", sensory_fields,
                   "Peripheral sensor parameters — canal/otolith dynamics, "
                   "visual delay, retinal noise, IPD.")}
    {_section_html("PlantParams", plant_fields,
                   "Extraocular plant — mechanical orbit dynamics.")}
  </main>
{search_js}
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_yml = os.path.join(repo_root, "docs", "parameters_schema.yaml")
    out_path   = os.path.join(repo_root, "docs", "parameters.html")

    schema, schema_path = _load_schema(schema_yml)
    if not _HAS_YAML:
        print("Note: PyYAML not installed — proceeding without schema enrichment.")

    # Introspect
    brain_raw   = _extract_fields(BrainParams)
    sensory_raw = _extract_fields(SensoryParams)
    plant_raw   = _extract_fields(PlantParams)

    # Enrich
    brain_fields   = [_enrich("brain",   n, d, c, schema) for n, d, c in brain_raw]
    sensory_fields = [_enrich("sensory", n, d, c, schema) for n, d, c in sensory_raw]
    plant_fields   = [_enrich("plant",   n, d, c, schema) for n, d, c in plant_raw]

    code_keys = (
        {f"brain.{n}"   for n, _, _ in brain_raw} |
        {f"sensory.{n}" for n, _, _ in sensory_raw} |
        {f"plant.{n}"   for n, _, _ in plant_raw}
    )
    schema_keys = set(schema.keys()) if isinstance(schema, dict) else set()
    missing_keys = sorted(code_keys - schema_keys)
    stale_keys   = sorted(schema_keys - code_keys)

    html = _build_html(brain_fields, sensory_fields, plant_fields,
                       schema_path, missing_keys, stale_keys)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote: {out_path}")
    print(f"  BrainParams:   {len(brain_fields)} fields")
    print(f"  SensoryParams: {len(sensory_fields)} fields")
    print(f"  PlantParams:   {len(plant_fields)} fields")
    if missing_keys:
        print(f"  TODO (missing schema): {len(missing_keys)}")
    if stale_keys:
        print(f"  Stale schema entries:  {len(stale_keys)}")


if __name__ == "__main__":
    main()
