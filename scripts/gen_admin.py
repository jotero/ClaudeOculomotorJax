"""Generate outputs/admin.html — standalone, works from file://.

Data is embedded as a JS constant so no HTTP server is needed.
Images reference server_figures/<run_id>.png via relative paths, which
browsers allow from file://.

Usage
-----
    python scripts/gen_admin.py            # regenerate from current CSV
"""

import csv
import json
import os
from pathlib import Path

_OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'
_LOG_FILE    = _OUTPUTS_DIR / 'simulation_log.csv'
_OUT_FILE    = _OUTPUTS_DIR / 'admin.html'

_LOG_COLUMNS = [
    'timestamp', 'run_id', 'version', 'prompt', 'mode', 'title',
    'figure_file', 'looks_correct', 'feedback',
]


def _fig_rel(figure_file: str) -> str:
    """Convert absolute figure path to relative path from outputs/."""
    if not figure_file:
        return ''
    base = figure_file.replace('\\', '/').split('/')[-1]
    return f'server_figures/{base}'


def load_rows() -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    with open(_LOG_FILE, newline='', encoding='utf-8') as f:
        rows = [dict(r) for r in csv.DictReader(f) if r.get('run_id')]
    for r in rows:
        r['figure_rel'] = _fig_rel(r.get('figure_file', ''))
    return rows


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OculomotorSim — Simulation Log</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0f1117; color: #e0e0e0; height: 100vh; display: flex;
  flex-direction: column; overflow: hidden;
}}

/* ── Top bar ── */
header {{
  background: #151820; border-bottom: 1px solid #2a2d3a;
  padding: 0 20px; display: flex; align-items: center; gap: 20px;
  flex-shrink: 0; height: 44px;
}}
header .title {{ font-size: 0.82rem; font-weight: 600; color: #aaa;
                letter-spacing: 0.06em; text-transform: uppercase; }}
header .count {{ font-size: 0.75rem; color: #555; margin-left: auto; }}
header input {{
  background: #0f1117; border: 1px solid #2a2d3a; border-radius: 6px;
  color: #ccc; font-size: 0.8rem; padding: 5px 10px; width: 260px; outline: none;
}}
header input:focus {{ border-color: #4a6cf7; }}
header input::placeholder {{ color: #444; }}

/* ── Main layout ── */
.layout {{ display: flex; flex: 1; overflow: hidden; }}

/* ── Table panel ── */
.table-panel {{
  width: 52%; border-right: 1px solid #1e2130;
  overflow-y: auto; flex-shrink: 0;
}}
table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; }}
thead th {{
  background: #0d0f18; color: #666; font-size: 0.68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  padding: 8px 10px; text-align: left; position: sticky; top: 0; z-index: 1;
  border-bottom: 1px solid #1e2130;
}}
tbody tr {{
  border-bottom: 1px solid #191c27; cursor: pointer;
  transition: background 0.1s;
}}
tbody tr:hover {{ background: #151820; }}
tbody tr.active {{ background: #1a1f35; border-left: 3px solid #4a6cf7; }}
tbody tr.active td:first-child {{ padding-left: 7px; }}
td {{ padding: 7px 10px; vertical-align: top; color: #bbb; }}
td.ts    {{ color: #555; white-space: nowrap; font-size: 0.7rem; }}
td.prompt-cell {{
  color: #ddd; max-width: 260px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}}
td.title-cell {{ color: #888; font-size: 0.7rem; max-width: 180px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mode-badge {{
  display: inline-block; font-size: 0.62rem; font-weight: 700; padding: 1px 6px;
  border-radius: 8px; white-space: nowrap; text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.mode-single     {{ background: #1e3a6e; color: #7ab3f7; }}
.mode-comparison {{ background: #1a3a28; color: #6fcf97; }}
.ok-dot {{
  display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-top: 4px;
}}
.ok-true  {{ background: #2ecc71; }}
.ok-false {{ background: #e74c3c; }}
.ok-null  {{ background: #333; }}
tr.hidden {{ display: none; }}

/* ── Figure panel ── */
.figure-panel {{
  flex: 1; overflow-y: auto; padding: 20px;
  display: flex; flex-direction: column; gap: 16px;
}}
.no-selection {{
  flex: 1; display: flex; align-items: center; justify-content: center;
  color: #333; font-size: 0.85rem; text-align: center; line-height: 1.8;
}}
.figure-panel img {{
  width: 100%; border-radius: 6px; border: 1px solid #2a2d3a;
  display: block;
}}
.detail-grid {{
  display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px;
  font-size: 0.75rem;
}}
.detail-grid .lbl {{ color: #555; text-transform: uppercase;
                    font-size: 0.65rem; letter-spacing: 0.05em; padding-top: 2px; }}
.detail-grid .val {{ color: #ccc; word-break: break-word; }}
.detail-grid .val.prompt-val {{ color: #eee; font-size: 0.8rem; line-height: 1.5; }}
.feedback-val {{ color: #888; font-style: italic; }}
</style>
</head>
<body>

<header>
  <span class="title">Simulation Log</span>
  <input id="search" placeholder="Filter by prompt…" oninput="filter(this.value)">
  <span class="count" id="count"></span>
</header>

<div class="layout">
  <div class="table-panel">
    <table id="log">
      <thead>
        <tr>
          <th style="width:90px">Date</th>
          <th style="width:76px">Mode</th>
          <th>Prompt</th>
          <th style="width:130px">Title</th>
          <th style="width:22px"></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <div class="figure-panel" id="panel">
    <div class="no-selection">← select a row to preview the figure</div>
  </div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const ALL_ROWS = {rows_json};

// ── Format timestamp ─────────────────────────────────────────────────────────
function fmtTs(ts) {{
  if (!ts) return '—';
  const m = ts.match(/(\\d{{4}})-(\\d{{2}})-(\\d{{2}})T(\\d{{2}}):(\\d{{2}})/);
  if (!m) return ts.slice(0, 10);
  return `${{m[1]}}-${{m[2]}}-${{m[3]}} ${{m[4]}}:${{m[5]}}`;
}}

// ── State ────────────────────────────────────────────────────────────────────
let rows = [];
let activeIdx = null;

// ── Render table ─────────────────────────────────────────────────────────────
function renderTable(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  rows = [...data].reverse();  // newest first
  rows.forEach((r, i) => {{
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.innerHTML = `
      <td class="ts">${{fmtTs(r.timestamp)}}</td>
      <td><span class="mode-badge mode-${{r.mode}}">${{r.mode || '—'}}</span></td>
      <td class="prompt-cell" title="${{esc(r.prompt)}}">${{esc(r.prompt)}}</td>
      <td class="title-cell" title="${{esc(r.title)}}">${{esc(r.title)}}</td>
      <td><span class="ok-dot ok-${{r.looks_correct === 'True' ? 'true' : r.looks_correct === 'False' ? 'false' : 'null'}}"></span></td>
    `;
    tr.addEventListener('click', () => selectRow(i, tr));
    tbody.appendChild(tr);
  }});
  updateCount();
}}

function esc(s) {{
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ── Select row → show figure ─────────────────────────────────────────────────
function selectRow(idx, tr) {{
  document.querySelectorAll('#tbody tr').forEach(r => r.classList.remove('active'));
  tr.classList.add('active');
  activeIdx = idx;
  const r = rows[idx];
  const panel = document.getElementById('panel');
  panel.innerHTML = '';

  if (r.figure_rel) {{
    const img = document.createElement('img');
    img.src = r.figure_rel;
    img.alt = r.title || 'simulation figure';
    panel.appendChild(img);
  }}

  const details = document.createElement('div');
  details.className = 'detail-grid';
  const ok = r.looks_correct === 'True'  ? '✓ looks correct'
           : r.looks_correct === 'False' ? '✗ incorrect' : '—';
  details.innerHTML = `
    <span class="lbl">Prompt</span>
    <span class="val prompt-val">${{esc(r.prompt)}}</span>
    <span class="lbl">Title</span>
    <span class="val">${{esc(r.title)}}</span>
    <span class="lbl">Mode</span>
    <span class="val"><span class="mode-badge mode-${{r.mode}}">${{r.mode||'—'}}</span></span>
    <span class="lbl">Date</span>
    <span class="val">${{fmtTs(r.timestamp)}}</span>
    <span class="lbl">Version</span>
    <span class="val">${{esc(r.version)}}</span>
    <span class="lbl">Run ID</span>
    <span class="val" style="font-family:monospace;font-size:0.68rem;color:#555">${{esc(r.run_id)}}</span>
    <span class="lbl">Correct</span>
    <span class="val">${{ok}}</span>
    ${{r.feedback ? `<span class="lbl">Feedback</span><span class="val feedback-val">${{esc(r.feedback)}}</span>` : ''}}
  `;
  panel.appendChild(details);
}}

// ── Filter ────────────────────────────────────────────────────────────────────
function filter(q) {{
  q = q.toLowerCase();
  let visible = 0;
  document.querySelectorAll('#tbody tr').forEach((tr, i) => {{
    const r = rows[i];
    const match = !q
      || (r.prompt||'').toLowerCase().includes(q)
      || (r.title||'').toLowerCase().includes(q)
      || (r.mode||'').toLowerCase().includes(q);
    tr.classList.toggle('hidden', !match);
    if (match) visible++;
  }});
  document.getElementById('count').textContent = `${{visible}} runs`;
}}

function updateCount() {{
  document.getElementById('count').textContent = `${{rows.length}} runs`;
}}

// ── Init ──────────────────────────────────────────────────────────────────────
renderTable(ALL_ROWS);
</script>
</body>
</html>
"""


def generate(rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = load_rows()
    rows_json = json.dumps(rows, ensure_ascii=False, indent=None)
    html = _HTML_TEMPLATE.format(rows_json=rows_json)
    _OUT_FILE.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    rows = load_rows()
    generate(rows)
    print(f"Written {len(rows)} rows → {_OUT_FILE}")
