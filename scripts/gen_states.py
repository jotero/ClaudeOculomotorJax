"""Generate docs/states.html — anatomical state / activation directory.

Layered on docs/states_schema.yaml.  Sister page to parameters.html: groups
each model state slice + key derived signal by anatomical subsystem,
showing shape, units, anatomy, description, and references.

The same YAML feeds the future LLM signal-registry that drives custom plot
panels — entries with `llm_plottable: true` are the candidates exposed.

Usage:
    python -X utf8 scripts/gen_states.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml


_REPO = Path(__file__).resolve().parent.parent
_YAML = _REPO / 'docs' / 'states_schema.yaml'
_HTML = _REPO / 'docs' / 'states.html'


# ── HTML / CSS ────────────────────────────────────────────────────────────────

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; min-height: 100vh; }
nav  { width: 240px; min-height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; flex-shrink: 0;
       overflow-y: auto; max-height: 100vh; }
nav h2  { font-size: 12px; padding: 0 16px 10px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav .nav-class { padding: 12px 16px 4px; color: #888; font-size: 10px;
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
.subsection  { margin-bottom: 28px; }
.subsection h3 { font-size: 14px; margin-bottom: 6px; color: #333;
                 border-bottom: 1px solid #ddd; padding-bottom: 4px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        font-size: 12px; box-shadow: 0 1px 3px rgba(0,0,0,.06);
        border-radius: 6px; overflow: hidden; }
th, td { padding: 7px 9px; text-align: left; vertical-align: top; }
th     { background: #efeff5; font-weight: 600; font-size: 11px;
         color: #555; text-transform: uppercase; letter-spacing: 0.04em; }
tr     { border-bottom: 1px solid #eee; }
tr:last-child { border-bottom: none; }
.signal-name { font-family: 'SF Mono', Consolas, monospace; font-weight: 600;
               color: #1a1a2e; }
.shape-cell  { font-family: 'SF Mono', Consolas, monospace; color: #2c5d99;
               font-size: 11px; white-space: nowrap; }
.axes-cell   { font-family: 'SF Mono', Consolas, monospace; font-size: 10.5px;
               color: #555; }
.units       { font-size: 11px; color: #888; white-space: nowrap; }
.encodes-cell { color: #1d3a8a; font-weight: 500; line-height: 1.4; }
.frame-cell   { font-size: 11px; color: #5a7090; font-style: italic;
                margin-top: 3px; }
.repr-tag     { display: inline-block; padding: 1px 6px; border-radius: 3px;
                font-size: 10px; font-weight: 600; margin-bottom: 4px; }
.repr-firing-rate          { background: #d4edda; color: #155724; }
.repr-firing-rate-bias     { background: #cfe2ef; color: #0c5460; }
.repr-rectified            { background: #d6f1d6; color: #1d3a1d; }
.repr-position             { background: #e8e0f5; color: #4a2671; }
.repr-bookkeeping          { background: #efe8d8; color: #6b5520; }
.activation-cell { font-family: 'SF Mono', Consolas, monospace;
                   font-size: 10.5px; color: #6a3da6; line-height: 1.4;
                   margin-top: 4px; }
.baseline-cell   { font-size: 10.5px; color: #888; margin-top: 2px; }
.desc-cell   { color: #333; line-height: 1.45; }
.anatomy-cell { font-size: 11px; color: #666; line-height: 1.45; }
.anatomy-cell .ax-tag {
    display: inline-block; padding: 1px 5px; margin-right: 3px;
    border-radius: 3px; font-family: 'SF Mono', Consolas, monospace;
    font-size: 10px; font-weight: 600;
    background: #e8e9ef; color: #333;
}
.computation { font-family: 'SF Mono', Consolas, monospace; font-size: 11px;
               color: #6a3da6; background: #f6f1fb; padding: 1px 6px;
               border-radius: 3px; display: inline-block; }
.population  { font-size: 10.5px; font-style: italic; color: #5a7090;
               margin-top: 3px; }
.refs        { font-size: 11px; color: #888; font-style: italic;
               margin-top: 4px; }
.llm-tag     { display: inline-block; padding: 2px 7px; border-radius: 10px;
               font-size: 10px; font-weight: 700; background: #d4edda;
               color: #155724; margin-left: 8px;
               cursor: help; }
.derived-tag { display: inline-block; padding: 2px 7px; border-radius: 10px;
               font-size: 10px; font-weight: 600; background: #fae6f5;
               color: #5c1c4a; margin-left: 8px;
               cursor: help; }
.warning     { background: #fff3cd; border-left: 4px solid #f6c90e;
               padding: 10px 14px; border-radius: 4px; font-size: 12px;
               margin-bottom: 18px; color: #856404; }
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_schema():
    with open(_YAML, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _format_shape(shape):
    if not shape:
        return ''
    return '(' + ', '.join(str(s) for s in shape) + ')'


def _format_axes(axes):
    if not axes:
        return ''
    if len(axes) > 6:
        return ', '.join(axes[:5]) + f', … (+{len(axes) - 5})'
    return ', '.join(axes)


def _refs_html(refs):
    if not refs:
        return ''
    return '<div class="refs">' + ' · '.join(refs) + '</div>'


def _anatomy_html(anatomy):
    """Render anatomy as either a string or a per-axis breakdown.

    Per-axis form (list of {axes, region}) shows each axis-group → region
    mapping on its own line, using <strong> for the axis labels.  Plain
    string form just renders as-is.
    """
    if not anatomy:
        return ''
    if isinstance(anatomy, list):
        rows = []
        for entry in anatomy:
            axes_label = ' · '.join(entry.get('axes', []))
            region     = entry.get('region', '')
            rows.append(
                f'<div><span class="ax-tag">{axes_label}</span> {region}</div>'
            )
        return '\n'.join(rows)
    # Plain string fallback
    return str(anatomy)


def _row_html(field):
    name      = field['name']
    klass     = field['class']
    is_derived = klass == 'derived'

    badges = []
    if field.get('llm_plottable'):
        badges.append('<span class="llm-tag" title="Exposable to LLM custom-panel signal picker">LLM</span>')
    if is_derived:
        badges.append('<span class="derived-tag" title="Computed per ODE step (not stored in SimState)">derived</span>')
    badge_html = ' '.join(badges)

    # source: state_slice OR computation
    if field.get('computation'):
        source_html = f'<span class="computation">{field["computation"]}</span>'
    elif field.get('state_slice'):
        source_html = f'<span class="computation">{field["state_slice"]}</span>'
    else:
        source_html = ''

    population_html = (
        f'<div class="population">{field["population"]}</div>'
        if field.get('population') else ''
    )

    desc = (field.get('description') or '').strip()
    paragraphs = [p.strip() for p in desc.split('\n\n') if p.strip()]
    desc_html = ''.join(f'<p>{p.replace(chr(10), " ")}</p>' for p in paragraphs) or desc

    encodes = field.get('encodes', '')
    frame   = field.get('frame', '')
    encodes_block = (
        f'<div class="encodes-cell">{encodes}</div>' if encodes else ''
    ) + (
        f'<div class="frame-cell">frame: {frame}</div>' if frame else ''
    )

    # Representation / activation / baseline block
    repr_str = field.get('representation', '')
    repr_class_map = {
        'linear-firing-rate':       'repr-firing-rate',
        'firing-rate-around-bias':  'repr-firing-rate-bias',
        'rectified-firing-rate':    'repr-rectified',
        'position-equivalent':      'repr-position',
        'scalar-bookkeeping':       'repr-bookkeeping',
    }
    repr_html = (
        f'<span class="repr-tag {repr_class_map.get(repr_str, "")}">{repr_str}</span>'
        if repr_str else ''
    )
    act_html = (
        f'<div class="activation-cell">activation: {field["activation"]}</div>'
        if field.get('activation') else ''
    )
    base_html = (
        f'<div class="baseline-cell">baseline: {field["baseline"]}</div>'
        if field.get('baseline') else ''
    )
    repr_block = repr_html + act_html + base_html

    return f"""
        <tr>
          <td class="signal-name">{name}{(' ' + badge_html) if badge_html else ''}</td>
          <td class="shape-cell">{_format_shape(field.get('shape'))}</td>
          <td class="axes-cell">{_format_axes(field.get('axes', []))}</td>
          <td class="units">{field.get('units', '')}</td>
          <td>{encodes_block}</td>
          <td>{repr_block}</td>
          <td class="desc-cell">{desc_html}{population_html}{_refs_html(field.get('references'))}</td>
          <td class="anatomy-cell">{_anatomy_html(field.get('anatomy'))}</td>
          <td>{source_html}</td>
        </tr>"""


def _group_label(group_key):
    return group_key.replace('_', ' ').title() if group_key else 'Ungrouped'


def _subsection_html(class_label, group_key, fields):
    rows = '\n'.join(_row_html(f) for f in fields)
    sid = f'{class_label.lower()}-{group_key.lower().replace(" ", "-")}'
    n = len(fields)
    return f"""
    <div class="subsection" id="{sid}">
      <h3>{_group_label(group_key)} <small style="font-size:11px;font-weight:400;color:#888">({n} signal{'s' if n != 1 else ''})</small></h3>
      <table>
        <thead><tr>
          <th style="width:140px">Signal</th>
          <th style="width:50px">Shape</th>
          <th style="width:140px">Axes</th>
          <th style="width:65px">Units</th>
          <th style="width:200px">Encodes / frame</th>
          <th style="width:220px">Representation / activation</th>
          <th>Description</th>
          <th style="width:170px">Anatomical locus</th>
          <th style="width:170px">Source</th>
        </tr></thead>
        <tbody>{rows}
        </tbody>
      </table>
    </div>"""


def _section_html(class_label, fields, intro=''):
    if not fields:
        return ''
    # Group by 'group' field
    by_group = {}
    for f in fields:
        by_group.setdefault(f.get('group') or 'ungrouped', []).append(f)
    subsections = '\n'.join(
        _subsection_html(class_label, g, by_group[g]) for g in sorted(by_group)
    )
    sid = f'section-{class_label.lower()}'
    return f"""
    <section class="section" id="{sid}">
      <h2>{class_label}</h2>
      <div class="desc">{intro}</div>
      {subsections}
    </section>"""


def _build_nav(class_groups):
    """Sidebar navigation tree (class → group)."""
    items = ['<h2>States &amp; activations</h2>']
    items.append('<a href="#section-Sensory">Sensory</a>')
    items.append('<a href="#section-Brain">Brain</a>')
    items.append('<a href="#section-Plant">Plant</a>')
    items.append('<a href="#section-Derived">Derived</a>')
    items.append('<div class="nav-class">By group</div>')
    for cls, groups in class_groups:
        for g in sorted(groups):
            sid = f'{cls.lower()}-{g.lower().replace(" ", "-")}'
            items.append(f'<a href="#{sid}" style="padding-left:32px;font-size:11px;color:#aaa">{_group_label(g)}</a>')
    return '\n'.join(items)


def _build_html(by_class):
    # Header
    head = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>States &amp; activations — Oculomotor model</title>
<style>{_HTML_CSS}</style>
</head><body>"""

    # Build class-group index for the nav
    class_groups = []
    for cls in ('Sensory', 'Brain', 'Plant', 'Derived'):
        fields = by_class.get(cls, [])
        groups = {f.get('group') or 'ungrouped' for f in fields}
        class_groups.append((cls, sorted(groups)))

    nav = f"<nav>{_build_nav(class_groups)}</nav>"

    intros = {
        'Sensory': 'Receptor afferents and visual delay cascades — peripheral inputs to the brain.',
        'Brain':   'Central oculomotor circuits: VS / NI / saccade generator / pursuit / vergence / accommodation / FCP.',
        'Plant':   'Eyeball orientation states (rotation vectors) + lens / accommodation plant.',
        'Derived': 'Signals computed per ODE step but not stored in SimState — exposed for plotting and downstream pathways.',
    }

    body_sections = ''.join(
        _section_html(cls, by_class.get(cls, []), intros.get(cls, ''))
        for cls in ('Sensory', 'Brain', 'Plant', 'Derived')
    )

    n_total = sum(len(v) for v in by_class.values())
    n_llm = sum(1 for v in by_class.values() for f in v if f.get('llm_plottable'))

    main = f"""
<main>
  <h1>Model states &amp; activations</h1>
  <div class="meta">
    {n_total} entries · {n_llm} LLM-plottable ·
    Source: <code>docs/states_schema.yaml</code> ·
    Companion to <a href="parameters.html">parameters.html</a>.
  </div>
  <p style="font-size:13px;color:#444;margin-bottom:18px;">
    Anatomical inventory of every state slice the simulator carries plus the
    key derived signals computed per ODE step.  The same YAML drives the
    forthcoming LLM signal registry for custom panel composition — entries
    marked <span class="llm-tag" style="margin-left:0">LLM</span> are
    candidates for the panel picker.
  </p>
  {body_sections}
</main>"""

    return head + nav + main + '</body></html>'


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    schema = _load_schema()

    # Group entries by top-level class (sensory / brain / plant / derived)
    by_class = {'Sensory': [], 'Brain': [], 'Plant': [], 'Derived': []}
    class_label_map = {'sensory': 'Sensory', 'brain': 'Brain',
                       'plant': 'Plant', 'derived': 'Derived'}

    for key, entry in schema.items():
        if not isinstance(entry, dict):
            continue
        try:
            cls, name = key.split('.', 1)
        except ValueError:
            continue
        label = class_label_map.get(cls)
        if not label:
            continue
        field = dict(entry)
        field['name'] = name
        field['class'] = cls
        by_class[label].append(field)

    html = _build_html(by_class)
    _HTML.parent.mkdir(parents=True, exist_ok=True)
    _HTML.write_text(html, encoding='utf-8')

    print(f'Wrote: {_HTML}')
    for cls, fields in by_class.items():
        n_llm = sum(1 for f in fields if f.get('llm_plottable'))
        print(f'  {cls:8} {len(fields):3d} entries  ({n_llm} LLM-plottable)')


if __name__ == '__main__':
    main()
