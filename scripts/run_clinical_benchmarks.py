"""Run all clinical benchmark scripts and generate docs/clinical_benchmarks/index.html.

Usage:
    python -X utf8 scripts/run_clinical_benchmarks.py           # run all sections
    python -X utf8 scripts/run_clinical_benchmarks.py --html-only # rebuild HTML only
    python -X utf8 scripts/run_clinical_benchmarks.py --show     # show figures

Individual sections can be re-run by running their own scripts:
    python -X utf8 scripts/bench_clinical_vestibular.py
    python -X utf8 scripts/bench_clinical_ni_vs.py
    python -X utf8 scripts/bench_clinical_cn_palsies.py
    python -X utf8 scripts/bench_clinical_saccades.py
    python -X utf8 scripts/bench_clinical_vergence.py
Then re-run run_clinical_benchmarks.py --html-only to rebuild the report.
"""

import sys
import os
import datetime
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_clinical_utils as utils
import bench_utils
import oculomotor

SHOW      = '--show' in sys.argv
HTML_ONLY = '--html-only' in sys.argv

MODULES = [
    'bench_clinical_vestibular',
    'bench_clinical_ni_vs',
    'bench_clinical_cn_palsies',
    'bench_clinical_saccades',
    'bench_clinical_vergence',
]


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; }
nav  { width: 210px; min-height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; flex-shrink: 0; }
nav h2  { font-size: 13px; padding: 0 16px 12px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav a   { display: block; padding: 8px 16px; color: #ccc; text-decoration: none;
          font-size: 13px; border-left: 3px solid transparent; }
nav a:hover { background: #2a2a4e; color: #fff; border-left-color: #e08214; }
main { flex: 1; padding: 32px; max-width: 1400px; }
h1   { font-size: 22px; margin-bottom: 4px; }
.meta { font-size: 12px; color: #888; margin-bottom: 32px; }
.section    { margin-bottom: 48px; }
.section h2 { font-size: 18px; margin-bottom: 6px; border-bottom: 2px solid #e08214;
              padding-bottom: 6px; }
.section > p { font-size: 13px; color: #555; margin-bottom: 16px; }
.fig-grid   { display: grid; grid-template-columns: repeat(auto-fill, minmax(580px, 1fr));
              gap: 20px; }
.fig-card   { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
              padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
.fig-card a img { width: 100%; border-radius: 4px; display: block;
                  border: 1px solid #eee; cursor: zoom-in; }
.fig-card h3 { font-size: 14px; margin: 12px 0 6px; }
.fig-card .desc { font-size: 12px; color: #555; margin-bottom: 8px; }
.expected   { background: #fff8f0; border-left: 3px solid #e08214;
              padding: 8px 10px; font-size: 12px; border-radius: 0 4px 4px 0;
              margin-bottom: 8px; }
.expected strong { display: block; font-size: 11px; color: #888;
                   text-transform: uppercase; margin-bottom: 2px; }
.citation   { font-size: 11px; color: #888; font-style: italic; }
.badge      { display: inline-block; padding: 2px 8px; border-radius: 12px;
              font-size: 10px; font-weight: 600; letter-spacing: 0.04em;
              text-transform: uppercase; margin-top: 8px; }
.badge.behavior { background: #d4edda; color: #155724; }
.badge.cascade  { background: #cce5ff; color: #004085; }
.banner     { background: #fff3cd; border: 1px solid #ffc107;
              padding: 10px 16px; border-radius: 6px; margin-bottom: 24px;
              font-size: 13px; color: #856404; }
.fig-pair       { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
                  align-items: start; }
.fig-pair .lbl  { font-size: 10px; color: #888; text-transform: uppercase;
                  letter-spacing: 0.04em; margin: 0 0 2px 2px; }
.diff-badge     { display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
                  text-transform: uppercase; margin-bottom: 6px; margin-right: 6px; }
.diff-match     { background: #d4edda; color: #155724; }
.diff-changed   { background: #f8d7da; color: #721c24; }
.diff-shape     { background: #fff3cd; color: #856404; }
.diff-noref     { background: #e2e3e5; color: #383d41; }
"""

_HTML_LIGHTBOX = """
<div id="lb" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,.85);z-index:1000;cursor:zoom-out;align-items:center;
     justify-content:center;">
  <img id="lb-img" style="max-width:95vw;max-height:95vh;border-radius:4px;">
</div>
<script>
(function(){
  var lb=document.getElementById('lb'),
      li=document.getElementById('lb-img');
  document.querySelectorAll('.fig-card a').forEach(function(a){
    a.addEventListener('click',function(e){
      e.preventDefault();
      li.src=a.href; lb.style.display='flex';
    });
  });
  lb.addEventListener('click',function(){ lb.style.display='none'; });
})();
</script>
"""


def _badge(fig_type):
    label = 'behavior' if fig_type == 'behavior' else 'cascade'
    return f'<span class="badge {label}">{label}</span>'


def _diff_badge_html(fig):
    status = fig.get('diff_status', 'no-ref')
    diff   = fig.get('diff')
    if status == 'match':
        return f'<span class="diff-badge diff-match">match (Δ={diff:.4f})</span>'
    if status == 'changed':
        return f'<span class="diff-badge diff-changed">CHANGED (Δ={diff:.4f})</span>'
    if status == 'shape-changed':
        return '<span class="diff-badge diff-shape">layout changed</span>'
    if status == 'no-ref':
        return '<span class="diff-badge diff-noref">no reference</span>'
    return ''


def _figure_card(fig):
    rel    = fig.get('rel', '')
    title  = fig.get('title', '')
    desc   = fig.get('description', '')
    exp    = fig.get('expected', '')
    cit    = fig.get('citation', '')
    ftype  = fig.get('type', 'behavior')
    refrel = fig.get('ref_rel', '')

    path  = fig.get('path', '')
    if path and not os.path.isfile(path):
        img_html = '<div style="padding:30px;text-align:center;color:#aaa;font-size:13px;">Figure not yet generated</div>'
    elif refrel:
        img_html = (
            '<div class="fig-pair">'
            f'  <div><div class="lbl">current</div>'
            f'    <a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a></div>'
            f'  <div><div class="lbl">reference</div>'
            f'    <a href="{refrel}" target="_blank"><img src="{refrel}" alt="{title} (reference)"></a></div>'
            '</div>'
        )
    else:
        img_html = f'<a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a>'

    return f"""
    <div class="fig-card">
      {_diff_badge_html(fig)}
      {img_html}
      <h3>{title}</h3>
      <p class="desc">{desc}</p>
      <div class="expected">
        <strong>Expected finding</strong>
        {exp}
      </div>
      <p class="citation">&#128214; {cit}</p>
      {_badge(ftype)}
    </div>"""


def _section_html(section_meta, figs):
    sid   = section_meta.get('id', '')
    title = section_meta.get('title', '')
    desc  = section_meta.get('description', '')
    cards = '\n'.join(_figure_card(f) for f in figs)
    return f"""
  <section class="section" id="{sid}">
    <h2>{title}</h2>
    <p>{desc}</p>
    <div class="fig-grid">
      {cards}
    </div>
  </section>"""


def generate_html(sections_data):
    """Generate docs/clinical_benchmarks/index.html."""
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__

    nav_links = '\n'.join(
        f'    <a href="#{s["id"]}">{s["title"]}</a>'
        for s, _ in sections_data
    )
    sections_html = '\n'.join(_section_html(s, f) for s, f in sections_data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OculomotorJax — Clinical Benchmark Report</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
  <nav>
    <h2>Clinical Sections</h2>
{nav_links}
  </nav>
  <main>
    <h1>OculomotorJax — Clinical Benchmarks</h1>
    <p class="meta">
      Generated: <strong>{ts}</strong> &nbsp;|&nbsp;
      Version: <strong>{ver}</strong> &nbsp;|&nbsp;
      <a href="../../docs/benchmarks/index.html">Normal Benchmarks</a>
    </p>
    <div class="banner">
      &#9883; Clinical simulation suite — lesion models and pathological eye movement patterns.
      All simulations use deterministic parameters (noise suppressed) for reproducible figures.
    </div>
{sections_html}
  </main>
  {_HTML_LIGHTBOX}
</body>
</html>"""

    os.makedirs(utils.BENCH_DIR, exist_ok=True)
    with open(utils.HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nHTML report written: {utils.HTML_PATH}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(utils.FIGS_DIR, exist_ok=True)
    sections_data = []

    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        if HTML_ONLY:
            figs = []
            for attr in ['FIGURES', '_FIGS']:
                if hasattr(mod, attr) and getattr(mod, attr) is not None:
                    figs = getattr(mod, attr)
                    break
            if not figs:
                try:
                    figs = mod.run(show=False)
                except Exception as e:
                    print(f'  Warning: {mod_name}.run() failed: {e}')
                    figs = []
        else:
            print(f'\n[{mod_name}]')
            try:
                figs = mod.run(show=SHOW)
            except Exception as e:
                print(f'  ERROR in {mod_name}: {e}')
                import traceback; traceback.print_exc()
                figs = []
        # Reference-comparison enrichment (no-op if no reference present).
        figs = [bench_utils.ref_meta(dict(f), base_dir=utils.CLIN_DIR,
                                     ref_dir=utils.CLIN_REF_DIR) for f in figs]
        sections_data.append((mod.SECTION, figs))

    generate_html(sections_data)
    print(f'\nDone. Open: {utils.HTML_PATH}')

    all_figs = [f for _, figs in sections_data for f in figs]
    if all_figs:
        from collections import Counter
        tally = Counter(f.get('diff_status', 'no-ref') for f in all_figs)
        print(f'\nReference comparison: {dict(tally)}')
        changed = [f['title'] for f in all_figs if f.get('diff_status') == 'changed']
        if changed:
            print('  Changed vs. reference:')
            for title in changed:
                print(f'    - {title}')


if __name__ == '__main__':
    main()
