"""Experimental / exploratory benchmarks.

Currently contains:
    1. Monocular occlusion — binocular fixation maintenance under three viewing conditions.

Usage:
    python -X utf8 scripts/bench_experiments.py
    python -X utf8 scripts/bench_experiments.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as utils

import numpy as np
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, SimConfig, simulate,
    _IDX_PURSUIT,
)
from oculomotor.sim.kinematics import build_target
from oculomotor.analysis import extract_burst

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = PARAMS_DEFAULT   # noiseless — deterministic


# ── Monocular occlusion ────────────────────────────────────────────────────────

_T_END  = 15.0
_T_FIX  = 5.0       # binocular fixation period before occlusion onset (s)
_DIST_M = 0.15      # target distance (m) — straight ahead, 15 cm

_CFG = SimConfig(warmup_s=30.0)   # 5 × tau_verg → vergence fully settled

_COND_LABELS = {
    'dark':       'Dark (both lose target)',
    'strobed':    'Strobed (position only)',
    'continuous': 'Continuous (monocular)',
}


def _make_flags(t_np, cond, occ_eye):
    T    = len(t_np)
    ones = np.ones(T,  dtype=np.float32)
    off  = np.where(t_np >= _T_FIX, 0.0, 1.0).astype(np.float32)
    ts   = np.where(t_np >= _T_FIX, 1.0, 0.0).astype(np.float32)

    if cond == 'dark':
        return off, off, np.zeros(T, dtype=np.float32)

    if occ_eye == 'left':
        tL, tR = off, ones
    else:
        tL, tR = ones, off

    strobed = ts if cond == 'strobed' else np.zeros(T, dtype=np.float32)
    return tL, tR, strobed


def _run_cond(t_np, cond, occ_eye):
    t  = jnp.array(t_np)
    T  = len(t_np)
    pt = jnp.tile(jnp.array([0.0, 0.0, _DIST_M]), (T, 1))
    tL, tR, ts = _make_flags(t_np, cond, occ_eye)
    return simulate(
        THETA, t,
        target                 = build_target(t, lin_pos=pt),
        scene_present_array    = jnp.zeros(T),
        target_present_L_array = jnp.array(tL),
        target_present_R_array = jnp.array(tR),
        target_strobed_array   = jnp.array(ts),
        return_states          = True,
        sim_config             = _CFG,
    )


def _occlusion(show):
    t_np = np.arange(0.0, _T_END, DT, dtype=np.float32)
    conditions = ['dark', 'strobed', 'continuous']

    results = {}
    s = _run_cond(t_np, 'dark', 'left')
    results[('dark', 'left')]  = s
    results[('dark', 'right')] = s   # symmetric

    for cond in ['strobed', 'continuous']:
        for occ in ['left', 'right']:
            results[(cond, occ)] = _run_cond(t_np, cond, occ)

    N_ROWS = 5
    fig, axes = plt.subplots(N_ROWS, 3, figsize=(13, 14), sharex=True)
    fig.suptitle(
        'Monocular occlusion — binocular fixation at 15 cm, dark room\n'
        'Vertical line = occlusion onset (t = 5 s)',
        fontsize=10,
    )

    row_labels = [
        'Left eye yaw (deg)',
        'Right eye yaw (deg)',
        'Vergence  L−R (deg)',
        'Pursuit cmd (deg/s)',
        'Saccade burst (deg/s)',
    ]
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8.5)

    for ci, cond in enumerate(conditions):
        axes[0, ci].set_title(_COND_LABELS[cond], fontsize=9)
        axes[N_ROWS - 1, ci].set_xlabel('Time (s)', fontsize=8)

        for occ in ['left', 'right']:
            st = results[(cond, occ)]

            eye_L    = np.array(st.plant[:, 0])
            eye_R    = np.array(st.plant[:, 3])
            vergence = eye_L - eye_R
            pursuit  = np.array(st.brain[:, _IDX_PURSUIT])[:, 0]
            burst    = np.array(extract_burst(st, THETA))[:, 0]

            if cond == 'dark':
                color, ls, lbl = utils.C['dark'],   '-',  'both occluded'
            elif occ == 'left':
                color, ls, lbl = utils.C['eye'],    '-',  'L eye occluded'
            else:
                color, ls, lbl = utils.C['target'], '--', 'R eye occluded'

            kw = dict(color=color, lw=1.5, ls=ls, label=lbl)
            axes[0, ci].plot(t_np, eye_L,    **kw)
            axes[1, ci].plot(t_np, eye_R,    **kw)
            axes[2, ci].plot(t_np, vergence, **kw)
            axes[3, ci].plot(t_np, pursuit,  **kw)
            axes[4, ci].plot(t_np, burst,    **kw)

            if cond == 'dark':
                break

        for row in range(N_ROWS):
            ax = axes[row, ci]
            ax.axvline(_T_FIX, color='gray', lw=0.8, ls='--', alpha=0.5)
            ax.grid(True, alpha=0.15)
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 3.0)
            mid  = 0.5 * (ylo + yhi)
            ax.set_ylim(mid - span / 2, mid + span / 2)
            if ci == 0:
                ax.legend(fontsize=7)

        for ci2 in range(3):
            ax = axes[4, ci2]
            ylo, yhi = ax.get_ylim()
            span = max(yhi - ylo, 20.0)
            mid  = 0.5 * (ylo + yhi)
            axes[4, ci2].set_ylim(mid - span / 2, mid + span / 2)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'occlusion', show=show,
                              figs_dir=utils.EXPT_FIGS_DIR, base_dir=utils.EXPT_DIR)
    return utils.fig_meta(
        path, rp,
        title='Monocular occlusion',
        description='Binocular fixation at 15 cm under dark, strobed, and continuous monocular viewing.',
        expected='Both eyes maintain stable vergence during binocular phase. '
                 'After occlusion: dark → slow drift, strobed → position hold without velocity, '
                 'continuous → stable monocular fixation.',
        citation='Typical clinical dissociated nystagmus / monocular occlusion paradigm',
    )


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='experiments', title='Experimental',
    description='Exploratory paradigms: monocular occlusion, binocular fixation maintenance.',
)


def run(show=False):
    print('\n=== Experiments ===')
    figs = []
    print('  1/1  monocular occlusion …')
    figs.append(_occlusion(show))
    return figs


# ── HTML generation ────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; }
nav  { width: 200px; min-height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; flex-shrink: 0; }
nav h2  { font-size: 13px; padding: 0 16px 12px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav a   { display: block; padding: 8px 16px; color: #ccc; text-decoration: none;
          font-size: 13px; border-left: 3px solid transparent; }
nav a:hover { background: #2a2a4e; color: #fff; border-left-color: #9b59b6; }
main { flex: 1; padding: 32px; max-width: 1400px; }
h1   { font-size: 22px; margin-bottom: 4px; }
.meta { font-size: 12px; color: #888; margin-bottom: 32px; }
.section    { margin-bottom: 48px; }
.section h2 { font-size: 18px; margin-bottom: 6px; border-bottom: 2px solid #9b59b6;
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
.expected   { background: #f5eeff; border-left: 3px solid #9b59b6;
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
"""

_LIGHTBOX = """
<div id="lb" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,.85);z-index:1000;cursor:zoom-out;align-items:center;
     justify-content:center;">
  <img id="lb-img" style="max-width:95vw;max-height:95vh;border-radius:4px;">
</div>
<script>
(function(){
  var lb=document.getElementById('lb'),li=document.getElementById('lb-img');
  document.querySelectorAll('.fig-card a').forEach(function(a){
    a.addEventListener('click',function(e){e.preventDefault();li.src=a.href;lb.style.display='flex';});
  });
  lb.addEventListener('click',function(){lb.style.display='none';});
})();
</script>
"""


def _fig_card(fig):
    rel, title = fig.get('rel',''), fig.get('title','')
    desc, exp  = fig.get('description',''), fig.get('expected','')
    cit        = fig.get('citation','')
    ftype      = fig.get('type','behavior')
    path       = fig.get('path','')
    if path and not os.path.isfile(path):
        img = '<div style="padding:30px;text-align:center;color:#aaa;font-size:13px;">Figure not yet generated</div>'
    else:
        img = f'<a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a>'
    badge = f'<span class="badge {ftype}">{ftype}</span>'
    return f"""
    <div class="fig-card">
      {img}
      <h3>{title}</h3>
      <p class="desc">{desc}</p>
      <div class="expected"><strong>Expected behavior</strong>{exp}</div>
      <p class="citation">&#128214; {cit}</p>
      {badge}
    </div>"""


def generate_html(figs):
    import datetime, oculomotor
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__
    nav_sections = '\n'.join(
        f'    <a href="#{f["title"].lower().replace(" ","_")}">{f["title"]}</a>' for f in figs
    )
    cards = '\n'.join(_fig_card(f) for f in figs)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OculomotorJax — Experiments</title>
  <style>{_CSS}</style>
</head>
<body>
  <nav>
    <h2 style="margin-bottom:4px;">Pages</h2>
    <a href="../">LLM Simulator</a>
    <a href="../benchmarks/">Benchmarks</a>
    <a href="../clinical_benchmarks/">Clinical Benchmarks</a>
    <a href="../parameters.html">Parameters</a>
    <div style="border-top:1px solid #2a2a4e;margin:10px 0 8px;"></div>
    <h2>Experiments</h2>
{nav_sections}
  </nav>
  <main>
    <h1>OculomotorJax — Experiments</h1>
    <p class="meta">
      Generated: <strong>{ts}</strong> &nbsp;|&nbsp;
      Version: <strong>{ver}</strong>
    </p>
    <section class="section">
      <h2>{SECTION['title']}</h2>
      <p>{SECTION['description']}</p>
      <div class="fig-grid">
        {cards}
      </div>
    </section>
  </main>
  {_LIGHTBOX}
</body>
</html>"""
    os.makedirs(utils.EXPT_DIR, exist_ok=True)
    with open(utils.EXPT_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nHTML report written: {utils.EXPT_HTML_PATH}')


if __name__ == '__main__':
    figs = run(show=SHOW)
    generate_html(figs)
