"""Shared utilities for clinical benchmark scripts (scripts/bench_clinical_*.py).

Output directory: docs/clinical_benchmarks/figures/   (images)
HTML report:      docs/clinical_benchmarks/index.html

Mirrors bench_utils.py but writes to a separate docs/clinical_benchmarks/ tree.
"""

import os
import sys
import datetime

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.normpath(os.path.join(_SCRIPTS, '..'))
_SRC     = os.path.join(_ROOT, 'src')
_DOCS    = os.path.join(_ROOT, 'docs')

for _p in [_SCRIPTS, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oculomotor  # noqa: E402

DOCS_DIR  = _DOCS
BENCH_DIR = os.path.join(_DOCS, 'clinical_benchmarks')
FIGS_DIR  = os.path.join(BENCH_DIR, 'figures')
REF_DIR   = os.path.join(BENCH_DIR, 'reference')
HTML_PATH = os.path.join(BENCH_DIR, 'index.html')

# Aliases so run_clinical_benchmarks.py can read them as utils.CLIN_*
CLIN_DIR     = BENCH_DIR
CLIN_FIGS_DIR = FIGS_DIR
CLIN_REF_DIR  = REF_DIR
CLIN_HTML_PATH = HTML_PATH


def save_fig(fig, name, show=False, dpi=150):
    """Save figure to docs/clinical_benchmarks/figures/{name}.png; return (path, rel)."""
    import matplotlib.pyplot as plt
    os.makedirs(FIGS_DIR, exist_ok=True)

    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__
    fig.text(0.998, 0.003, f'{ts}  |  {ver}',
             ha='right', va='bottom', fontsize=6, color='#888888',
             transform=fig.transFigure)

    path = os.path.join(FIGS_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    rp = os.path.relpath(path, BENCH_DIR).replace('\\', '/')
    print(f'  [{name}] saved → {os.path.basename(path)}')
    return path, rp


def fig_meta(path, rp, title, description, expected, citation, fig_type='behavior'):
    return dict(path=path, rel=rp, title=title, description=description,
                expected=expected, citation=citation, type=fig_type)


# ── Shared color palette ───────────────────────────────────────────────────────

C = dict(
    head='#555555',
    eye='#2166ac',
    scene='#1b7837',
    target='#d6604d',
    burst='#b2182b',
    vs='#35978f',
    ni='#4dac26',
    dark='#999999',
    spv='#2166ac',
    no_vis='#d6604d',
    canal='#762a83',
    pursuit='#1a9850',
    refractory='#762a83',
)
