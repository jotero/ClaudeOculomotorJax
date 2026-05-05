"""Shared utilities for benchmark scripts (scripts/bench_*.py).

Output directory: docs/figures/   (images)
HTML report:      docs/index.html
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
BENCH_DIR = os.path.join(_DOCS, 'benchmarks')
FIGS_DIR  = os.path.join(BENCH_DIR, 'figures')
REF_DIR   = os.path.join(BENCH_DIR, 'reference')
HTML_PATH = os.path.join(BENCH_DIR, 'index.html')

EXPT_DIR      = os.path.join(_DOCS, 'experiments')
EXPT_FIGS_DIR = os.path.join(EXPT_DIR, 'figures')
EXPT_HTML_PATH = os.path.join(EXPT_DIR, 'index.html')

CLIN_DIR      = os.path.join(_DOCS, 'clinical_benchmarks')
CLIN_FIGS_DIR = os.path.join(CLIN_DIR, 'figures')
CLIN_REF_DIR  = os.path.join(CLIN_DIR, 'reference')
CLIN_HTML_PATH = os.path.join(CLIN_DIR, 'index.html')


def fmt_param_overrides(params):
    """Diff `params` against PARAMS_DEFAULT and return a one-line override string.

    Returns 'defaults' if nothing differs, else a comma-separated list of the
    non-default fields, e.g. 'AC_A=0, CA_C=0, sigma_acc=0'. Walks the three
    sub-NamedTuples (sensory, brain, plant) and reports any field whose value
    differs from default. Handles scalars and arrays.
    """
    from oculomotor.sim.simulator import PARAMS_DEFAULT
    import numpy as np
    overrides = []
    for sub in ('sensory', 'brain', 'plant'):
        cur, ref = getattr(params, sub), getattr(PARAMS_DEFAULT, sub)
        for fname in cur._fields:
            cv = getattr(cur, fname)
            rv = getattr(ref, fname)
            try:
                same = bool(np.allclose(np.asarray(cv), np.asarray(rv)))
            except (TypeError, ValueError):
                same = (cv == rv)
            if not same:
                # Format the override value compactly
                cv_arr = np.asarray(cv)
                if cv_arr.shape == ():
                    val_str = f'{float(cv):g}'
                else:
                    val_str = np.array2string(cv_arr, precision=3, separator=',',
                                              suppress_small=True, max_line_width=80)
                overrides.append(f'{fname}={val_str}')
    return ', '.join(overrides) if overrides else 'defaults'


def save_fig(fig, name, show=False, dpi=150, figs_dir=None, base_dir=None,
             params=None, conditions=None):
    """Save figure to {figs_dir}/{name}.png with watermark; return (path, rel).

    The timestamp and git version are embedded in the bottom-right corner.
    If `params` is provided, a one-line list of non-default overrides is also
    embedded in the bottom-left.  If `conditions` is provided, a one-line
    stimulus-conditions string (e.g. 'Lit, midline target stepping 3 m → 0.3 m')
    is embedded just above the overrides line.

    The filename is fixed (no timestamp) so re-running a script overwrites
    the previous figure.

    figs_dir: directory to save PNGs (default: FIGS_DIR)
    base_dir: root for computing rel path in HTML (default: BENCH_DIR)
    """
    import matplotlib.pyplot as plt
    if figs_dir is None:
        figs_dir = FIGS_DIR
    if base_dir is None:
        base_dir = BENCH_DIR
    os.makedirs(figs_dir, exist_ok=True)

    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__
    fig.text(0.998, 0.003, f'{ts}  |  {ver}',
             ha='right', va='bottom', fontsize=6, color='#888888',
             transform=fig.transFigure)

    # Bottom-left footer: parameter overrides only. The `conditions` argument is
    # kept for API compatibility but no longer rendered — it tended to drift out
    # of sync with the actual stimulus across edits, so we trust the param
    # overrides line (read directly from params) and rely on each bench's main
    # suptitle to describe the stimulus.
    if params is not None:
        overrides = fmt_param_overrides(params)
        fig.text(0.005, 0.003, f'Param overrides: {overrides}',
                 ha='left', va='bottom', fontsize=6, color='#888888',
                 transform=fig.transFigure)

    path = os.path.join(figs_dir, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    if show:
        # Non-blocking show so multiple figures accumulate; the bench script's
        # __main__ should call plt.show() (blocking) at the very end to keep
        # the windows open until the user closes them.
        plt.show(block=False)
    else:
        plt.close(fig)
    rp = os.path.relpath(path, base_dir).replace('\\', '/')
    print(f'  [{name}] saved → {os.path.basename(path)}')
    return path, rp


def fig_meta(path, rp, title, description, expected, citation, fig_type='behavior'):
    return dict(path=path, rel=rp, title=title, description=description,
                expected=expected, citation=citation, type=fig_type)


# ── Reference-comparison helpers (regression tracking) ───────────────────────

def reference_for(fig_path, base_dir=None, ref_dir=None):
    """Return path to the reference PNG for a given figure, or None if no ref.

    By default, looks in ``<docs/<bench>/reference/<basename>``; pass an
    explicit ref_dir to override (e.g. CLIN_REF_DIR for clinical benches).
    """
    if base_dir is None:
        base_dir = BENCH_DIR
    if ref_dir is None:
        ref_dir = REF_DIR
    if not fig_path:
        return None
    fname = os.path.basename(fig_path)
    candidate = os.path.join(ref_dir, fname)
    return candidate if os.path.isfile(candidate) else None


def diff_metric(current_path, reference_path):
    """Mean per-pixel absolute difference (0..1) between two PNGs.

    Returns None if PIL/numpy unavailable, files don't exist, or the two
    images have different dimensions (e.g., layout changed). 0 = pixel-identical;
    typical "noise-floor" matplotlib re-renders are 0.001–0.01; large layout or
    behaviour shifts run > 0.02.
    """
    try:
        import numpy as _np
        from PIL import Image
    except ImportError:
        return None
    if not (current_path and reference_path
            and os.path.isfile(current_path) and os.path.isfile(reference_path)):
        return None
    try:
        a = _np.asarray(Image.open(current_path).convert('RGB'), dtype=_np.float32)
        b = _np.asarray(Image.open(reference_path).convert('RGB'), dtype=_np.float32)
    except Exception:
        return None
    if a.shape != b.shape:
        return None  # layout drift — flag as "shape changed" instead of a number
    return float(_np.mean(_np.abs(a - b)) / 255.0)


def ref_meta(fig_dict, base_dir=None, ref_dir=None, diff_threshold=0.005):
    """Augment a fig_meta dict with reference-comparison fields.

    Adds:
        ref_path     : absolute path to reference PNG, or None
        ref_rel      : HTML-relative path, or ''
        diff         : float [0,1] mean per-pixel diff, or None
        diff_status  : 'match' | 'changed' | 'shape-changed' | 'no-ref' | 'unavailable'

    Modifies fig_dict in place AND returns it for chaining.
    """
    fig_path = fig_dict.get('path', '')
    ref_path = reference_for(fig_path, base_dir=base_dir, ref_dir=ref_dir)
    if ref_path is None:
        fig_dict.update(ref_path=None, ref_rel='', diff=None, diff_status='no-ref')
        return fig_dict
    ref_rel = os.path.relpath(ref_path, base_dir or BENCH_DIR).replace('\\', '/')
    diff = diff_metric(fig_path, ref_path)
    if diff is None:
        # Either tooling unavailable or shape mismatch — distinguish below.
        try:
            import numpy as _np  # noqa: F401
            from PIL import Image  # noqa: F401
            status = 'shape-changed'
        except ImportError:
            status = 'unavailable'
    elif diff <= diff_threshold:
        status = 'match'
    else:
        status = 'changed'
    fig_dict.update(ref_path=ref_path, ref_rel=ref_rel, diff=diff, diff_status=status)
    return fig_dict


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
