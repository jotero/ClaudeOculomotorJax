"""OculomotorJax — JAX-based differentiable primate oculomotor model."""

import subprocess
import os


def _get_version() -> str:
    """Return a version string derived from git describe."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        v = subprocess.check_output(
            ['git', 'describe', '--tags', '--always', '--dirty'],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return v
    except Exception:
        return 'unknown'


__version__ = _get_version()
