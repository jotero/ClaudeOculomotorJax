"""Freeze the current state of bench figures as the regression reference.

Copies PNGs from ``docs/<bench>/figures/`` to ``docs/<bench>/reference/`` so the
HTML report can show a side-by-side comparison and flag any drift on later runs.

Usage:
    python -X utf8 scripts/freeze_reference.py <name> [<name>...]
        # Freeze specific figure(s) by base name (no .png).
        # Searches both main and clinical figures dirs.

    python -X utf8 scripts/freeze_reference.py --all
        # Freeze every PNG currently in either figures dir.

    python -X utf8 scripts/freeze_reference.py --all --main
    python -X utf8 scripts/freeze_reference.py --all --clinical
        # Restrict to one bench suite.

Examples:
    python -X utf8 scripts/freeze_reference.py gravity_tilt_suppression
    python -X utf8 scripts/freeze_reference.py --all --main
"""

import argparse
import os
import shutil
import sys

# Repo-relative paths
_REPO   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIRS = {
    "main":     {
        "figs": os.path.join(_REPO, "docs", "benchmarks",          "figures"),
        "ref":  os.path.join(_REPO, "docs", "benchmarks",          "reference"),
    },
    "clinical": {
        "figs": os.path.join(_REPO, "docs", "clinical_benchmarks", "figures"),
        "ref":  os.path.join(_REPO, "docs", "clinical_benchmarks", "reference"),
    },
}


def _find_figure(name, suite_filter=None):
    """Return list of (suite, src_path) matches for a base figure name."""
    matches = []
    if not name.endswith(".png"):
        name = name + ".png"
    for suite, paths in _DIRS.items():
        if suite_filter and suite != suite_filter:
            continue
        candidate = os.path.join(paths["figs"], name)
        if os.path.isfile(candidate):
            matches.append((suite, candidate))
    return matches


def _freeze_one(suite, src_path):
    """Copy one figure to its reference folder."""
    fname    = os.path.basename(src_path)
    ref_dir  = _DIRS[suite]["ref"]
    os.makedirs(ref_dir, exist_ok=True)
    dst_path = os.path.join(ref_dir, fname)
    shutil.copy2(src_path, dst_path)
    rel_dst  = os.path.relpath(dst_path, _REPO).replace("\\", "/")
    print(f"  [{suite}] {fname}  ->  {rel_dst}")


def _all_figures(suite_filter=None):
    """List all (suite, path) pairs of PNGs currently in figures/ dirs."""
    out = []
    for suite, paths in _DIRS.items():
        if suite_filter and suite != suite_filter:
            continue
        figs_dir = paths["figs"]
        if not os.path.isdir(figs_dir):
            continue
        for fname in sorted(os.listdir(figs_dir)):
            if fname.lower().endswith(".png"):
                out.append((suite, os.path.join(figs_dir, fname)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*",
                    help="Figure base names (without .png) to freeze.")
    ap.add_argument("--all", action="store_true",
                    help="Freeze every PNG currently in the figures dir(s).")
    ap.add_argument("--main", action="store_true",
                    help="Restrict to docs/benchmarks (main suite).")
    ap.add_argument("--clinical", action="store_true",
                    help="Restrict to docs/clinical_benchmarks (clinical suite).")
    args = ap.parse_args()

    if args.main and args.clinical:
        print("--main and --clinical are mutually exclusive.")
        sys.exit(2)
    suite_filter = "main" if args.main else "clinical" if args.clinical else None

    if args.all:
        targets = _all_figures(suite_filter)
        if not targets:
            print("No PNG files found to freeze.")
            sys.exit(1)
        print(f"Freezing {len(targets)} figure(s):")
        for suite, src in targets:
            _freeze_one(suite, src)
        return

    if not args.names:
        ap.print_help()
        sys.exit(1)

    total = 0
    for name in args.names:
        matches = _find_figure(name, suite_filter)
        if not matches:
            print(f"  WARNING: no figure named '{name}' found.")
            continue
        for suite, src in matches:
            _freeze_one(suite, src)
            total += 1

    if total == 0:
        sys.exit(1)
    print(f"\nFroze {total} figure(s).")


if __name__ == "__main__":
    main()
