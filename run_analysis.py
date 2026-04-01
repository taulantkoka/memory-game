#!/usr/bin/env python3
"""
============================================================================
  Memory Game — Bounded Working Memory Analysis
============================================================================

  Runs the full analysis pipeline and saves all figures as PDF + SVG
  to the figures/ directory.

  Usage:
    python run_analysis.py              # Run everything
    python run_analysis.py --only 01    # Run only Miller vs Zwick
    python run_analysis.py --list       # Show available analyses

  Project structure:
    simulations/  → Analysis scripts (00–04)
    figures/      → Output (PDF + SVG)
    report/       → Report markdown
============================================================================
"""

import sys
import os
import argparse
from pathlib import Path

# ---- Path setup ----
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "simulations"))
os.chdir(ROOT)

# Ensure figures/ exists
(ROOT / "figures").mkdir(exist_ok=True)

# ---- Monkey-patch matplotlib to save to figures/ as PDF + SVG ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_original_savefig = plt.Figure.savefig

def _patched_savefig(self, fname, *args, **kwargs):
    """Redirect all saves to figures/ as BOTH PDF and SVG."""
    fname = str(fname)
    base = os.path.splitext(os.path.basename(fname))[0]

    pdf_path = ROOT / "figures" / f"{base}.pdf"
    svg_path = ROOT / "figures" / f"{base}.svg"

    kwargs.setdefault("bbox_inches", "tight")

    _original_savefig(self, str(pdf_path), *args, **kwargs)
    _original_savefig(self, str(svg_path), format='svg', **kwargs)

    print(f"  -> {pdf_path.name} + {svg_path.name}")

plt.Figure.savefig = _patched_savefig


# ---- Analysis modules ----
ANALYSES = {
    "00": ("Exact DP computation",              "simulations/00_exact_dp.py"),
    "01": ("Bounded-memory optimum vs Zwick",   "simulations/01_bounded_vs_zwick.py"),
    "02": ("Fluctuation robustness",            "simulations/02_fluctuation.py"),
    "03": ("Asymmetric memory",                 "simulations/03_asymmetric.py"),
    "04": ("Draw rate vs capacity",             "simulations/04_draw_rate.py"),
}


def run_script(path):
    """Execute an analysis script."""
    full_path = ROOT / path
    if not full_path.exists():
        print(f"  SKIP: {path} not found")
        return False

    print(f"\n{'='*70}")
    print(f"  Running: {path}")
    print(f"{'='*70}")

    script_dir = str(full_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        exec(open(full_path).read(), {
            "__name__": "__main__",
            "__file__": str(full_path)
        })
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Memory game analysis")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific analysis (e.g. '01')")
    parser.add_argument("--list", action="store_true",
                        help="List available analyses")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable analyses:")
        for key, (name, path) in ANALYSES.items():
            print(f"  {key}: {name} ({path})")
        return

    print("\n" + "=" * 70)
    print("  MEMORY GAME — BOUNDED WORKING MEMORY ANALYSIS")
    print("=" * 70)
    print(f"  Project root: {ROOT}")
    print(f"  Figures dir:  {ROOT / 'figures'}")

    results = {}

    for key, (name, path) in ANALYSES.items():
        if args.only and args.only != key:
            continue
        success = run_script(path)
        results[key] = (name, success)

    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    for key, (name, success) in results.items():
        status = "✓" if success else "✗"
        print(f"  {status}  {key}: {name}")

    # List generated figures
    figs = sorted((ROOT / "figures").glob("*.pdf"))
    if figs:
        print(f"\n  Generated {len(figs)} figure pairs in figures/:")
        for f in figs:
            print(f"    {f.name}")

    print(f"\n  Report: report/report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
