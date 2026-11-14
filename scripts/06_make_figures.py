"""
Master figure generation script for the Solar Flare Forecasting Capstone.
Jointly maintained by:
  - Agent A6 (Visualization Lead - EDA): Figures F00 to F11
  - Agent A7 (Visualization Lead - Evaluation): Figures F12 to F21

Usage:
  python scripts/06_make_figures.py --eda      # Generate EDA figures F00-F11
  python scripts/06_make_figures.py --eval     # Generate Evaluation figures F12-F21
  python scripts/06_make_figures.py --all      # Generate all 22 figures F00-F21
"""

import sys
import argparse
from typing import Dict, Any

from solarflare.viz_eda import generate_all_eda_figures


def run_eda() -> Dict[str, Any]:
    print("=" * 70)
    print("STEP 1: Generating EDA and Clue Figures (F00 to F11)")
    print("=" * 70)
    figs = generate_all_eda_figures(save=True)
    print(f"Generated {len(figs)} EDA figures in results/figures/.\n")
    return figs


def run_eval() -> Dict[str, Any]:
    print("=" * 70)
    print("STEP 2: Generating Evaluation and Performance Figures (F12 to F21)")
    print("=" * 70)
    try:
        from solarflare.viz_eval import generate_all_eval_figures
        figs = generate_all_eval_figures()
        print(f"Generated {len(figs)} evaluation figures in results/figures/.\n")
        return figs
    except Exception as exc:
        print(f"Evaluation figure generator encountered: {exc}", file=sys.stderr)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality figures for Solar Flare Forecasting capstone.")
    parser.add_argument("--eda", action="store_true", help="Generate only EDA figures (F00 to F11)")
    parser.add_argument("--eval", action="store_true", help="Generate only Evaluation figures (F12 to F21)")
    parser.add_argument("--all", action="store_true", help="Generate all 22 figures (F00 to F21)")

    args = parser.parse_args()

    # Default to --all if no specific flag provided
    if not (args.eda or args.eval or args.all):
        args.all = True

    if args.eda:
        run_eda()
    elif args.eval:
        run_eval()
    elif args.all:
        run_eda()
        run_eval()


if __name__ == "__main__":
    main()
