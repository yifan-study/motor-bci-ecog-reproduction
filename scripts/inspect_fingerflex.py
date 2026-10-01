#!/usr/bin/env python3
"""Inspect all 9 fingerflex patients from the Miller ECoG Library.

Prints a summary table with channel counts, samples, duration, flex range,
and data types. This is Milestone 1.1 of the roadmap.

Usage:
    python scripts/inspect_fingerflex.py
    python scripts/inspect_fingerflex.py --data-dir data/fingerflex
"""

import argparse
import os
import sys

import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load_fingerflex import PATIENTS, load_fingerflex


def main():
    parser = argparse.ArgumentParser(description="Inspect fingerflex data files.")
    parser.add_argument("--data-dir", default="data/fingerflex/data",
                        help="Directory containing patient subdirectories with .mat files")
    args = parser.parse_args()

    print("=" * 90)
    print("Fingerflex Data Inspection — Miller ECoG Library (2019)")
    print("=" * 90)

    header = (
        f"{'Patient':<9} {'Channels':>8} {'Samples':>10} {'Duration(s)':>12} "
        f"{'Flex dims':>9} {'Flex range':>14} {'ECoG dtype':>12} {'Flex dtype':>12}"
    )
    print(f"\n{header}")
    print("-" * 90)

    results = []
    failed = []

    for pid in PATIENTS:
        try:
            d = load_fingerflex(pid, data_dir=args.data_dir)
            ecog = d["ecog"]
            flex = d["flex"]
            stim = d["stim"]
            sr = d["sr"]

            n_samples, n_channels = ecog.shape
            duration = n_samples / sr
            flex_min = flex.min()
            flex_max = flex.max()
            n_flex = flex.shape[1]

            # Check for issues
            has_nan = np.isnan(ecog).any() or np.isnan(flex).any()
            has_inf = np.isinf(ecog).any() or np.isinf(flex).any()
            const_channels = np.sum(ecog.std(axis=0) == 0)

            row = {
                "patient": pid,
                "channels": n_channels,
                "samples": n_samples,
                "duration": duration,
                "n_flex": n_flex,
                "flex_min": flex_min,
                "flex_max": flex_max,
                "ecog_dtype": str(ecog.dtype),
                "flex_dtype": str(flex.dtype),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "const_channels": const_channels,
                "stim_unique": len(np.unique(stim)),
            }
            results.append(row)

            flex_range_str = f"[{flex_min:.1f}, {flex_max:.1f}]"
            print(
                f"{pid:<9} {n_channels:>8} {n_samples:>10,} {duration:>12.1f} "
                f"{n_flex:>9} {flex_range_str:>14} {str(ecog.dtype):>12} {str(flex.dtype):>12}"
            )

        except Exception as e:
            failed.append((pid, str(e)))
            print(f"{pid:<9} {'FAILED':>8}  — {e}")

    print("-" * 90)

    # Summary statistics
    if results:
        channels = [r["channels"] for r in results]
        durations = [r["duration"] for r in results]
        print(f"\nLoaded: {len(results)}/{len(PATIENTS)} patients")
        print(f"Channel range: {min(channels)} — {max(channels)}")
        print(f"Duration range: {min(durations):.1f}s — {max(durations):.1f}s")

        # Data quality checks
        print("\n--- Data Quality ---")
        for r in results:
            issues = []
            if r["has_nan"]:
                issues.append("NaN values")
            if r["has_inf"]:
                issues.append("Inf values")
            if r["const_channels"] > 0:
                issues.append(f"{r['const_channels']} constant channels")
            if issues:
                print(f"  {r['patient']}: WARNING — {', '.join(issues)}")
            else:
                print(f"  {r['patient']}: OK")

    if failed:
        print(f"\nFailed to load {len(failed)} patients:")
        for pid, err in failed:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    main()
