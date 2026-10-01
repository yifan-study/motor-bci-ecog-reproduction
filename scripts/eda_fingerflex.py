#!/usr/bin/env python3
"""Exploratory data analysis for fingerflex data (Milestone 1.2).

Generates and saves plots for:
  1. Raw ECoG traces (5s window) for 3 patients
  2. Finger flexion signals over time
  3. Stimulus/cue codes over time
  4. Power spectral density (PSD) — check existing filtering

Saves all figures to results/eda/.

Usage:
    python scripts/eda_fingerflex.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.load_fingerflex import PATIENTS, load_fingerflex

SAVE_DIR = "results/eda"
SR = 1000


def setup():
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")


def plot_ecog_traces(patients=("bp", "ht", "wm"), start_sec=100, duration_sec=5):
    """Plot raw ECoG traces for a few patients (5-second window)."""
    fig, axes = plt.subplots(len(patients), 1, figsize=(14, 4 * len(patients)))
    if len(patients) == 1:
        axes = [axes]

    for ax, pid in zip(axes, patients):
        d = load_fingerflex(pid)
        ecog = d["ecog"]
        start = start_sec * SR
        end = start + duration_sec * SR
        snippet = ecog[start:end, :]

        t = np.arange(snippet.shape[0]) / SR + start_sec
        # Plot every channel, offset vertically for visibility
        offsets = np.arange(snippet.shape[1]) * snippet.std() * 3
        for ch in range(snippet.shape[1]):
            ax.plot(t, snippet[:, ch] + offsets[ch], linewidth=0.3, color="C0", alpha=0.7)
        ax.set_title(f"Patient {pid} — {snippet.shape[1]} channels, raw ECoG")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel (offset)")
        ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "01_ecog_traces.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_finger_flexion(patients=("bp", "cc")):
    """Plot all 5 finger flexion signals for a few patients."""
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    fig, axes = plt.subplots(len(patients), 1, figsize=(14, 4 * len(patients)))
    if len(patients) == 1:
        axes = [axes]

    for ax, pid in zip(axes, patients):
        d = load_fingerflex(pid)
        flex = d["flex"]
        t = np.arange(flex.shape[0]) / SR

        for f in range(5):
            ax.plot(t, flex[:, f], linewidth=0.5, label=finger_names[f], alpha=0.8)
        ax.set_title(f"Patient {pid} — finger flexion signals")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flexion (raw)")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "02_finger_flexion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_cue_stim(patients=("bp", "cc")):
    """Plot cue and stim codes over time to understand trial structure."""
    fig, axes = plt.subplots(len(patients), 2, figsize=(14, 3 * len(patients)))
    if len(patients) == 1:
        axes = axes.reshape(1, -1)

    for i, pid in enumerate(patients):
        d = load_fingerflex(pid)
        t = np.arange(len(d["cue"])) / SR

        axes[i, 0].plot(t, d["cue"], linewidth=0.5, color="C1")
        axes[i, 0].set_title(f"{pid} — cue codes")
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_ylabel("Cue value")

        axes[i, 1].plot(t, d["stim"], linewidth=0.5, color="C2")
        axes[i, 1].set_title(f"{pid} — stim codes")
        axes[i, 1].set_xlabel("Time (s)")
        axes[i, 1].set_ylabel("Stim value")

        # Print unique values
        print(f"  {pid}: cue unique={np.unique(d['cue']).astype(int).tolist()}, "
              f"stim unique={np.unique(d['stim']).astype(int).tolist()}")

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "03_cue_stim.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_psd(patients=("bp", "ht", "wm"), channels=(0, 10, 20)):
    """Compute and plot PSD to check what filtering is already in the raw data."""
    fig, axes = plt.subplots(len(patients), 1, figsize=(12, 4 * len(patients)))
    if len(patients) == 1:
        axes = [axes]

    for ax, pid in zip(axes, patients):
        d = load_fingerflex(pid)
        ecog = d["ecog"]

        for ch in channels:
            if ch >= ecog.shape[1]:
                continue
            f, psd = welch(ecog[:, ch], fs=SR, nperseg=4096)
            ax.semilogy(f, psd, linewidth=0.8, label=f"ch {ch}", alpha=0.8)

        # Mark 60 Hz and harmonics
        for harmonic in [60, 120, 180]:
            ax.axvline(harmonic, color="red", linestyle="--", linewidth=0.5, alpha=0.5)

        ax.set_title(f"Patient {pid} — PSD (raw, unfiltered)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power spectral density")
        ax.set_xlim(0, 300)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "04_psd_raw.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def main():
    setup()
    print("=== EDA: Fingerflex Data ===\n")

    print("1. Plotting raw ECoG traces...")
    plot_ecog_traces()

    print("\n2. Plotting finger flexion signals...")
    plot_finger_flexion()

    print("\n3. Plotting cue/stim codes...")
    plot_cue_stim()

    print("\n4. Plotting PSD...")
    plot_psd()

    print(f"\nAll plots saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
