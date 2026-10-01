#!/usr/bin/env python
"""Analyze per-finger decoding difficulty in BCI-IV finger flexion data.

Investigates why certain fingers (especially Middle/ch2) are much harder to
decode than others. Computes target statistics and generates diagnostic plots.

Usage:
    /mnt/beegfs/home/yyu2024/miniconda3/envs/pytorch_ml/bin/python scripts/analyze_fingers.py

Outputs saved to results/finger_analysis/.

Analyses performed:
    1. Per-finger variance (low variance = hard to predict)
    2. Per-finger % time near zero (static = hard)
    3. Cross-correlation matrix between fingers (high corr = hard to disentangle)
    4. Per-finger SNR (movement amplitude vs baseline noise)
    5. Movement event analysis (frequency and amplitude of discrete movements)
    6. Predicted vs actual time-series plots for worst (Middle) vs best finger
"""

import sys
import os

# Set working directory to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, ".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from src.data.load_bci4 import load_bci4
from src.data.preprocessing import normalize_and_car, filter_ecog
from src.data.preprocessing_lomtev import (
    compute_spectrograms, downsample_spectrograms,
    crop_for_time_delay, robust_scale_spectrograms, minmax_scale_flex,
    interpolate_fingerflex,
)
from src.evaluation.metrics import pearson_r_per_channel, smooth_predictions
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


# ── Configuration ─────────────────────────────────────────────────────

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
SUBJECTS = [1, 2, 3]

# Results directories for the channel-dropout runs (best model)
RESULT_DIRS = {
    1: "results/20260323_010508_transformer_1",
    2: "results/20260323_010200_transformer_2",
    3: "results/20260323_005712_transformer_3",
}

OUTPUT_DIR = "results/finger_analysis"

# Per-finger test correlations from the channel dropout run
# (used for labeling and cross-referencing)
REPORTED_TEST_R = {
    1: [0.628, 0.779, 0.383, 0.688, 0.608],
    2: [0.613, 0.581, 0.386, 0.589, 0.551],
    3: [0.838, 0.729, 0.677, 0.800, 0.827],
}


# ── Data Loading ──────────────────────────────────────────────────────

def load_raw_finger_targets(subject_id):
    """Load raw (unprocessed) finger flexion targets for a subject.

    Returns train and test targets at the native 1000 Hz storage rate.
    Shape: (5, time) for each.
    """
    raw = load_bci4(subject_id, data_dir="data/bci4")
    return {
        "train_flex": raw["train_flex"].T,   # (5, time_train @ 1000 Hz)
        "test_flex": raw["test_flex"].T,      # (5, time_test @ 1000 Hz)
        "sr": raw["sr"],
        "flex_sr": raw["flex_sr"],
        "n_channels": raw["n_channels"],
    }


def load_processed_test_data(subject_id, config):
    """Replicate the full BCI-IV Lomtev pipeline for a single subject.

    Returns the processed test spectrogram, test flex targets (after all
    transforms), and the scalers needed for inverse-transforming predictions.
    Also returns the full continuous test data for model evaluation.
    """
    data_cfg = config["data"]
    prep_cfg = config.get("preprocessing_lomtev", config.get("preprocessing", {}))
    win_cfg = config.get("windowing_lomtev", config.get("windowing", {}))
    split_cfg = config["split"]

    # Load raw
    data_dir = data_cfg.get("bci4_data_dir", "data/bci4")
    raw = load_bci4(subject_id, data_dir=data_dir)
    sr = raw["sr"]

    # Z-score + CAR
    train_ecog_norm, _ = normalize_and_car(raw["train_ecog"])
    test_ecog_norm, _ = normalize_and_car(raw["test_ecog"])

    train_ecog_ct = train_ecog_norm.T
    test_ecog_ct = test_ecog_norm.T

    # Bandpass + notch
    l_freq = prep_cfg.get("bandpass_lo", 40.0)
    h_freq = prep_cfg.get("bandpass_hi", 300.0)
    powerline = prep_cfg.get("notch_freqs", [60])[0]

    train_ecog_filt = filter_ecog(
        train_ecog_ct.T, sr=sr, l_freq=l_freq, h_freq=h_freq,
        powerline_freq=powerline
    ).T
    test_ecog_filt = filter_ecog(
        test_ecog_ct.T, sr=sr, l_freq=l_freq, h_freq=h_freq,
        powerline_freq=powerline
    ).T

    # Wavelets + downsample
    n_wavelets = prep_cfg.get("n_wavelets", 40)
    ds_fs = prep_cfg.get("downsample_fs", 100)
    n_jobs = prep_cfg.get("n_jobs", 1)

    train_spec = compute_spectrograms(
        train_ecog_filt, sr=sr, l_freq=l_freq, h_freq=h_freq,
        n_wavelets=n_wavelets, n_jobs=n_jobs
    )
    test_spec = compute_spectrograms(
        test_ecog_filt, sr=sr, l_freq=l_freq, h_freq=h_freq,
        n_wavelets=n_wavelets, n_jobs=n_jobs
    )
    train_spec = downsample_spectrograms(train_spec, cur_fs=sr, new_fs=ds_fs)
    test_spec = downsample_spectrograms(test_spec, cur_fs=sr, new_fs=ds_fs)

    # Interpolate finger data
    train_flex = interpolate_fingerflex(
        raw["train_flex"].T, cur_fs=sr, true_fs=raw["flex_sr"], needed_hz=ds_fs
    )
    test_flex = interpolate_fingerflex(
        raw["test_flex"].T, cur_fs=sr, true_fs=raw["flex_sr"], needed_hz=ds_fs
    )

    # Align lengths
    train_min = min(train_spec.shape[-1], train_flex.shape[-1])
    train_spec = train_spec[..., :train_min]
    train_flex = train_flex[..., :train_min]
    test_min = min(test_spec.shape[-1], test_flex.shape[-1])
    test_spec = test_spec[..., :test_min]
    test_flex = test_flex[..., :test_min]

    # Carve validation from train
    val_frac = split_cfg.get("val_frac", 0.15)
    if val_frac > 0:
        n_train_total = train_spec.shape[-1]
        n_val = int(n_train_total * val_frac)
        n_train = n_train_total - n_val
        train_spec = train_spec[..., :n_train]
        train_flex = train_flex[..., :n_train]

    # Scale (fit on train, apply to test)
    train_spec, _, test_spec, spec_scaler = robust_scale_spectrograms(
        train_spec, None, test_spec
    )
    train_flex, _, test_flex, flex_scaler = minmax_scale_flex(
        train_flex, None, test_flex
    )

    # Time delay crop
    delay_sec = prep_cfg.get("time_delay_sec", 0.2)
    test_flex, test_spec = crop_for_time_delay(test_flex, test_spec, delay_sec, ds_fs)

    return {
        "test_spec": test_spec,       # (channels, wavelets, time) — processed
        "test_flex": test_flex,        # (5, time) — processed (MinMax scaled)
        "flex_scaler": flex_scaler,
        "n_channels": raw["n_channels"],
        "n_wavelets": n_wavelets,
        "ds_fs": ds_fs,
    }


def load_model_and_predict(subject_id, test_spec, config, device):
    """Load the best checkpoint and run inference on the test spectrogram.

    Returns predictions as np.ndarray, shape (5, time).
    """
    exp_dir = RESULT_DIRS[subject_id]
    ckpt_path = os.path.join(exp_dir, "best_model.pt")

    if not os.path.isfile(ckpt_path):
        print(f"  WARNING: checkpoint not found at {ckpt_path}, skipping predictions")
        return None

    # Build model from config
    # We need dataset_info-like dict for build_model
    ds_info = {
        "n_channels": config["data"].get("n_channels_override", None),
        "n_input_features": config.get("preprocessing_lomtev", {}).get("n_wavelets", 40),
    }
    # For BCI-IV, n_channels varies per subject — load from the raw data
    raw = load_bci4(subject_id, data_dir=config["data"].get("bci4_data_dir", "data/bci4"))
    ds_info["n_channels"] = raw["n_channels"]

    model = build_model(config, ds_info)
    state_dict = torch.load(ckpt_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Run inference: test_spec is (channels, wavelets, time)
    inp = torch.from_numpy(test_spec.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze(0).cpu().numpy()  # (5, time)

    # Apply smoothing (sigma=1 as in config)
    eval_cfg = config.get("evaluation", {})
    sigma = eval_cfg.get("smooth_sigma_test", 1)
    if sigma > 0:
        pred = smooth_predictions(pred, sigma)

    return pred


# ── Analysis Functions ────────────────────────────────────────────────

def compute_finger_stats(flex, label=""):
    """Compute comprehensive per-finger statistics.

    Parameters
    ----------
    flex : np.ndarray, shape (5, time)
        Raw finger flexion data (NOT MinMax scaled).
    label : str
        Label for printing.

    Returns
    -------
    dict with per-finger statistics.
    """
    n_fingers, T = flex.shape
    results = {}

    # 1. Per-finger variance
    variances = np.var(flex, axis=1)
    results["variance"] = variances

    # 2. Per-finger standard deviation
    stds = np.std(flex, axis=1)
    results["std"] = stds

    # 3. Per-finger range (max - min)
    ranges = np.ptp(flex, axis=1)
    results["range"] = ranges

    # 4. Percentage of time near zero (absolute value < threshold)
    # Use 10% of each finger's range as the "near zero" threshold
    near_zero_pcts = []
    for f in range(n_fingers):
        # Threshold: 10% of full range, centered on median
        median_val = np.median(flex[f])
        threshold = 0.10 * ranges[f] if ranges[f] > 0 else 1e-8
        near_zero = np.mean(np.abs(flex[f] - median_val) < threshold)
        near_zero_pcts.append(near_zero)
    results["pct_near_median"] = np.array(near_zero_pcts)

    # 5. Percentage of time truly at/near zero (absolute value)
    # Different: how often the finger is at rest (zero flexion)
    abs_threshold = 0.5  # raw data units
    pct_at_zero = np.mean(np.abs(flex) < abs_threshold, axis=1)
    results["pct_near_zero"] = pct_at_zero

    # 6. Kurtosis and skewness
    results["kurtosis"] = scipy_stats.kurtosis(flex, axis=1)
    results["skewness"] = scipy_stats.skew(flex, axis=1)

    # 7. Movement event detection: count discrete movements
    # A "movement" = contiguous region where |derivative| > threshold
    movement_counts = []
    movement_amplitudes = []
    for f in range(n_fingers):
        signal = flex[f]
        # Smooth derivative
        dx = np.diff(signal)
        # Threshold: 2x median absolute deviation
        mad = np.median(np.abs(dx))
        if mad < 1e-8:
            movement_counts.append(0)
            movement_amplitudes.append(0.0)
            continue
        moving = np.abs(dx) > 3 * mad
        # Count transitions from not-moving to moving
        transitions = np.diff(moving.astype(int))
        n_events = np.sum(transitions == 1)
        movement_counts.append(n_events)

        # Average peak-to-trough amplitude of movements
        if n_events > 0:
            # Use the 90th percentile of absolute values as movement amplitude
            movement_amplitudes.append(np.percentile(np.abs(signal), 90))
        else:
            movement_amplitudes.append(0.0)

    results["n_movements"] = np.array(movement_counts)
    results["movement_amplitude_p90"] = np.array(movement_amplitudes)

    # 8. SNR estimate: signal power vs noise power
    # Signal = low-pass filtered version (movement timescale ~ <2 Hz)
    # Noise = residual (high-frequency jitter)
    snr_values = []
    for f in range(n_fingers):
        signal = flex[f]
        # Crude low-pass: running average with window ~0.5s (at whatever sample rate)
        # For 1000 Hz data: window=500, for 100 Hz: window=50
        window = min(500, T // 4)
        if window < 2:
            snr_values.append(0.0)
            continue
        kernel = np.ones(window) / window
        smooth_sig = np.convolve(signal, kernel, mode="same")
        residual = signal - smooth_sig
        signal_power = np.var(smooth_sig)
        noise_power = np.var(residual)
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float("inf")
        snr_values.append(snr_db)
    results["snr_db"] = np.array(snr_values)

    # 9. Cross-correlation matrix between fingers
    corr_matrix = np.corrcoef(flex)
    results["cross_corr"] = corr_matrix

    return results


def print_finger_stats(stats, subject_id, label, test_r=None):
    """Pretty-print per-finger statistics."""
    print(f"\n{'=' * 80}")
    print(f"  Subject {subject_id} — {label}")
    print(f"{'=' * 80}")

    header = f"  {'Finger':<10}"
    for name in FINGER_NAMES:
        header += f" {name:>10}"
    print(header)
    print("  " + "-" * 60)

    if test_r is not None:
        row = f"  {'test_r':<10}"
        for r in test_r:
            row += f" {r:>10.3f}"
        print(row)
        print("  " + "-" * 60)

    metrics = [
        ("variance", "Variance", ".4f"),
        ("std", "Std dev", ".4f"),
        ("range", "Range", ".2f"),
        ("pct_near_median", "% near med", ".1%"),
        ("pct_near_zero", "% near 0", ".1%"),
        ("kurtosis", "Kurtosis", ".2f"),
        ("skewness", "Skewness", ".2f"),
        ("n_movements", "# movements", "d"),
        ("movement_amplitude_p90", "Amp (p90)", ".2f"),
        ("snr_db", "SNR (dB)", ".1f"),
    ]

    for key, name, fmt in metrics:
        row = f"  {name:<10}"
        for v in stats[key]:
            row += f" {v:>10{fmt}}"
        print(row)

    print(f"\n  Cross-correlation matrix:")
    print(f"  {'':>10}", end="")
    for name in FINGER_NAMES:
        print(f" {name:>8}", end="")
    print()
    for i, name_i in enumerate(FINGER_NAMES):
        print(f"  {name_i:>10}", end="")
        for j in range(5):
            val = stats["cross_corr"][i, j]
            if i == j:
                print(f" {'1.000':>8}", end="")
            else:
                print(f" {val:>8.3f}", end="")
        print()


# ── Plotting Functions ────────────────────────────────────────────────

def plot_finger_variance_comparison(all_stats, output_dir):
    """Bar chart comparing per-finger variance across subjects."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Per-Finger Variance of Raw Flexion Targets (Test Set)", fontsize=14)

    for idx, subject_id in enumerate(SUBJECTS):
        ax = axes[idx]
        stats = all_stats[subject_id]
        variances = stats["test"]["variance"]
        colors = ["#d62728" if i == 2 else "#1f77b4" for i in range(5)]  # Red for Middle
        bars = ax.bar(FINGER_NAMES, variances, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(f"Subject {subject_id}", fontsize=12)
        ax.set_ylabel("Variance" if idx == 0 else "")
        ax.tick_params(axis="x", rotation=45)

        # Annotate with test r
        for i, (bar, r) in enumerate(zip(bars, REPORTED_TEST_R[subject_id])):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"r={r:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "finger_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/finger_variance.png")


def plot_cross_correlation_matrices(all_stats, output_dir):
    """Heatmap of finger cross-correlation for each subject."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Cross-Correlation Between Finger Flexion Targets (Test Set)", fontsize=14)

    for idx, subject_id in enumerate(SUBJECTS):
        ax = axes[idx]
        corr = all_stats[subject_id]["test"]["cross_corr"]

        # Mask diagonal for better color contrast
        mask = np.eye(5, dtype=bool)
        corr_display = corr.copy()
        corr_display[mask] = np.nan

        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(5))
        ax.set_xticklabels(FINGER_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(FINGER_NAMES)
        ax.set_title(f"Subject {subject_id}", fontsize=12)

        # Annotate cells
        for i in range(5):
            for j in range(5):
                val = corr[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)

    fig.colorbar(im, ax=axes, label="Pearson r", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cross_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/cross_correlation.png")


def plot_finger_time_series(all_raw, output_dir):
    """Plot raw finger flexion time series for each subject (test set)."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("Raw Finger Flexion Targets — Test Set", fontsize=14)

    for idx, subject_id in enumerate(SUBJECTS):
        ax = axes[idx]
        flex = all_raw[subject_id]["test_flex"]  # (5, time)
        T = flex.shape[1]
        t = np.arange(T) / 1000.0  # time in seconds (1000 Hz storage rate)

        for f in range(5):
            alpha = 1.0 if f == 2 else 0.6  # Highlight Middle finger
            lw = 2.0 if f == 2 else 1.0
            ax.plot(t, flex[f], label=FINGER_NAMES[f], alpha=alpha, linewidth=lw)

        ax.set_title(f"Subject {subject_id}", fontsize=12)
        ax.set_xlabel("Time (s)" if idx == 2 else "")
        ax.set_ylabel("Flexion (raw)")
        ax.legend(loc="upper right", fontsize=8, ncol=5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "finger_time_series.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/finger_time_series.png")


def plot_finger_histograms(all_raw, output_dir):
    """Histogram of finger flexion values for each subject."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    fig.suptitle("Distribution of Finger Flexion Values — Test Set", fontsize=14)

    for idx, subject_id in enumerate(SUBJECTS):
        flex = all_raw[subject_id]["test_flex"]
        for f in range(5):
            ax = axes[idx, f]
            data = flex[f]
            color = "#d62728" if f == 2 else "#1f77b4"
            ax.hist(data, bins=80, color=color, alpha=0.7, edgecolor="none")
            ax.set_title(f"S{subject_id} - {FINGER_NAMES[f]}", fontsize=10)
            if idx == 2:
                ax.set_xlabel("Flexion value")
            if f == 0:
                ax.set_ylabel("Count")
            # Add stats annotation
            var = np.var(data)
            pct_zero = np.mean(np.abs(data) < 0.5) * 100
            ax.text(0.95, 0.95, f"var={var:.1f}\n{pct_zero:.0f}% ~0",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "finger_histograms.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/finger_histograms.png")


def plot_summary_table(all_stats, output_dir):
    """Create a summary table figure: variance, SNR, % near zero, and test r."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Per-Finger Difficulty Analysis Summary", fontsize=14, y=0.98)

    x = np.arange(5)
    width = 0.25
    colors_subj = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green for S1, S2, S3

    # Panel 1: Variance
    ax = axes[0, 0]
    for idx, subject_id in enumerate(SUBJECTS):
        vals = all_stats[subject_id]["test"]["variance"]
        ax.bar(x + idx * width, vals, width, label=f"S{subject_id}", color=colors_subj[idx])
    ax.set_ylabel("Variance")
    ax.set_title("Per-Finger Variance")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FINGER_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: SNR
    ax = axes[0, 1]
    for idx, subject_id in enumerate(SUBJECTS):
        vals = all_stats[subject_id]["test"]["snr_db"]
        ax.bar(x + idx * width, vals, width, label=f"S{subject_id}", color=colors_subj[idx])
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Per-Finger Signal-to-Noise Ratio")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FINGER_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: % near zero
    ax = axes[1, 0]
    for idx, subject_id in enumerate(SUBJECTS):
        vals = all_stats[subject_id]["test"]["pct_near_zero"] * 100
        ax.bar(x + idx * width, vals, width, label=f"S{subject_id}", color=colors_subj[idx])
    ax.set_ylabel("% time")
    ax.set_title("% Time Near Zero (|flex| < 0.5)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FINGER_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Test r (from reported results)
    ax = axes[1, 1]
    for idx, subject_id in enumerate(SUBJECTS):
        vals = REPORTED_TEST_R[subject_id]
        ax.bar(x + idx * width, vals, width, label=f"S{subject_id}", color=colors_subj[idx])
    ax.set_ylabel("Pearson r")
    ax.set_title("Per-Finger Test Correlation (Channel Dropout Model)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(FINGER_NAMES)
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="r=0.5")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_panels.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/summary_panels.png")


def plot_variance_vs_test_r(all_stats, output_dir):
    """Scatter plot: per-finger variance vs test r, across all subjects."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Finger Variance vs Decoding Accuracy", fontsize=14)

    markers = ["o", "s", "^"]
    colors = plt.cm.tab10(np.arange(5))

    for idx, subject_id in enumerate(SUBJECTS):
        variances = all_stats[subject_id]["test"]["variance"]
        test_r = REPORTED_TEST_R[subject_id]
        for f in range(5):
            label = FINGER_NAMES[f] if idx == 0 else None
            ax.scatter(variances[f], test_r[f],
                       c=[colors[f]], marker=markers[idx], s=100, edgecolors="black",
                       label=label, zorder=5)
            ax.annotate(f"S{subject_id}", (variances[f], test_r[f]),
                        fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points")

    # Fit trend line across all points
    all_var = []
    all_r = []
    for subject_id in SUBJECTS:
        all_var.extend(all_stats[subject_id]["test"]["variance"])
        all_r.extend(REPORTED_TEST_R[subject_id])
    all_var = np.array(all_var)
    all_r = np.array(all_r)

    # Linear regression
    slope, intercept, r_val, p_val, _ = scipy_stats.linregress(all_var, all_r)
    x_fit = np.linspace(all_var.min(), all_var.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, "k--", alpha=0.5,
            label=f"Trend: r={r_val:.2f}, p={p_val:.3f}")

    ax.set_xlabel("Finger Flexion Variance (test set)")
    ax.set_ylabel("Test Pearson r")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add marker legend for subjects
    for idx, subject_id in enumerate(SUBJECTS):
        ax.scatter([], [], c="gray", marker=markers[idx], s=60,
                   label=f"Subject {subject_id}")
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "variance_vs_test_r.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/variance_vs_test_r.png")


def plot_predictions_vs_actual(subject_id, pred, test_flex, ds_fs, output_dir):
    """Plot predicted vs actual for worst (Middle) and best finger.

    Parameters
    ----------
    pred : np.ndarray, shape (5, time) — model predictions (MinMax scaled)
    test_flex : np.ndarray, shape (5, time) — ground truth (MinMax scaled)
    ds_fs : int — sampling rate (100 Hz)
    """
    test_r = REPORTED_TEST_R[subject_id]
    worst_idx = np.argmin(test_r)
    best_idx = np.argmax(test_r)

    # Align lengths
    T = min(pred.shape[1], test_flex.shape[1])
    pred = pred[:, :T]
    test_flex = test_flex[:, :T]
    t = np.arange(T) / ds_fs  # time in seconds

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"Subject {subject_id} — Predicted vs Actual (MinMax-Scaled)",
        fontsize=14
    )

    # Best finger
    ax = axes[0]
    ax.plot(t, test_flex[best_idx], label="Actual", color="#1f77b4", alpha=0.8, linewidth=1)
    ax.plot(t, pred[best_idx], label="Predicted", color="#ff7f0e", alpha=0.8, linewidth=1)
    ax.set_title(
        f"BEST: {FINGER_NAMES[best_idx]} (r = {test_r[best_idx]:.3f})",
        fontsize=12, color="green"
    )
    ax.set_ylabel("Flexion (scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Worst finger
    ax = axes[1]
    ax.plot(t, test_flex[worst_idx], label="Actual", color="#1f77b4", alpha=0.8, linewidth=1)
    ax.plot(t, pred[worst_idx], label="Predicted", color="#ff7f0e", alpha=0.8, linewidth=1)
    ax.set_title(
        f"WORST: {FINGER_NAMES[worst_idx]} (r = {test_r[worst_idx]:.3f})",
        fontsize=12, color="red"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flexion (scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"pred_vs_actual_S{subject_id}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")

    # Also plot a zoomed-in version (first 30 seconds)
    t_zoom = 30.0  # seconds
    n_zoom = min(int(t_zoom * ds_fs), T)
    t_z = t[:n_zoom]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle(
        f"Subject {subject_id} — First {t_zoom:.0f}s Zoom (MinMax-Scaled)",
        fontsize=14
    )

    ax = axes[0]
    ax.plot(t_z, test_flex[best_idx, :n_zoom], label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(t_z, pred[best_idx, :n_zoom], label="Predicted", color="#ff7f0e", linewidth=1.5)
    ax.set_title(
        f"BEST: {FINGER_NAMES[best_idx]} (r = {test_r[best_idx]:.3f})",
        fontsize=12, color="green"
    )
    ax.set_ylabel("Flexion (scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_z, test_flex[worst_idx, :n_zoom], label="Actual", color="#1f77b4", linewidth=1.5)
    ax.plot(t_z, pred[worst_idx, :n_zoom], label="Predicted", color="#ff7f0e", linewidth=1.5)
    ax.set_title(
        f"WORST: {FINGER_NAMES[worst_idx]} (r = {test_r[worst_idx]:.3f})",
        fontsize=12, color="red"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Flexion (scaled)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"pred_vs_actual_S{subject_id}_zoom.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


def plot_all_fingers_predictions(subject_id, pred, test_flex, ds_fs, output_dir):
    """Plot predicted vs actual for ALL 5 fingers, stacked."""
    test_r = REPORTED_TEST_R[subject_id]
    T = min(pred.shape[1], test_flex.shape[1])
    pred = pred[:, :T]
    test_flex = test_flex[:, :T]
    t = np.arange(T) / ds_fs

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"Subject {subject_id} — All Fingers: Predicted vs Actual", fontsize=14)

    for f in range(5):
        ax = axes[f]
        ax.plot(t, test_flex[f], label="Actual", color="#1f77b4", alpha=0.8, linewidth=0.8)
        ax.plot(t, pred[f], label="Predicted", color="#ff7f0e", alpha=0.8, linewidth=0.8)
        title_color = "red" if test_r[f] < 0.5 else ("green" if test_r[f] > 0.7 else "black")
        ax.set_title(f"{FINGER_NAMES[f]} (r = {test_r[f]:.3f})", fontsize=11, color=title_color)
        ax.set_ylabel("Flexion")
        if f == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    fname = f"all_fingers_S{subject_id}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_dir}/{fname}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  FINGER DECODING DIFFICULTY ANALYSIS — BCI-IV Competition Dataset")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load config (same for all subjects, just change subject ID)
    config_path = "configs/bci4_transformer_d1024_L6_h16_chdrop.yaml"
    if not os.path.isfile(config_path):
        print(f"WARNING: config {config_path} not found, trying from results dir")
        config_path = os.path.join(RESULT_DIRS[1], "config.yaml")
    config = load_config(config_path)

    set_seed(config.get("seed", 42))

    # ── Part 1: Raw target statistics ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 1: Raw Finger Flexion Target Statistics")
    print("=" * 80)

    all_raw = {}
    all_stats = {}

    for subject_id in SUBJECTS:
        print(f"\nLoading raw data for Subject {subject_id}...")
        raw_data = load_raw_finger_targets(subject_id)
        all_raw[subject_id] = raw_data

        # Compute stats on raw test targets
        train_stats = compute_finger_stats(raw_data["train_flex"], label=f"S{subject_id} train")
        test_stats = compute_finger_stats(raw_data["test_flex"], label=f"S{subject_id} test")
        all_stats[subject_id] = {"train": train_stats, "test": test_stats}

        print_finger_stats(test_stats, subject_id, "Test Set (Raw)",
                           test_r=REPORTED_TEST_R[subject_id])

    # ── Part 2: Correlation analysis ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 2: Inter-Finger Correlation Analysis")
    print("=" * 80)

    for subject_id in SUBJECTS:
        corr = all_stats[subject_id]["test"]["cross_corr"]
        print(f"\nSubject {subject_id} — Average off-diagonal |correlation| per finger:")
        for f in range(5):
            off_diag = [abs(corr[f, j]) for j in range(5) if j != f]
            avg_corr = np.mean(off_diag)
            print(f"  {FINGER_NAMES[f]:>8}: avg |r| with others = {avg_corr:.3f}")

        # Find the most correlated pair
        upper_tri = np.triu_indices(5, k=1)
        corr_values = corr[upper_tri]
        pair_idx = np.argmax(np.abs(corr_values))
        i, j = upper_tri[0][pair_idx], upper_tri[1][pair_idx]
        print(f"  Most correlated pair: {FINGER_NAMES[i]}-{FINGER_NAMES[j]} "
              f"(r = {corr[i, j]:.3f})")

    # ── Part 3: Key findings summary ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 3: Key Findings")
    print("=" * 80)

    print("\n  Variance ranking (test set, low = hard to decode):")
    for subject_id in SUBJECTS:
        variances = all_stats[subject_id]["test"]["variance"]
        ranking = np.argsort(variances)  # ascending = hardest first
        rank_str = " < ".join(
            f"{FINGER_NAMES[i]}({variances[i]:.1f})" for i in ranking
        )
        print(f"  S{subject_id}: {rank_str}")

    print("\n  % time near zero ranking (test set, high = hard):")
    for subject_id in SUBJECTS:
        pct = all_stats[subject_id]["test"]["pct_near_zero"] * 100
        ranking = np.argsort(-pct)  # descending = hardest first
        rank_str = " > ".join(
            f"{FINGER_NAMES[i]}({pct[i]:.0f}%)" for i in ranking
        )
        print(f"  S{subject_id}: {rank_str}")

    print("\n  SNR ranking (test set, low SNR = hard):")
    for subject_id in SUBJECTS:
        snr = all_stats[subject_id]["test"]["snr_db"]
        ranking = np.argsort(snr)  # ascending = hardest first
        rank_str = " < ".join(
            f"{FINGER_NAMES[i]}({snr[i]:.1f}dB)" for i in ranking
        )
        print(f"  S{subject_id}: {rank_str}")

    # ── Part 4: Generate plots (target statistics) ────────────────────
    print("\n" + "=" * 80)
    print("  PART 4: Generating Plots")
    print("=" * 80)

    print("\nPlotting target statistics...")
    plot_finger_variance_comparison(all_stats, OUTPUT_DIR)
    plot_cross_correlation_matrices(all_stats, OUTPUT_DIR)
    plot_finger_time_series(all_raw, OUTPUT_DIR)
    plot_finger_histograms(all_raw, OUTPUT_DIR)
    plot_summary_table(all_stats, OUTPUT_DIR)
    plot_variance_vs_test_r(all_stats, OUTPUT_DIR)

    # ── Part 5: Model predictions analysis ────────────────────────────
    print("\n" + "=" * 80)
    print("  PART 5: Model Predictions Analysis")
    print("=" * 80)

    for subject_id in SUBJECTS:
        print(f"\nProcessing Subject {subject_id}...")

        # Override subject in config
        config_s = load_config(config_path)
        config_s["data"]["subject"] = subject_id

        print(f"  Building preprocessed test data...")
        processed = load_processed_test_data(subject_id, config_s)

        print(f"  Loading model and running inference...")
        pred = load_model_and_predict(
            subject_id,
            processed["test_spec"],
            config_s,
            device,
        )

        if pred is not None:
            # Verify our predictions match the reported numbers
            T = min(pred.shape[1], processed["test_flex"].shape[1])
            pred_aligned = pred[:, :T]
            target_aligned = processed["test_flex"][:, :T]
            computed_r = pearson_r_per_channel(pred_aligned, target_aligned)
            print(f"  Computed test r: {' '.join(f'{r:.3f}' for r in computed_r)}")
            print(f"  Reported test r: {' '.join(f'{r:.3f}' for r in REPORTED_TEST_R[subject_id])}")

            # Generate prediction plots
            plot_predictions_vs_actual(
                subject_id, pred, processed["test_flex"],
                processed["ds_fs"], OUTPUT_DIR
            )
            plot_all_fingers_predictions(
                subject_id, pred, processed["test_flex"],
                processed["ds_fs"], OUTPUT_DIR
            )
        else:
            print(f"  Skipping prediction plots (no checkpoint)")

    # ── Done ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"  DONE — All results saved to {OUTPUT_DIR}/")
    print("=" * 80)
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
