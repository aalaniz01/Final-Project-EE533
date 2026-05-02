"""
section5_compare_spikes.py
--------------------------
EE 533 Final Project - Section 5c: Spike Timing and Temporal Alignment

This script compares Cadence circuit spikes vs Python SNN spikes for the
4 selected hidden-layer neurons.

What it does:
    1. Loads your 4 Cadence /vout CSV exports (one per neuron)
    2. Extracts spike times from each Cadence waveform
    3. Runs the trained Python SNN on the same image to get Python spike times
    4. Computes spike timing metrics (MAE, RMSE, max error, rate error)
    5. Saves report-ready plots and a metrics CSV

Place this file in your snn/ folder:
    snn/section5_compare_spikes.py

Run from project root AFTER you have exported /vout CSVs from Cadence:
    python -m snn.section5_compare_spikes

Or with explicit paths:
    python -m snn.section5_compare_spikes `
        --checkpoint  validation/section4_outputs/section4_circuit_snn_best.pt `
        --weights-pt  validation/section5_outputs/section5_neuron_weights.pt `
        --cadence-dir validation/section5_outputs/cadence_vout `
        --image-index 0 `
        --output-dir  validation/section5_outputs

Expected Cadence CSV format (one file per neuron):
    Two columns: time and voltage
    Example filename: neuron104_vout.csv
    The script will look for files matching: neuron<ID>_vout.csv

Outputs written to --output-dir:
    section5c_metrics.csv               Spike timing metrics table
    section5c_raster.png                Raster plot (Cadence vs Python)
    section5c_spike_agreement.png       Spike time scatter plot
    section5c_voltage_traces.png        Cadence vout waveforms with spike markers
    section5c_summary.txt               Plain-text summary for your report
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from snn.section4_train_mnist_circuit_snn import CircuitInformedMNISTSNN


# =============================================================================
# Spike extraction from Cadence vout
# =============================================================================

def load_cadence_vout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a Cadence /vout CSV.
    Expects two columns: time (seconds) and voltage (V).
    Handles optional headers automatically.
    """
    df = pd.read_csv(path, header=None)

    # If first row looks like a header, skip it
    try:
        float(df.iloc[0, 0])
    except (ValueError, TypeError):
        df = pd.read_csv(path, header=0)

    time_s = df.iloc[:, 0].to_numpy(dtype=float)
    vout_V = df.iloc[:, 1].to_numpy(dtype=float)
    return time_s, vout_V


def extract_spike_times(
    time_s: np.ndarray,
    vout_V: np.ndarray,
    threshold_V: float = 0.5,
    refractory_s: float = 0.5e-3,
) -> np.ndarray:
    """
    Extract spike times from a Cadence vout waveform.

    A spike is detected when vout crosses upward through threshold_V.
    refractory_s prevents double-counting the same spike.

    Returns array of spike times in seconds.
    """
    above = vout_V >= threshold_V
    # Rising edges: False -> True transitions
    rising = above & np.concatenate([[False], ~above[:-1]])
    candidate_indices = np.where(rising)[0]

    spike_times = []
    for idx in candidate_indices:
        t = float(time_s[idx])
        if not spike_times or (t - spike_times[-1]) > refractory_s:
            spike_times.append(t)

    return np.array(spike_times)


# =============================================================================
# Python SNN spike extraction
# =============================================================================

def get_python_spikes(
    model: CircuitInformedMNISTSNN,
    image: torch.Tensor,            # [1, H, W]
    neuron_indices: list[int],
    time_steps: int,
    dt_ms: float,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """
    Run the Python SNN on one image and collect hidden-layer spike times
    for the 4 selected neurons.

    Returns a dict: {neuron_index: array of spike times in seconds}
    """
    model.eval()
    functional.reset_net(model)

    x = image.flatten().unsqueeze(0)   # [1, input_size]
    spike_times_per_neuron = {idx: [] for idx in neuron_indices}

    torch.manual_seed(seed)
    with torch.no_grad():
        for t in range(time_steps):
            spikes_in     = model._poisson_encode(x)
            curr          = torch.sigmoid(model.fc1(spikes_in))
            hidden_spikes = model.hidden_neuron(curr)   # [1, hidden_size]

            # Record spike time (center of the time step) for active neurons
            t_center_s = (t + 0.5) * dt_ms * 1e-3

            for idx in neuron_indices:
                if hidden_spikes[0, idx].item() > 0.5:
                    spike_times_per_neuron[idx].append(t_center_s)

    functional.reset_net(model)

    return {idx: np.array(times) for idx, times in spike_times_per_neuron.items()}


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    cadence_spikes_s: np.ndarray,
    python_spikes_s: np.ndarray,
    neuron_idx: int,
    total_time_s: float,
) -> dict:
    """
    Compute spike timing and rate agreement metrics for one neuron.
    All times are converted to ms for readability.
    """
    n_cad = len(cadence_spikes_s)
    n_py  = len(python_spikes_s)
    n_matched = min(n_cad, n_py)

    if n_matched > 0:
        errors_s = python_spikes_s[:n_matched] - cadence_spikes_s[:n_matched]
        mae_ms   = float(np.mean(np.abs(errors_s)) * 1e3)
        rmse_ms  = float(np.sqrt(np.mean(errors_s ** 2)) * 1e3)
        max_ms   = float(np.max(np.abs(errors_s)) * 1e3)
    else:
        mae_ms = rmse_ms = max_ms = float("nan")

    cad_rate  = n_cad  / total_time_s if total_time_s > 0 else float("nan")
    py_rate   = n_py   / total_time_s if total_time_s > 0 else float("nan")
    rate_err  = (abs(py_rate - cad_rate) / cad_rate * 100.0
                 if cad_rate > 0 else float("nan"))

    # Temporal alignment: fraction of Python spikes within 1 ms of a Cadence spike
    aligned = 0
    if n_cad > 0 and n_py > 0:
        for pt in python_spikes_s:
            if np.min(np.abs(cadence_spikes_s - pt)) <= 1e-3:
                aligned += 1
        alignment_pct = aligned / n_py * 100.0
    else:
        alignment_pct = float("nan")

    return {
        "neuron_index"          : neuron_idx,
        "cadence_spikes"        : n_cad,
        "python_spikes"         : n_py,
        "matched_spikes"        : n_matched,
        "spike_time_MAE_ms"     : round(mae_ms,  4),
        "spike_time_RMSE_ms"    : round(rmse_ms, 4),
        "spike_time_max_err_ms" : round(max_ms,  4),
        "cadence_rate_Hz"       : round(cad_rate, 2),
        "python_rate_Hz"        : round(py_rate,  2),
        "rate_error_pct"        : round(rate_err, 2),
        "temporal_alignment_pct": round(alignment_pct, 2),
    }


# =============================================================================
# Plots
# =============================================================================

def plot_voltage_traces(
    cadence_data: dict[int, tuple],   # {neuron_idx: (time_s, vout_V)}
    cadence_spikes: dict[int, np.ndarray],
    python_spikes: dict[int, np.ndarray],
    threshold_V: float,
    out_path: Path,
) -> None:
    """
    Plot Cadence vout waveforms with Cadence and Python spike markers overlaid.
    This is the clearest way to show temporal alignment in your report.
    """
    neuron_indices = list(cadence_data.keys())
    n = len(neuron_indices)
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    fig, axes = plt.subplots(n, 1, figsize=(11, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, idx, color in zip(axes, neuron_indices, colors):
        time_s, vout_V = cadence_data[idx]
        time_ms = time_s * 1e3

        # Voltage trace
        ax.plot(time_ms, vout_V, color=color, linewidth=1.2,
                label="Cadence /vout", zorder=2)

        # Threshold line
        ax.axhline(threshold_V, color="gray", linestyle="--",
                   linewidth=0.8, label=f"Threshold ({threshold_V} V)")

        # Cadence spike markers
        cad_s = cadence_spikes[idx]
        if len(cad_s) > 0:
            ax.scatter(cad_s * 1e3,
                       np.ones(len(cad_s)) * threshold_V * 1.05,
                       marker="v", s=60, color="navy",
                       label=f"Cadence spikes ({len(cad_s)})", zorder=5)

        # Python spike markers
        py_s = python_spikes[idx]
        if len(py_s) > 0:
            ax.scatter(py_s * 1e3,
                       np.ones(len(py_s)) * threshold_V * 0.90,
                       marker="^", s=60, color="red",
                       label=f"Python spikes ({len(py_s)})", zorder=5)

        ax.set_ylabel("Voltage (V)", fontsize=9)
        ax.set_title(f"Neuron {idx}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle("Section 5c: Cadence vs Python - Voltage Traces with Spike Markers",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved voltage traces    -> {out_path}")


def plot_raster(
    cadence_spikes: dict[int, np.ndarray],
    python_spikes: dict[int, np.ndarray],
    total_time_ms: float,
    out_path: Path,
) -> None:
    """
    Raster plot: each row is one neuron, dots show when it spiked.
    Blue = Cadence, Red = Python.
    This is the standard way to show temporal alignment side by side.
    """
    neuron_indices = list(cadence_spikes.keys())
    n = len(neuron_indices)

    fig, ax = plt.subplots(figsize=(10, 4))

    for row, idx in enumerate(neuron_indices):
        cad_s = cadence_spikes[idx]
        py_s  = python_spikes[idx]

        if len(cad_s) > 0:
            ax.scatter(cad_s * 1e3,
                       np.full(len(cad_s), row + 0.15),
                       marker="|", s=200, linewidths=2.5,
                       color="steelblue", label="Cadence" if row == 0 else "")

        if len(py_s) > 0:
            ax.scatter(py_s * 1e3,
                       np.full(len(py_s), row - 0.15),
                       marker="|", s=200, linewidths=2.5,
                       color="crimson", label="Python" if row == 0 else "")

    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Neuron {idx}" for idx in neuron_indices])
    ax.set_xlabel("Time (ms)")
    ax.set_xlim(0, total_time_ms)
    ax.set_title("Section 5c: Spike Raster - Cadence vs Python",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved raster plot       -> {out_path}")


def plot_spike_agreement(
    cadence_spikes: dict[int, np.ndarray],
    python_spikes: dict[int, np.ndarray],
    out_path: Path,
) -> None:
    """
    Scatter plot: Cadence spike time vs Python spike time for each matched pair.
    Points on the diagonal = perfect agreement.
    """
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]
    neuron_indices = list(cadence_spikes.keys())

    fig, ax = plt.subplots(figsize=(6, 6))

    all_times = []

    for idx, color in zip(neuron_indices, colors):
        cad_s = cadence_spikes[idx]
        py_s  = python_spikes[idx]
        n = min(len(cad_s), len(py_s))

        if n > 0:
            cad_ms = cad_s[:n] * 1e3
            py_ms  = py_s[:n]  * 1e3
            ax.scatter(cad_ms, py_ms, color=color, s=60, zorder=4,
                       label=f"Neuron {idx} ({n} pairs)")
            all_times.extend(cad_ms.tolist())
            all_times.extend(py_ms.tolist())

    if all_times:
        lo = min(all_times) - 1
        hi = max(all_times) + 1
    else:
        lo, hi = 0, 10

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect agreement")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Cadence spike time (ms)")
    ax.set_ylabel("Python spike time (ms)")
    ax.set_title("Section 5c: Spike Time Agreement",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved agreement plot    -> {out_path}")


# =============================================================================
# Summary text
# =============================================================================

def write_summary(
    metrics_rows: list[dict],
    out_path: Path,
    image_label: int,
    total_time_ms: float,
) -> None:
    lines = [
        "=" * 62,
        "EE 533  Section 5c - Spike Timing and Temporal Alignment",
        "=" * 62,
        f"Test image label : {image_label}",
        f"Simulation window: {total_time_ms:.1f} ms",
        "",
        f"{'Neuron':<10} {'Cad':>5} {'Py':>5} {'MAE(ms)':>10} "
        f"{'RMSE(ms)':>10} {'MaxErr(ms)':>12} {'RateErr%':>10} {'Align%':>8}",
        "-" * 62,
    ]

    for r in metrics_rows:
        lines.append(
            f"{r['neuron_index']:<10} "
            f"{r['cadence_spikes']:>5} "
            f"{r['python_spikes']:>5} "
            f"{r['spike_time_MAE_ms']:>10.3f} "
            f"{r['spike_time_RMSE_ms']:>10.3f} "
            f"{r['spike_time_max_err_ms']:>12.3f} "
            f"{r['rate_error_pct']:>10.2f} "
            f"{r['temporal_alignment_pct']:>8.1f}"
        )

    lines += [
        "-" * 62,
        "",
        "Column legend:",
        "  Cad        = number of spikes detected in Cadence",
        "  Py         = number of spikes in Python SNN",
        "  MAE(ms)    = mean absolute spike time error",
        "  RMSE(ms)   = root mean square spike time error",
        "  MaxErr(ms) = worst-case spike time error",
        "  RateErr%   = firing rate disagreement percentage",
        "  Align%     = fraction of Python spikes within 1 ms of a Cadence spike",
        "",
        "Files produced:",
        "  section5c_metrics.csv",
        "  section5c_raster.png",
        "  section5c_spike_agreement.png",
        "  section5c_voltage_traces.png",
        "=" * 62,
    ]

    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nSaved summary -> {out_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Section 5c: compare Cadence vs Python spike timing."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("validation/section4_outputs/section4_circuit_snn_best.pt"),
    )
    p.add_argument(
        "--weights-pt",
        type=Path,
        default=Path("validation/section5_outputs/section5_neuron_weights.pt"),
        help="PT file produced by section5_weight_extraction.py",
    )
    p.add_argument(
        "--cadence-dir",
        type=Path,
        default=Path("validation/section5_outputs/cadence_vout"),
        help=(
            "Folder containing Cadence /vout CSV files.\n"
            "Expected filenames: neuron104_vout.csv, neuron48_vout.csv, etc."
        ),
    )
    p.add_argument("--data-dir",    type=Path, default=Path("./data"))
    p.add_argument("--output-dir",  type=Path, default=Path("validation/section5_outputs"))
    p.add_argument("--image-index", type=int,  default=0,
                   help="Which MNIST test image was used in Cadence (default: 0)")
    p.add_argument("--threshold",   type=float, default=0.5,
                   help="Voltage threshold for spike detection in Cadence vout (V)")
    p.add_argument("--seed",        type=int,  default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint : {args.checkpoint}")
    ckpt      = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    hidden_size      = ckpt_args.get("hidden_size",      128)
    input_resolution = ckpt_args.get("input_resolution",  28)
    time_steps       = ckpt_args.get("time_steps",        10)
    dt_ms            = ckpt_args.get("dt_ms",             1.0)
    I_max_nA         = ckpt_args.get("i_max_nA",        100.0)
    total_time_ms    = time_steps * dt_ms

    # ── Load selected neuron info ─────────────────────────────────────────────
    print(f"Loading weights    : {args.weights_pt}")
    w_data         = torch.load(args.weights_pt, map_location="cpu")
    neuron_indices = w_data["neuron_indices"]
    print(f"  Neuron indices   : {neuron_indices}")

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = CircuitInformedMNISTSNN(
        input_size=input_resolution ** 2,
        hidden_size=hidden_size,
        time_steps=time_steps,
        dt_ms=dt_ms,
        i_max_nA=I_max_nA,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Load the test image ───────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution)),
        transforms.ToTensor(),
    ])
    test_set = datasets.MNIST(
        root=str(args.data_dir), train=False,
        transform=transform, download=True,
    )
    image, label = test_set[args.image_index]
    print(f"Test image index   : {args.image_index}  (label = {label})")

    # ── Load Cadence vout CSVs ────────────────────────────────────────────────
    print(f"\nLooking for Cadence vout files in: {args.cadence_dir}")

    if not args.cadence_dir.exists():
        print(
            f"\n  WARNING: cadence_vout folder not found at {args.cadence_dir}\n"
            "  Creating it now with placeholder instructions.\n"
            "  Export your Cadence /vout waveforms as CSV files named:\n"
        )
        args.cadence_dir.mkdir(parents=True, exist_ok=True)
        for idx in neuron_indices:
            print(f"    neuron{idx}_vout.csv")
        print(
            "\n  Each CSV should have two columns: time(s) and voltage(V).\n"
            "  Then re-run this script.\n"
        )
        return

    cadence_data   = {}
    cadence_spikes = {}
    missing        = []

    for idx in neuron_indices:
        vout_path = args.cadence_dir / f"neuron{idx}_vout.csv"
        if not vout_path.exists():
            missing.append(str(vout_path))
            continue
        time_s, vout_V = load_cadence_vout(vout_path)
        spikes         = extract_spike_times(time_s, vout_V, args.threshold)
        cadence_data[idx]   = (time_s, vout_V)
        cadence_spikes[idx] = spikes
        print(f"  Neuron {idx:4d}: loaded {vout_path.name}  "
              f"-> {len(spikes)} spikes detected")

    if missing:
        print(
            f"\n  Missing Cadence files:\n"
            + "\n".join(f"    {m}" for m in missing)
            + "\n\n  Export these from Cadence and re-run."
        )
        if not cadence_data:
            return

    # ── Get Python spikes ─────────────────────────────────────────────────────
    print("\nRunning Python SNN on the same image...")
    python_spikes = get_python_spikes(
        model, image, list(cadence_data.keys()),
        time_steps, dt_ms, seed=args.seed,
    )
    for idx, spikes in python_spikes.items():
        print(f"  Neuron {idx:4d}: {len(spikes)} Python spikes  "
              f"at t = {[round(s*1e3, 2) for s in spikes]} ms")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics_rows = []
    for idx in cadence_data:
        row = compute_metrics(
            cadence_spikes[idx],
            python_spikes[idx],
            idx,
            total_time_ms * 1e-3,
        )
        metrics_rows.append(row)

    metrics_path = args.output_dir / "section5c_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"  Saved metrics CSV  -> {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_voltage_traces(
        cadence_data, cadence_spikes, python_spikes,
        threshold_V=args.threshold,
        out_path=args.output_dir / "section5c_voltage_traces.png",
    )
    plot_raster(
        cadence_spikes, python_spikes,
        total_time_ms=total_time_ms,
        out_path=args.output_dir / "section5c_raster.png",
    )
    plot_spike_agreement(
        cadence_spikes, python_spikes,
        out_path=args.output_dir / "section5c_spike_agreement.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(
        metrics_rows,
        out_path=args.output_dir / "section5c_summary.txt",
        image_label=int(label),
        total_time_ms=total_time_ms,
    )


if __name__ == "__main__":
    main()