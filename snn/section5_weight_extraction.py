"""
section5_weight_extraction.py
------------------------------
EE 533 Final Project - Section 5: Partial Hardware Validation (4-Neuron Circuit)

This script automates Steps 2-5 of the Section 5 workflow:

    Step 2 — Verify neurons are distinct after training
    Step 3 — Pick the 4 most active neurons from the hidden layer
    Step 4 — Extract their weights from the trained checkpoint
    Step 5 — Generate input current waveforms in PWLF format for Cadence

Run from the project root AFTER section4_train_mnist_circuit_snn.py:

    python -m snn.section5_weight_extraction

Or with custom paths:

    python -m snn.section5_weight_extraction \\
        --checkpoint  validation/section4_outputs/section4_circuit_snn_best.pt \\
        --data-dir    ./data \\
        --output-dir  validation/section5_outputs \\
        --num-images  10 \\
        --seed        42

Outputs (all written to --output-dir):
    section5_neuron_weights.pt          PyTorch file with weights of the 4 neurons
    section5_neuron_weights.csv         Human-readable weight table (one row per neuron)
    section5_weight_health.png          Plot confirming neurons are distinct
    section5_neuron<N>_image<M>_pwlf.csv  Cadence PWLF current waveform per neuron per image
    section5_pwlf_preview.png           Preview plot of all 4 waveforms for image 0
    section5_summary.txt                Run summary with all key numbers
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

# ── import your model and neuron ──────────────────────────────────────────────
# Adjust the import path if your file lives elsewhere.
from snn.section4_train_mnist_circuit_snn import CircuitInformedMNISTSNN


# =============================================================================
# Step 2 — Verify that hidden-layer neurons are distinct
# =============================================================================

def verify_weight_diversity(
    weights: torch.Tensor,          # shape [hidden_size, input_size]
    out_dir: Path,
    threshold_std: float = 1e-4,
) -> None:
    """
    Print and plot statistics that confirm neurons have differentiated
    during training.  Raises RuntimeError if neurons are suspiciously
    uniform (likely due to the dead-gradient bug that was already fixed).
    """
    print("\n" + "=" * 60)
    print("STEP 2 — Weight diversity check")
    print("=" * 60)

    # Per-neuron stats
    row_means = weights.mean(dim=1)   # [hidden]
    row_stds  = weights.std(dim=1)    # [hidden]

    overall_std = row_means.std().item()
    print(f"  Hidden neurons        : {weights.shape[0]}")
    print(f"  Input features        : {weights.shape[1]}")
    print(f"  Mean of row-means     : {row_means.mean().item():.6f}")
    print(f"  Std  of row-means     : {overall_std:.6f}  (want >> {threshold_std})")
    print(f"  Min  row std          : {row_stds.min().item():.6f}")
    print(f"  Max  row std          : {row_stds.max().item():.6f}")

    if overall_std < threshold_std:
        raise RuntimeError(
            f"Weight diversity is too low (std of row-means = {overall_std:.2e}). "
            "Neurons appear identical — re-train with the fixed section4 script."
        )
    print("  OK Neurons are distinct.\n")

    # Plot histogram of per-neuron mean weights
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(row_means.numpy(), bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of per-neuron weight means")
    axes[0].set_xlabel("Weight mean")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(row_stds.numpy(), bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("Distribution of per-neuron weight stds")
    axes[1].set_xlabel("Weight std")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Step 2: Hidden-layer weight diversity", fontweight="bold")
    fig.tight_layout()
    path = out_dir / "section5_weight_health.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved weight-health plot -> {path}")


# =============================================================================
# Step 3 — Pick the 4 most active neurons
# =============================================================================

def pick_active_neurons(
    model: CircuitInformedMNISTSNN,
    images: torch.Tensor,           # [batch, 1, H, W]
    time_steps: int,
    n_select: int = 4,
    seed: int = 42,
) -> list[int]:
    """
    Run a forward pass and return the indices of the n_select neurons
    with the highest average spike count across the batch.
    Choosing active neurons ensures the Cadence waveforms will actually
    produce spikes — dead neurons produce flat zero current.
    """
    print("=" * 60)
    print("STEP 3 — Selecting 4 most active hidden neurons")
    print("=" * 60)

    model.eval()
    functional.reset_net(model)

    x = images.flatten(start_dim=1)                     # [batch, input_size]
    spike_counts = torch.zeros(model.hidden_size)

    torch.manual_seed(seed)
    with torch.no_grad():
        for t in range(time_steps):
            spikes_in    = model._poisson_encode(x)
            curr         = torch.sigmoid(model.fc1(spikes_in))
            hidden_spikes = model.hidden_neuron(curr)
            spike_counts += hidden_spikes.sum(dim=0).cpu()

    functional.reset_net(model)

    # Sort descending; take top n_select
    sorted_indices = spike_counts.argsort(descending=True).tolist()
    selected = sorted_indices[:n_select]

    print(f"  Evaluated {images.shape[0]} images x {time_steps} time steps")
    print(f"\n  Top {n_select} active neurons:")
    for rank, idx in enumerate(selected):
        print(f"    Rank {rank+1}: neuron {idx:4d}  "
              f"total spikes = {int(spike_counts[idx])}")

    print(f"\n  Selected indices: {selected}\n")
    return selected


# =============================================================================
# Step 4 — Extract and save weights
# =============================================================================

def extract_and_save_weights(
    checkpoint: dict,
    neuron_indices: list[int],
    out_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull fc1 weights and biases for the selected neurons,
    save as both a .pt file and a human-readable CSV.

    Returns (fc1_weights [4, input_size], fc1_bias [4]).
    """
    print("=" * 60)
    print("STEP 4 — Extracting weights for selected neurons")
    print("=" * 60)

    state = checkpoint["model_state_dict"]
    fc1_w_all = state["fc1.weight"]    # [hidden_size, input_size]
    fc1_b_all = state["fc1.bias"]      # [hidden_size]

    w4 = fc1_w_all[neuron_indices]     # [4, input_size]
    b4 = fc1_b_all[neuron_indices]     # [4]

    for local_i, global_i in enumerate(neuron_indices):
        print(f"  Neuron {global_i:4d}:  "
              f"weight mean={w4[local_i].mean().item():+.5f}  "
              f"weight std={w4[local_i].std().item():.5f}  "
              f"bias={b4[local_i].item():+.5f}")

    # ── .pt file ─────────────────────────────────────────────────────────────
    pt_path = out_dir / "section5_neuron_weights.pt"
    torch.save({
        "neuron_indices" : neuron_indices,
        "fc1_weights"    : w4,
        "fc1_bias"       : b4,
        "checkpoint_args": checkpoint.get("args", {}),
    }, pt_path)
    print(f"\n  Saved weights (PyTorch) -> {pt_path}")

    # ── CSV (one row per neuron, columns = weight_0 … weight_N-1, bias) ──────
    import csv
    csv_path = out_dir / "section5_neuron_weights.csv"
    n_inputs = w4.shape[1]
    with open(csv_path, "w", newline="") as f:
        header = ["neuron_index"] + [f"w{i}" for i in range(n_inputs)] + ["bias"]
        writer = csv.writer(f)
        writer.writerow(header)
        for local_i, global_i in enumerate(neuron_indices):
            row = [global_i] + w4[local_i].tolist() + [b4[local_i].item()]
            writer.writerow(row)
    print(f"  Saved weights (CSV)    -> {csv_path}\n")

    return w4, b4


# =============================================================================
# Step 5 — Generate PWLF current waveforms for Cadence
# =============================================================================

def generate_pwlf(
    w4: torch.Tensor,               # [4, input_size]
    b4: torch.Tensor,               # [4]
    neuron_indices: list[int],
    images: torch.Tensor,           # [num_images, 1, H, W]
    labels: list[int],
    time_steps: int,
    dt_ms: float,
    I_max_nA: float,
    out_dir: Path,
    seed: int = 42,
) -> None:
    """
    For every (neuron, image) pair write a PWLF CSV with columns:
        time_s   — absolute time in seconds  (Cadence native unit)
        current_A — current in amperes        (Cadence native unit)
        time_ms  — same in ms  (for readability)
        current_nA — same in nA (for readability)

    The PWLF step function holds each current level for exactly dt_ms,
    matching the time-step resolution of the Python SNN.
    """
    print("=" * 60)
    print("STEP 5 — Generating Cadence PWLF waveforms")
    print("=" * 60)
    print(f"  Neurons   : {neuron_indices}")
    print(f"  Images    : {len(images)}")
    print(f"  Time steps: {time_steps}")
    print(f"  dt        : {dt_ms} ms")
    print(f"  I_max     : {I_max_nA} nA\n")

    all_waveforms = {}   # key = neuron_index, value = list of I_nA per step (image 0 only, for plot)

    for img_idx, (img_tensor, label) in enumerate(zip(images, labels)):
        img_flat = img_tensor.flatten()    # [input_size], values in [0,1]

        for local_i, global_i in enumerate(neuron_indices):
            w = w4[local_i]    # [input_size]
            b = b4[local_i]    # scalar

            times_ms    = []
            currents_nA = []

            for t in range(time_steps):
                # Reproducible Bernoulli encoding — same seed as training eval
                torch.manual_seed(seed + img_idx * 1000 + t)
                spikes = (torch.rand_like(img_flat) <= img_flat).float()

                # Weighted sum -> sigmoid -> physical current
                weighted = (w * spikes).sum() + b
                I_nA = torch.sigmoid(weighted).item() * I_max_nA

                # PWLF step: current is constant over [t*dt, (t+1)*dt]
                t_start_ms = t * dt_ms
                t_end_ms   = (t + 1) * dt_ms

                # Each step needs a start point and end point at same level.
                # For a clean step-function add a tiny ramp at transitions.
                if t == 0:
                    times_ms.append(0.0)
                    currents_nA.append(I_nA)
                else:
                    # Step down/up at the exact transition point
                    times_ms.append(t_start_ms)
                    currents_nA.append(currents_nA[-1])   # hold previous value until transition
                    times_ms.append(t_start_ms)
                    currents_nA.append(I_nA)              # jump to new value

                times_ms.append(t_end_ms)
                currents_nA.append(I_nA)

            # Store image-0 waveforms for the preview plot
            if img_idx == 0:
                all_waveforms[global_i] = (list(times_ms), list(currents_nA))

            # ── Write CSV ────────────────────────────────────────────────────
            fname = (out_dir /
                     f"section5_neuron{global_i}_image{img_idx}"
                     f"_label{label}_pwlf.csv")
            with open(fname, "w") as f:
                f.write("time_s,current_A,time_ms,current_nA\n")
                for t_ms, I_nA_val in zip(times_ms, currents_nA):
                    f.write(
                        f"{t_ms * 1e-3:.9f},"
                        f"{I_nA_val * 1e-9:.6e},"
                        f"{t_ms:.6f},"
                        f"{I_nA_val:.6f}\n"
                    )

        if img_idx % max(1, len(images) // 5) == 0:
            print(f"  Wrote PWLF for image {img_idx}/{len(images)-1}  (label={label})")

    print(f"\n  OK All PWLF files written to {out_dir}")

    # ── Preview plot (image 0 only) ──────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for ax, (global_i, color) in zip(axes, zip(neuron_indices, colors)):
        t_ms, I_nA = all_waveforms[global_i]
        ax.step(t_ms, I_nA, where="post", color=color, linewidth=1.5,
                label=f"Neuron {global_i}")
        ax.set_ylabel("I (nA)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time (ms)")
    fig.suptitle(
        f"Step 5: PWLF input currents — Image 0  "
        f"(label={int(labels[0])}, {time_steps} steps x {dt_ms} ms)",
        fontweight="bold",
    )
    fig.tight_layout()
    plot_path = out_dir / "section5_pwlf_preview.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"  Saved PWLF preview plot -> {plot_path}\n")


# =============================================================================
# Summary report
# =============================================================================

def write_summary(
    out_dir: Path,
    neuron_indices: list[int],
    checkpoint_args: dict,
    n_images: int,
    time_steps: int,
    dt_ms: float,
    I_max_nA: float,
    test_accuracy: float,
) -> None:
    path = out_dir / "section5_summary.txt"
    total_time_ms = time_steps * dt_ms
    lines = [
        "=" * 60,
        "EE 533  Section 5 — Weight Extraction Summary",
        "=" * 60,
        "",
        "Trained model info:",
        f"  Checkpoint test accuracy : {test_accuracy:.4f}",
        f"  Hidden size              : {checkpoint_args.get('hidden_size', '?')}",
        f"  Input resolution         : {checkpoint_args.get('input_resolution', '?')}",
        f"  Time steps               : {time_steps}",
        f"  dt                       : {dt_ms} ms",
        f"  I_max                    : {I_max_nA} nA",
        "",
        "Selected neurons (4 most active):",
    ] + [f"  {i+1}. neuron index {idx}" for i, idx in enumerate(neuron_indices)] + [
        "",
        "PWLF waveform parameters:",
        f"  Simulation time per image: {total_time_ms:.1f} ms  ({time_steps} x {dt_ms} ms)",
        f"  Current range            : 0 - {I_max_nA} nA",
        f"  Number of test images    : {n_images}",
        "",
        "Files written:",
        "  section5_neuron_weights.pt",
        "  section5_neuron_weights.csv",
        "  section5_weight_health.png",
        "  section5_pwlf_preview.png",
    ] + [
        f"  section5_neuron<N>_image<M>_label<L>_pwlf.csv  x {4 * n_images} files"
    ] + [
        "",
        "Next step -> Cadence:",
        "  1. Open each *_pwlf.csv and use time_s / current_A columns.",
        "  2. Create a PWLF current source for each of the 4 neuron instances.",
        "  3. Run transient for {:.1f} ms.".format(total_time_ms),
        "  4. Export /vout for each neuron.",
        "  5. Use section3 compare_spikes() to compute timing agreement.",
        "=" * 60,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved summary -> {path}")


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Section 5: extract weights + generate Cadence PWLF waveforms."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("validation/section4_outputs/section4_circuit_snn_best.pt"),
        help="Path to the .pt checkpoint from section4_train_mnist_circuit_snn.py",
    )
    p.add_argument("--data-dir",   type=Path, default=Path("./data"))
    p.add_argument("--output-dir", type=Path, default=Path("validation/section5_outputs"))
    p.add_argument(
        "--num-images", type=int, default=10,
        help="Number of MNIST test images to generate PWLF files for.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run section4_train_mnist_circuit_snn.py first."
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args  = checkpoint.get("args", {})
    test_acc   = checkpoint.get("test_accuracy", float("nan"))

    hidden_size      = ckpt_args.get("hidden_size",      128)
    input_resolution = ckpt_args.get("input_resolution", 28)
    time_steps       = ckpt_args.get("time_steps",       10)
    dt_ms            = ckpt_args.get("dt_ms",            1.0)
    I_max_nA         = ckpt_args.get("i_max_nA",         100.0)
    input_size       = input_resolution * input_resolution

    print(f"  test_accuracy    : {test_acc:.4f}")
    print(f"  hidden_size      : {hidden_size}")
    print(f"  input_resolution : {input_resolution}")
    print(f"  time_steps       : {time_steps}")
    print(f"  dt_ms            : {dt_ms}")
    print(f"  I_max_nA         : {I_max_nA}")

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = CircuitInformedMNISTSNN(
        input_size=input_size,
        hidden_size=hidden_size,
        time_steps=time_steps,
        dt_ms=dt_ms,
        i_max_nA=I_max_nA,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Load test images ──────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((input_resolution, input_resolution)),
        transforms.ToTensor(),
    ])
    test_set = datasets.MNIST(
        root=str(args.data_dir), train=False, transform=transform, download=True
    )
    loader = DataLoader(test_set, batch_size=args.num_images, shuffle=False)
    images, labels = next(iter(loader))
    images = images[:args.num_images]
    labels = labels[:args.num_images].tolist()

    # ── Step 2: verify diversity ──────────────────────────────────────────────
    fc1_w_all = checkpoint["model_state_dict"]["fc1.weight"]
    verify_weight_diversity(fc1_w_all, args.output_dir)

    # ── Step 3: pick 4 neurons ────────────────────────────────────────────────
    neuron_indices = pick_active_neurons(
        model, images, time_steps, n_select=4, seed=args.seed
    )

    # ── Step 4: extract weights ───────────────────────────────────────────────
    w4, b4 = extract_and_save_weights(checkpoint, neuron_indices, args.output_dir)

    # ── Step 5: generate PWLF ────────────────────────────────────────────────
    generate_pwlf(
        w4=w4,
        b4=b4,
        neuron_indices=neuron_indices,
        images=images,
        labels=labels,
        time_steps=time_steps,
        dt_ms=dt_ms,
        I_max_nA=I_max_nA,
        out_dir=args.output_dir,
        seed=args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary(
        out_dir=args.output_dir,
        neuron_indices=neuron_indices,
        checkpoint_args=ckpt_args,
        n_images=args.num_images,
        time_steps=time_steps,
        dt_ms=dt_ms,
        I_max_nA=I_max_nA,
        test_accuracy=test_acc,
    )


if __name__ == "__main__":
    main()