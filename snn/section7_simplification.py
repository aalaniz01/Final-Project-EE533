"""
section7_simplification.py
---------------------------
EE 533 Final Project - Section 7: Network Simplification

Goal: Reduce model complexity while maintaining >75% accuracy.
Sweeps reductions in:
    - Hidden layer size  : 128, 64, 32
    - Time steps         : 20, 10, 5
    - Input resolution   : 28x28, 14x14, 7x7
    - Weight quantization: none, 8-bit, 4-bit, 2-bit (post-training)

Uses the exact same CircuitInformedMNISTSNN from Section 4.
Saves results after every run so it can resume if interrupted.

Place in snn/ folder:
    snn/section7_simplification.py

Run from project root:
    python -m snn.section7_simplification

Outputs written to validation/section7_outputs/:
    section7_all_results.csv         Full sweep results
    section7_summary.txt             Best configs above 75%
    section7_plots/                  All report figures
"""

import csv
import gc
import random
from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional

from neuron_model import CircuitIFNeuron


# =============================================================================
# Config — exact same hyperparams as Section 4
# =============================================================================

HIDDEN_SIZES  = [128, 64, 32]
TIME_STEPS    = [20, 10, 5]
RESOLUTIONS   = [28, 14, 7]
QUANT_BITS    = [None, 8, 4, 2]

EPOCHS        = 10
BATCH_SIZE    = 128
LR            = 1e-3
DT_MS         = 1.0
I_MAX_NA      = 100.0
SEED          = 42
TARGET_ACC    = 0.75

OUTPUT_DIR    = Path("validation/section7_outputs")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model — identical to Section 4 CircuitInformedMNISTSNN
# =============================================================================

class CircuitInformedMNISTSNN(nn.Module):
    def __init__(
        self,
        input_size : int   = 28 * 28,
        hidden_size: int   = 128,
        time_steps : int   = 10,
        dt_ms      : float = 1.0,
        i_max_nA   : float = 100.0,
    ) -> None:
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.time_steps  = time_steps

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_neuron = CircuitIFNeuron(dt_ms=dt_ms, I_max_nA=i_max_nA)

        self.fc2 = nn.Linear(hidden_size, 10, bias=True)
        self.output_neuron = CircuitIFNeuron(dt_ms=dt_ms, I_max_nA=i_max_nA)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _poisson_encode(x: torch.Tensor) -> torch.Tensor:
        return (torch.rand_like(x) <= x).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        out = 0.0
        for _ in range(self.time_steps):
            spikes = self._poisson_encode(x)
            h      = self.hidden_neuron(torch.sigmoid(self.fc1(spikes)))
            out    = out + self.output_neuron(torch.sigmoid(self.fc2(h)))
        return out / self.time_steps

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Quantization (post-training, same as Section 6)
# =============================================================================

def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    q_model  = deepcopy(model)
    n_levels = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if "weight" in name or "bias" in name:
                max_val = param.abs().max().clamp(min=1e-8)
                scale   = max_val / n_levels
                param.data = torch.round(param.data / scale) * scale
    return q_model


# =============================================================================
# Data
# =============================================================================

def make_loaders(resolution: int, batch_size: int) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    train_set = datasets.MNIST("./data", train=True,  transform=transform, download=True)
    test_set  = datasets.MNIST("./data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=False)
    return train_loader, test_loader


# =============================================================================
# Train / Evaluate
# =============================================================================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            functional.reset_net(model)
            out  = model(images)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
            functional.reset_net(model)
    return correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc  = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            functional.reset_net(model)
            out  = model(images)
            loss = F.cross_entropy(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)

        acc      = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"    Epoch {epoch:02d}/{epochs} | acc={acc:.4f}")

    return best_acc


# =============================================================================
# Resume helpers
# =============================================================================

def load_existing(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["accuracy"]    = float(r["accuracy"])
        r["resolution"]  = int(r["resolution"])
        r["time_steps"]  = int(r["time_steps"])
        r["hidden_size"] = int(r["hidden_size"])
        r["parameters"]  = int(r["parameters"])
        r["above_target"]= r["above_target"] == "True"
    return rows


def already_done(results, hidden, resolution, time_steps, bits) -> bool:
    q = str(bits) if bits is not None else "none"
    for r in results:
        if (r["hidden_size"]       == hidden and
            r["resolution"]        == resolution and
            r["time_steps"]        == time_steps and
            r["quantization_bits"] == q):
            return True
    return False


# =============================================================================
# Main sweep
# =============================================================================

def run_sweep(device: torch.device) -> list[dict]:
    csv_path = OUTPUT_DIR / "section7_all_results.csv"
    results  = load_existing(csv_path)
    if results:
        print(f"Resuming: {len(results)} results already done.")

    total_combos = len(HIDDEN_SIZES) * len(RESOLUTIONS) * len(TIME_STEPS)
    total_runs   = total_combos * len(QUANT_BITS)
    run_id       = 0

    print(f"Total training combos : {total_combos}")
    print(f"Total runs (with quant): {total_runs}")
    print(f"Target accuracy        : {TARGET_ACC:.0%}")
    print("=" * 65)

    for hidden, resolution, time_steps in product(
        HIDDEN_SIZES, RESOLUTIONS, TIME_STEPS
    ):
        run_id += 1
        input_size = resolution * resolution
        n_params   = (input_size * hidden + hidden +   # fc1 weight + bias
                      hidden * 10 + 10)                # fc2 weight + bias

        label = (f"[{run_id}/{total_combos}] "
                 f"hidden={hidden} | res={resolution}x{resolution} | T={time_steps} "
                 f"| params={n_params:,}")

        # Skip if all quant levels done
        if all(already_done(results, hidden, resolution, time_steps, q)
               for q in QUANT_BITS):
            print(f"\n{label}  [SKIPPED]")
            continue

        print(f"\n{label}")
        set_seed(SEED)

        model = CircuitInformedMNISTSNN(
            input_size=input_size,
            hidden_size=hidden,
            time_steps=time_steps,
            dt_ms=DT_MS,
            i_max_nA=I_MAX_NA,
        ).to(device)

        train_loader, test_loader = make_loaders(resolution, BATCH_SIZE)

        try:
            full_acc = train_model(
                model, train_loader, test_loader, EPOCHS, LR, device
            )
        except Exception as e:
            print(f"    ERROR during training: {e}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Evaluate at each quantization level
        for bits in QUANT_BITS:
            if already_done(results, hidden, resolution, time_steps, bits):
                print(f"    quant={str(bits) if bits else 'none':>4} [already saved]")
                continue

            try:
                if bits is None:
                    acc     = full_acc
                    label_q = "none"
                else:
                    q_model = quantize_model(model, bits)
                    acc     = evaluate(q_model, test_loader, device)
                    label_q = str(bits)
                    del q_model

                row = {
                    "hidden_size"      : hidden,
                    "resolution"       : resolution,
                    "time_steps"       : time_steps,
                    "quantization_bits": label_q,
                    "accuracy"         : round(acc, 5),
                    "parameters"       : n_params,
                    "above_target"     : acc >= TARGET_ACC,
                }
                results.append(row)

                status = "OK >75%" if acc >= TARGET_ACC else "below 75%"
                print(f"    quant={label_q:>4} | acc={acc:.4f} | {status}")

                # Save after every result
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerows(results)

            except Exception as e:
                print(f"    ERROR at quant={bits}: {e}")
                torch.cuda.empty_cache()
                continue

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return results


# =============================================================================
# Plots
# =============================================================================

def save_plots(results: list[dict], out_dir: Path) -> None:
    import pandas as pd
    plots_dir = out_dir / "section7_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df_full = df[df["quantization_bits"] == "none"].copy()

    colors  = {128: "steelblue", 64: "darkorange", 32: "forestgreen"}
    markers = {128: "o", 64: "s", 32: "^"}

    # ── 1. Accuracy vs Resolution (one line per hidden size) ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, T in zip(axes, TIME_STEPS):
        sub = df_full[df_full["time_steps"] == T]
        for h in HIDDEN_SIZES:
            grp = sub[sub["hidden_size"] == h].sort_values("resolution")
            ax.plot(grp["resolution"], grp["accuracy"],
                    marker=markers[h], color=colors[h], linewidth=2,
                    label=f"Hidden={h}")
        ax.axhline(TARGET_ACC, color="red", linestyle="--",
                   linewidth=1.2, label="75% target")
        ax.set_title(f"T={T} time steps", fontweight="bold")
        ax.set_xlabel("Resolution (px)")
        ax.set_xticks(RESOLUTIONS)
        ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test Accuracy")
    fig.suptitle("Section 7: Accuracy vs Resolution by Hidden Size",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "sec7_accuracy_vs_resolution.png", dpi=200)
    plt.close(fig)

    # ── 2. Accuracy vs Hidden Size ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    line_colors = {28: "steelblue", 14: "darkorange", 7: "forestgreen"}
    for ax, T in zip(axes, TIME_STEPS):
        sub = df_full[df_full["time_steps"] == T]
        for res in RESOLUTIONS:
            grp = sub[sub["resolution"] == res].sort_values("hidden_size")
            ax.plot(grp["hidden_size"], grp["accuracy"],
                    marker="o", color=line_colors[res], linewidth=2,
                    label=f"{res}x{res}")
        ax.axhline(TARGET_ACC, color="red", linestyle="--",
                   linewidth=1.2, label="75% target")
        ax.set_title(f"T={T} time steps", fontweight="bold")
        ax.set_xlabel("Hidden Layer Size")
        ax.set_xticks(HIDDEN_SIZES)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test Accuracy")
    fig.suptitle("Section 7: Accuracy vs Hidden Size by Resolution",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "sec7_accuracy_vs_hidden.png", dpi=200)
    plt.close(fig)

    # ── 3. Accuracy vs Quantization for all configs above 75% ────────────────
    passing = df_full[df_full["above_target"] == True].copy()
    quant_order  = ["none", "8", "4", "2"]
    quant_labels = ["Full\nPrecision", "8-bit", "4-bit", "2-bit"]

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for _, base_row in passing.iterrows():
        h   = base_row["hidden_size"]
        res = base_row["resolution"]
        T   = base_row["time_steps"]
        sub = df[(df["hidden_size"] == h) &
                 (df["resolution"]  == res) &
                 (df["time_steps"]  == T)]
        accs = []
        for q in quant_order:
            row = sub[sub["quantization_bits"] == q]
            accs.append(float(row["accuracy"].values[0]) if len(row) else np.nan)

        lbl = f"H={h} {res}x{res} T={T}"
        ax.plot(range(4), accs, marker="o", linewidth=1.5, label=lbl)
        plotted += 1

    ax.axhline(TARGET_ACC, color="red", linestyle="--",
               linewidth=1.5, label="75% target")
    ax.set_xticks(range(4))
    ax.set_xticklabels(quant_labels)
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Section 7: Quantization Effect on Configs Above 75%",
                 fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "sec7_quantization_effect.png", dpi=200)
    plt.close(fig)

    # ── 4. Parameter count vs accuracy bubble chart ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df_full["parameters"],
        df_full["accuracy"],
        c=df_full["time_steps"],
        cmap="viridis",
        s=80,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(TARGET_ACC, color="red", linestyle="--",
               linewidth=1.5, label="75% target")
    plt.colorbar(scatter, ax=ax, label="Time Steps")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Section 7: Model Size vs Accuracy",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "sec7_params_vs_accuracy.png", dpi=200)
    plt.close(fig)

    print(f"All plots saved to {plots_dir}")


# =============================================================================
# Summary
# =============================================================================

def write_summary(results: list[dict], out_dir: Path) -> None:
    import pandas as pd
    df      = pd.DataFrame(results)
    df_full = df[df["quantization_bits"] == "none"].copy()

    # Configs above target at full precision
    passing = df_full[df_full["above_target"] == True].sort_values(
        ["parameters", "time_steps", "resolution"]
    )

    # For each passing config, find which quant levels also pass
    lines = [
        "=" * 65,
        "EE 533  Section 7 - Network Simplification Summary",
        "=" * 65,
        f"Target accuracy: {TARGET_ACC:.0%}",
        "",
        "CONFIGS ABOVE 75% AT FULL PRECISION (sorted by model size):",
        "-" * 65,
        f"{'Hidden':>8} {'Res':>8} {'T':>4} {'Params':>10} {'Acc':>8}  Max quant still >75%",
        "-" * 65,
    ]

    for _, row in passing.iterrows():
        h   = int(row["hidden_size"])
        res = int(row["resolution"])
        T   = int(row["time_steps"])
        acc = float(row["accuracy"])
        p   = int(row["parameters"])

        # Find max quantization that still passes
        sub = df[(df["hidden_size"] == h) &
                 (df["resolution"]  == res) &
                 (df["time_steps"]  == T)]
        max_quant = "full only"
        for bits in [8, 4, 2]:
            q_row = sub[sub["quantization_bits"] == str(bits)]
            if len(q_row) and float(q_row["accuracy"].values[0]) >= TARGET_ACC:
                max_quant = f"{bits}-bit"

        lines.append(
            f"{h:>8} {res:>6}x{res:<2} {T:>4} {p:>10,} {acc:>8.4f}  {max_quant}"
        )

    # Find the single most simplified config that passes
    if len(passing) > 0:
        best = passing.iloc[0]
        lines += [
            "",
            "MOST SIMPLIFIED CONFIG ABOVE 75%:",
            "-" * 65,
            f"  Hidden size : {int(best['hidden_size'])}",
            f"  Resolution  : {int(best['resolution'])}x{int(best['resolution'])}",
            f"  Time steps  : {int(best['time_steps'])}",
            f"  Parameters  : {int(best['parameters']):,}",
            f"  Accuracy    : {float(best['accuracy']):.4f}",
        ]

        # Compare to full model
        full_row = df_full[
            (df_full["hidden_size"] == 128) &
            (df_full["resolution"]  == 28) &
            (df_full["time_steps"]  == 20)
        ]
        if len(full_row):
            full_acc    = float(full_row["accuracy"].values[0])
            full_params = int(full_row["parameters"].values[0])
            param_reduction = (1 - int(best["parameters"]) / full_params) * 100
            acc_drop        = full_acc - float(best["accuracy"])
            lines += [
                "",
                "COMPARISON TO FULL MODEL (H=128, 28x28, T=20):",
                "-" * 65,
                f"  Full model accuracy  : {full_acc:.4f}",
                f"  Full model params    : {full_params:,}",
                f"  Simplified accuracy  : {float(best['accuracy']):.4f}",
                f"  Simplified params    : {int(best['parameters']):,}",
                f"  Parameter reduction  : {param_reduction:.1f}%",
                f"  Accuracy drop        : {acc_drop:.4f} ({acc_drop*100:.2f}%)",
            ]

    lines += ["", "=" * 65]
    text = "\n".join(lines)
    (out_dir / "section7_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_sweep(device)

    save_plots(results, OUTPUT_DIR)
    write_summary(results, OUTPUT_DIR)

    print(f"\nAll Section 7 outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()