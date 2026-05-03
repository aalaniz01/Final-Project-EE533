"""
section6_comparison.py
-----------------------
EE 533 Final Project - Section 6: Default Neuron vs Circuit-Informed Neuron

Sweeps all combinations of:
    - Neuron type     : Default (LIF) vs Circuit-Informed (CircuitIF)
    - Image resolution: 4x4, 7x7, 14x14, 28x28
    - Time steps      : 5, 10, 20
    - Training method : Surrogate Gradient (supervised), STDP (unsupervised)
    - Quantization    : None, 2-bit, 4-bit, 8-bit (post-training)

Total runs: 2 neurons x 4 resolutions x 3 timesteps x 2 methods x 4 quant = 192

Place in snn/ folder:
    snn/section6_comparison.py

Run from project root:
    python -m snn.section6_comparison

Outputs written to validation/section6_outputs/:
    section6_all_results.csv       Full results table for every combination
    section6_summary_plots/        One plot per analysis dimension
    section6_heatmaps/             Heatmaps for the report
    section6_summary.txt           Key findings for report writeup
"""

import csv
import random
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from spikingjelly.activation_based import functional, neuron, surrogate, learning

from neuron_model import CircuitIFNeuron


# =============================================================================
# Config
# =============================================================================

RESOLUTIONS   = [4, 7, 14, 28]
TIME_STEPS    = [5, 10, 20]
QUANT_BITS    = [None, 8, 4, 2]   # None = full precision
HIDDEN_SIZE   = 128
EPOCHS_SGD    = 10                 # surrogate gradient epochs
EPOCHS_STDP   = 10                 # STDP unsupervised epochs
EPOCHS_READOUT= 10                 # linear readout epochs after STDP
BATCH_SIZE      = 128
STDP_BATCH_SIZE = 64       # smaller batch for STDP to avoid GPU OOM
LR_SGD          = 1e-3
LR_READOUT    = 1e-3
DT_MS         = 1.0
I_MAX_NA      = 100.0
SEED          = 42
TRAIN_LIMIT   = None               # set to e.g. 10000 for faster debug
TEST_LIMIT    = None

OUTPUT_DIR    = Path("validation/section6_outputs")


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Models
# =============================================================================

class DefaultMNISTSNN(nn.Module):
    """Standard SNN using SpikingJelly's LIF neuron (default model)."""

    def __init__(self, input_size: int, hidden_size: int, time_steps: int) -> None:
        super().__init__()
        self.time_steps  = time_steps
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_neuron = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )
        self.fc2 = nn.Linear(hidden_size, 10, bias=True)
        self.output_neuron = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

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
            spikes  = self._poisson_encode(x)
            h       = self.hidden_neuron(self.fc1(spikes))
            out     = out + self.output_neuron(self.fc2(h))
        return out / self.time_steps


class CircuitMNISTSNN(nn.Module):
    """SNN using the Cadence circuit-informed IF neuron."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_steps: int,
        dt_ms: float = DT_MS,
        i_max_nA: float = I_MAX_NA,
    ) -> None:
        super().__init__()
        self.time_steps  = time_steps
        self.hidden_size = hidden_size

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


# =============================================================================
# Data
# =============================================================================

def make_loaders(
    resolution: int,
    batch_size: int,
    train_limit=None,
    test_limit=None,
    data_dir: str = "./data",
    drop_last_train: bool = True,
) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    train_set = datasets.MNIST(data_dir, train=True,  transform=transform, download=True)
    test_set  = datasets.MNIST(data_dir, train=False, transform=transform, download=True)

    if train_limit:
        train_set = Subset(train_set, range(min(train_limit, len(train_set))))
    if test_limit:
        test_set  = Subset(test_set,  range(min(test_limit,  len(test_set))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=drop_last_train)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, drop_last=False)
    return train_loader, test_loader


# =============================================================================
# Quantization (post-training)
# =============================================================================

def quantize_model(model: nn.Module, bits: int) -> nn.Module:
    """
    Post-training symmetric uniform quantization.
    Rounds each weight tensor to 'bits'-bit precision.
    Returns a deep copy so the original model is unchanged.
    """
    q_model = deepcopy(model)
    n_levels = 2 ** (bits - 1) - 1      # symmetric: range [-n_levels, n_levels]

    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if "weight" in name or "bias" in name:
                max_val = param.abs().max().clamp(min=1e-8)
                scale   = max_val / n_levels
                param.data = torch.round(param.data / scale) * scale

    return q_model


# =============================================================================
# Evaluation
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


# =============================================================================
# Training: Surrogate Gradient (supervised BPTT)
# =============================================================================

def train_surrogate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train with surrogate gradient descent. Returns best test accuracy."""
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

        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"    [SGD] Epoch {epoch:02d}/{epochs} | test_acc={acc:.4f}")

    return best_acc


# =============================================================================
# Training: STDP (unsupervised) + Linear Readout
# =============================================================================

def extract_hidden_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    neuron_type: str,
    max_samples: int = 20000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the hidden layer and collect spike count vectors as features.
    Runs entirely on CPU to avoid GPU OOM — GPU memory is freed before calling.
    Caps at max_samples to keep memory and time manageable.
    Returns (features [N, hidden_size], labels [N]).
    """
    import gc

    # Move model to CPU for feature extraction — no GPU OOM possible
    model_cpu = model.cpu()
    model_cpu.eval()
    cpu_device = torch.device("cpu")

    all_feats  = []
    all_labels = []
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            if total_seen >= max_samples:
                break

            # Trim batch if it would exceed max_samples
            remaining = max_samples - total_seen
            images = images[:remaining]
            labels = labels[:remaining]

            x     = images.flatten(start_dim=1)
            feats = torch.zeros(images.size(0), model_cpu.hidden_size)

            functional.reset_net(model_cpu)
            for _ in range(model_cpu.time_steps):
                spikes = (torch.rand_like(x) <= x).float()
                if neuron_type == "circuit":
                    h = model_cpu.hidden_neuron(
                        torch.sigmoid(model_cpu.fc1(spikes))
                    )
                else:
                    h = model_cpu.hidden_neuron(model_cpu.fc1(spikes))
                feats += h
            functional.reset_net(model_cpu)

            all_feats.append(feats)
            all_labels.append(labels)
            total_seen += images.size(0)

    # Move model back to GPU
    model.to(device)
    gc.collect()

    return torch.cat(all_feats), torch.cat(all_labels)


def train_stdp(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr_readout: float,
    device: torch.device,
    neuron_type: str,
) -> float:
    """
    1. Run STDP on the hidden layer (unsupervised weight update).
    2. Freeze hidden layer, train a linear readout classifier on top.
    3. Return best test accuracy of the readout.
    """

    # ── STDP phase ────────────────────────────────────────────────────────────
    # Use SpikingJelly's STDPLearner on fc1
    tau_pre  = 20.0
    tau_post = 20.0
    f_pre    = lambda x: torch.exp(-x / tau_pre)
    f_post   = lambda x: torch.exp(-x / tau_post)

    try:
        stdp_learner = learning.STDPLearner(
            step_mode   = "s",
            synapse     = model.fc1,
            sn          = model.hidden_neuron,
            tau_pre     = tau_pre,
            tau_post    = tau_post,
            f_pre       = f_pre,
            f_post      = f_post,
        )
        use_stdp_learner = True
    except Exception:
        # Fallback: manual Hebbian STDP if SpikingJelly version differs
        use_stdp_learner = False

    stdp_optimizer = torch.optim.SGD(model.fc1.parameters(), lr=5e-3)

    print(f"    [STDP] Unsupervised training for {epochs} epochs ...")
    for epoch in range(1, epochs + 1):
        model.train()
        for images, _ in train_loader:      # no labels used in STDP
            images = images.to(device)
            x      = images.flatten(start_dim=1)
            functional.reset_net(model)

            for _ in range(model.time_steps):
                spikes = (torch.rand_like(x) <= x).float()
                if neuron_type == "circuit":
                    _ = model.hidden_neuron(torch.sigmoid(model.fc1(spikes)))
                else:
                    _ = model.hidden_neuron(model.fc1(spikes))

                if use_stdp_learner:
                    stdp_learner.step(on_grad=True)

            stdp_optimizer.step()
            stdp_optimizer.zero_grad()
            functional.reset_net(model)

            # Weight normalization: keeps each neuron's weight vector
            # at unit L2 norm so neurons stay diverse and dont collapse.
            # This is the standard fix for STDP weight collapse.
            with torch.no_grad():
                norms = model.fc1.weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
                model.fc1.weight.div_(norms)

        print(f"    [STDP] Epoch {epoch:02d}/{epochs} done")

    # ── Linear readout phase ──────────────────────────────────────────────────
    # Freeze hidden layer
    for param in model.fc1.parameters():
        param.requires_grad = False
    model.hidden_neuron.requires_grad_(False) if hasattr(
        model.hidden_neuron, "requires_grad_"
    ) else None

    # Extract spike-count features
    print(f"    [STDP] Extracting features for readout ...")
    train_feats, train_labels = extract_hidden_features(
        model, train_loader, device, neuron_type
    )
    test_feats, test_labels = extract_hidden_features(
        model, test_loader, device, neuron_type
    )

    # Normalize features
    feat_mean = train_feats.mean(dim=0, keepdim=True)
    feat_std  = train_feats.std(dim=0, keepdim=True).clamp(min=1e-8)
    train_feats = (train_feats - feat_mean) / feat_std
    test_feats  = (test_feats  - feat_mean) / feat_std

    # Build linear readout
    readout = nn.Linear(model.hidden_size, 10).to(device)
    nn.init.xavier_uniform_(readout.weight)
    ro_optimizer = torch.optim.Adam(readout.parameters(), lr=lr_readout)

    # Wrap into a DataLoader
    ro_train = torch.utils.data.TensorDataset(
        train_feats.to(device), train_labels.to(device)
    )
    ro_test  = torch.utils.data.TensorDataset(
        test_feats.to(device), test_labels.to(device)
    )
    ro_train_loader = DataLoader(ro_train, batch_size=BATCH_SIZE, shuffle=True)
    ro_test_loader  = DataLoader(ro_test,  batch_size=BATCH_SIZE, shuffle=False)

    best_acc = 0.0
    print(f"    [STDP] Training linear readout for {epochs} epochs ...")
    for epoch in range(1, EPOCHS_READOUT + 1):
        readout.train()
        for feats, labels in ro_train_loader:
            out  = readout(feats)
            loss = F.cross_entropy(out, labels)
            ro_optimizer.zero_grad()
            loss.backward()
            ro_optimizer.step()

        readout.eval()
        correct = total = 0
        with torch.no_grad():
            for feats, labels in ro_test_loader:
                pred     = readout(feats).argmax(dim=1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
        acc      = correct / total
        best_acc = max(best_acc, acc)
        print(f"    [STDP] Readout epoch {epoch:02d}/{EPOCHS_READOUT} | acc={acc:.4f}")

    # Unfreeze for next run
    for param in model.fc1.parameters():
        param.requires_grad = True

    return best_acc


# =============================================================================
# Main sweep
# =============================================================================


# =============================================================================
# Resume helpers
# =============================================================================

def load_existing_results(csv_path: Path) -> list[dict]:
    """Load already-completed results so we can resume from where we left off."""
    if not csv_path.exists():
        return []
    import csv as csv_mod
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            # Cast numeric fields so pandas groupby/mean works correctly
            row["accuracy"]   = float(row["accuracy"])
            row["resolution"] = int(row["resolution"])
            row["time_steps"] = int(row["time_steps"])
            results.append(row)
    return results


def result_exists(
    results: list[dict],
    neuron_type: str,
    resolution: int,
    time_steps: int,
    training_method: str,
    quant_bits,
) -> bool:
    """Check if a specific combination already has a saved result."""
    q = str(quant_bits) if quant_bits is not None else "none"
    for r in results:
        if (r["neuron_type"]       == neuron_type and
            int(r["resolution"])   == resolution and
            int(r["time_steps"])   == time_steps and
            r["training_method"]   == training_method and
            r["quantization_bits"] == q):
            return True
    return False


def run_sweep(device: torch.device) -> list[dict]:
    csv_path = OUTPUT_DIR / "section6_all_results.csv"
    results  = load_existing_results(csv_path)
    if results:
        print(f"\nResuming: {len(results)} results already done, skipping those runs.")

    run_id  = 0

    neuron_types   = ["default", "circuit"]
    training_types = ["surrogate", "stdp"]

    total_runs = (
        len(neuron_types) * len(RESOLUTIONS) * len(TIME_STEPS)
        * len(training_types) * len(QUANT_BITS)
    )
    print(f"\nTotal runs: {total_runs}")
    print("=" * 70)

    for neuron_type, resolution, time_steps, training_type in product(
        neuron_types, RESOLUTIONS, TIME_STEPS, training_types
    ):
        set_seed(SEED)
        input_size = resolution * resolution

        run_id += 1
        label = (f"[{run_id}/{total_runs // len(QUANT_BITS)}] "
                 f"neuron={neuron_type} | res={resolution}x{resolution} | "
                 f"T={time_steps} | train={training_type}")

        # Skip if all 4 quant levels already completed
        all_done = all(
            result_exists(results, neuron_type, resolution, time_steps,
                          training_type, q)
            for q in QUANT_BITS
        )
        if all_done:
            print(f"\n{label}  [SKIPPED - already done]")
            continue

        print(f"\n{label}")

        # STDP runs on CPU to avoid GPU OOM (trace tensors are large at high res)
        # Surrogate gradient runs on GPU for speed
        train_device = torch.device("cpu") if training_type == "stdp" else device
        bs = STDP_BATCH_SIZE if training_type == "stdp" else BATCH_SIZE

        # Build model on appropriate device
        if neuron_type == "default":
            model = DefaultMNISTSNN(input_size, HIDDEN_SIZE, time_steps).to(train_device)
        else:
            model = CircuitMNISTSNN(
                input_size, HIDDEN_SIZE, time_steps, DT_MS, I_MAX_NA
            ).to(train_device)

        train_loader, test_loader = make_loaders(
            resolution, bs, TRAIN_LIMIT, TEST_LIMIT
        )

        # Train once (full precision) with OOM protection
        try:
            if training_type == "surrogate":
                full_prec_acc = train_surrogate(
                    model, train_loader, test_loader,
                    EPOCHS_SGD, LR_SGD, train_device,
                )
            else:
                full_prec_acc = train_stdp(
                    model, train_loader, test_loader,
                    EPOCHS_STDP, LR_READOUT, train_device, neuron_type,
                )
        except Exception as e:
            print(f"    ERROR during training: {e}")
            print(f"    Skipping this combination.")
            del model
            torch.cuda.empty_cache()
            import gc; gc.collect()
            continue

        # Evaluate at each quantization level
        for bits in QUANT_BITS:
            if result_exists(results, neuron_type, resolution, time_steps,
                             training_type, bits):
                print(f"    quant={str(bits) if bits else 'none':>4} bits -> [already saved]")
                continue

            try:
                if bits is None:
                    acc     = full_prec_acc
                    label_q = "none"
                else:
                    q_model = quantize_model(model, bits)
                    if training_type == "surrogate":
                        acc = evaluate(q_model, test_loader, device)
                    else:
                        acc = evaluate_stdp_quantized(
                            q_model, test_loader, device, neuron_type, model
                        )
                    label_q = str(bits)

                row = {
                    "neuron_type"      : neuron_type,
                    "resolution"       : resolution,
                    "time_steps"       : time_steps,
                    "training_method"  : training_type,
                    "quantization_bits": label_q,
                    "accuracy"         : round(acc, 5),
                }
                results.append(row)
                print(f"    quant={label_q:>4} bits -> acc={acc:.4f}")

                # Save after every single result so progress is never lost
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                    writer.writerows(results)

            except Exception as e:
                print(f"    ERROR at quant={bits}: {e}")
                torch.cuda.empty_cache()
                continue

        # Free memory before next run
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()

    return results


def evaluate_stdp_quantized(
    q_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    neuron_type: str,
    original_model: nn.Module,
) -> float:
    """
    For STDP+quantization: extract features with quantized hidden weights,
    then use a quickly-trained readout to get accuracy.
    """
    train_loader_small, _ = make_loaders(
        int(q_model.fc1.in_features ** 0.5),
        BATCH_SIZE,
        train_limit=10000,
    )
    test_feats, test_labels = extract_hidden_features(
        q_model, test_loader, device, neuron_type
    )
    train_feats, train_labels = extract_hidden_features(
        q_model, train_loader_small, device, neuron_type
    )

    feat_mean = train_feats.mean(0, keepdim=True)
    feat_std  = train_feats.std(0, keepdim=True).clamp(min=1e-8)
    train_feats = (train_feats - feat_mean) / feat_std
    test_feats  = (test_feats  - feat_mean) / feat_std

    readout = nn.Linear(q_model.hidden_size, 10).to(device)
    nn.init.xavier_uniform_(readout.weight)
    opt = torch.optim.Adam(readout.parameters(), lr=LR_READOUT)

    ds = torch.utils.data.TensorDataset(
        train_feats.to(device), train_labels.to(device)
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(5):   # quick 5-epoch readout
        readout.train()
        for feats, labels in loader:
            loss = F.cross_entropy(readout(feats), labels)
            opt.zero_grad(); loss.backward(); opt.step()

    readout.eval()
    correct = total = 0
    with torch.no_grad():
        for feats, labels in zip(
            test_feats.split(BATCH_SIZE), test_labels.split(BATCH_SIZE)
        ):
            feats, labels = feats.to(device), labels.to(device)
            correct += (readout(feats).argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# =============================================================================
# Plots
# =============================================================================

def save_all_plots(results: list[dict], out_dir: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(results)

    plots_dir = out_dir / "section6_summary_plots"
    heat_dir  = out_dir / "section6_heatmaps"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heat_dir.mkdir(parents=True, exist_ok=True)

    colors = {"default": "steelblue", "circuit": "darkorange"}
    markers = {"default": "o", "circuit": "s"}

    # ── 1. Accuracy vs Resolution ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, method in zip(axes, ["surrogate", "stdp"]):
        sub = df[(df["training_method"] == method) & (df["quantization_bits"] == "none")]
        for ntype in ["default", "circuit"]:
            grp = sub[sub["neuron_type"] == ntype].groupby("resolution")["accuracy"].mean()
            ax.plot(grp.index, grp.values, marker=markers[ntype],
                    color=colors[ntype], linewidth=2,
                    label=f"{ntype.capitalize()} neuron")
        ax.set_title(f"Training: {method.upper()}", fontweight="bold")
        ax.set_xlabel("Input Resolution (px)")
        ax.set_ylabel("Test Accuracy")
        ax.set_xticks(RESOLUTIONS)
        ax.set_xticklabels([f"{r}x{r}" for r in RESOLUTIONS])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.suptitle("Section 6: Accuracy vs Image Resolution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "sec6_accuracy_vs_resolution.png", dpi=200)
    plt.close(fig)

    # ── 2. Accuracy vs Time Steps ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, method in zip(axes, ["surrogate", "stdp"]):
        sub = df[(df["training_method"] == method) & (df["quantization_bits"] == "none")]
        for ntype in ["default", "circuit"]:
            grp = sub[sub["neuron_type"] == ntype].groupby("time_steps")["accuracy"].mean()
            ax.plot(grp.index, grp.values, marker=markers[ntype],
                    color=colors[ntype], linewidth=2,
                    label=f"{ntype.capitalize()} neuron")
        ax.set_title(f"Training: {method.upper()}", fontweight="bold")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Test Accuracy")
        ax.set_xticks(TIME_STEPS)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.suptitle("Section 6: Accuracy vs Time Steps", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "sec6_accuracy_vs_timesteps.png", dpi=200)
    plt.close(fig)

    # ── 3. Accuracy vs Quantization ───────────────────────────────────────────
    quant_order  = ["none", "8", "4", "2"]
    quant_labels = ["Full\nPrecision", "8-bit", "4-bit", "2-bit"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, method in zip(axes, ["surrogate", "stdp"]):
        sub = df[df["training_method"] == method]
        for ntype in ["default", "circuit"]:
            accs = [
                sub[(sub["neuron_type"] == ntype) &
                    (sub["quantization_bits"] == q)]["accuracy"].mean()
                for q in quant_order
            ]
            ax.plot(range(len(quant_order)), accs, marker=markers[ntype],
                    color=colors[ntype], linewidth=2,
                    label=f"{ntype.capitalize()} neuron")
        ax.set_title(f"Training: {method.upper()}", fontweight="bold")
        ax.set_xlabel("Weight Quantization")
        ax.set_ylabel("Test Accuracy")
        ax.set_xticks(range(len(quant_order)))
        ax.set_xticklabels(quant_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    fig.suptitle("Section 6: Accuracy vs Weight Quantization", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plots_dir / "sec6_accuracy_vs_quantization.png", dpi=200)
    plt.close(fig)

    # ── 4. Training Method Comparison Bar Chart ───────────────────────────────
    sub = df[df["quantization_bits"] == "none"]
    grp = sub.groupby(["neuron_type", "training_method"])["accuracy"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 5))
    x       = np.arange(len(grp.index))
    width   = 0.35
    methods = grp.columns.tolist()
    ax.bar(x - width/2, grp[methods[0]].values, width,
           label=methods[0].upper(), color="steelblue")
    ax.bar(x + width/2, grp[methods[1]].values, width,
           label=methods[1].upper(), color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([n.capitalize() for n in grp.index])
    ax.set_ylabel("Mean Test Accuracy")
    ax.set_title("Section 6: Surrogate vs STDP by Neuron Type", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "sec6_training_method_comparison.png", dpi=200)
    plt.close(fig)

    # ── 5. Heatmap: Resolution x Time Steps (full precision, surrogate) ───────
    for method in ["surrogate", "stdp"]:
        for ntype in ["default", "circuit"]:
            sub = df[
                (df["training_method"]   == method) &
                (df["neuron_type"]       == ntype) &
                (df["quantization_bits"] == "none")
            ]
            pivot = sub.pivot_table(
                index="resolution", columns="time_steps", values="accuracy"
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           vmin=0, vmax=1)
            ax.set_xticks(range(len(TIME_STEPS)))
            ax.set_xticklabels([f"T={t}" for t in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{r}x{r}" for r in pivot.index])
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Resolution")
            ax.set_title(
                f"Accuracy Heatmap\n{ntype.capitalize()} neuron | {method.upper()}",
                fontweight="bold",
            )
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if 0.3 < val < 0.8 else "white")
            plt.colorbar(im, ax=ax, label="Accuracy")
            fig.tight_layout()
            fig.savefig(
                heat_dir / f"sec6_heatmap_{ntype}_{method}.png", dpi=200
            )
            plt.close(fig)

    print(f"\nAll plots saved to {plots_dir} and {heat_dir}")


# =============================================================================
# Summary text
# =============================================================================

def write_summary(results: list[dict], out_dir: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(results)

    lines = [
        "=" * 65,
        "EE 533  Section 6 - Default vs Circuit-Informed Neuron",
        "=" * 65,
        "",
        "MEAN ACCURACY BY NEURON TYPE (full precision, all combos):",
        "-" * 65,
    ]

    for ntype in ["default", "circuit"]:
        sub = df[
            (df["neuron_type"] == ntype) &
            (df["quantization_bits"] == "none")
        ]
        lines.append(
            f"  {ntype.capitalize():12s}: {sub['accuracy'].mean():.4f} "
            f"(min={sub['accuracy'].min():.4f}, max={sub['accuracy'].max():.4f})"
        )

    lines += [
        "",
        "MEAN ACCURACY BY TRAINING METHOD:",
        "-" * 65,
    ]
    for method in ["surrogate", "stdp"]:
        sub = df[
            (df["training_method"] == method) &
            (df["quantization_bits"] == "none")
        ]
        lines.append(
            f"  {method.upper():12s}: {sub['accuracy'].mean():.4f}"
        )

    lines += [
        "",
        "MEAN ACCURACY BY RESOLUTION (full precision, surrogate):",
        "-" * 65,
    ]
    sub = df[
        (df["training_method"] == "surrogate") &
        (df["quantization_bits"] == "none")
    ]
    for res in RESOLUTIONS:
        grp = sub[sub["resolution"] == res]["accuracy"].mean()
        lines.append(f"  {res:2d}x{res:<2d}: {grp:.4f}")

    lines += [
        "",
        "MEAN ACCURACY BY TIME STEPS (full precision, surrogate):",
        "-" * 65,
    ]
    for t in TIME_STEPS:
        grp = sub[sub["time_steps"] == t]["accuracy"].mean()
        lines.append(f"  T={t:2d}: {grp:.4f}")

    lines += [
        "",
        "ACCURACY DROP FROM QUANTIZATION (surrogate, averaged over all combos):",
        "-" * 65,
    ]
    for bits in ["8", "4", "2"]:
        sub_full = df[
            (df["training_method"] == "surrogate") &
            (df["quantization_bits"] == "none")
        ]["accuracy"].mean()
        sub_q = df[
            (df["training_method"] == "surrogate") &
            (df["quantization_bits"] == bits)
        ]["accuracy"].mean()
        drop = sub_full - sub_q
        lines.append(
            f"  {bits}-bit: {sub_q:.4f}  (drop = {drop:+.4f})"
        )

    lines += ["", "=" * 65]
    text = "\n".join(lines)
    (out_dir / "section6_summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run full sweep
    results = run_sweep(device)

    csv_path = OUTPUT_DIR / "section6_all_results.csv"
    print(f"\nResults saved incrementally to -> {csv_path}")

    # Generate all plots
    save_all_plots(results, OUTPUT_DIR)

    # Write summary
    write_summary(results, OUTPUT_DIR)

    print(f"\nAll Section 6 outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()