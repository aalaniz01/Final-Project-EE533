"""
section5_generate_pwlf_exact.py
-------------------------------
EE 533 Final Project - Task 5 PWLF Generator

This version follows the Cadence PWLF instructions exactly:

Required output:
    neuron_0_pwlf.txt
    neuron_1_pwlf.txt
    neuron_2_pwlf.txt
    neuron_3_pwlf.txt

Each file:
    - plain .txt
    - two columns only
    - no header
    - time in seconds
    - current in amps
    - current clipped to 1e-9 A through 100e-9 A
    - repeated timestamps are used to force step changes in Cadence

Recommended location:
    validation/section5_generate_pwlf_exact.py

Run from project root:
    python validation/section5_generate_pwlf_exact.py \
        --checkpoint validation/section4_outputs/section4_circuit_snn_best.pt \
        --sample-index 0

Optional, pick specific hidden neurons:
    python validation/section5_generate_pwlf_exact.py \
        --checkpoint validation/section4_outputs/section4_circuit_snn_best.pt \
        --sample-index 0 \
        --neuron-indices 0 1 2 3
"""

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Allow this script to be run directly from:
#     python validation/section5_generate_pwlf_exact.py
# by adding the project root folder to Python's import path.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spikingjelly.activation_based import functional
from neuron_model import CircuitIFNeuron


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CircuitInformedMNISTSNN(nn.Module):
    """
    Same architecture used in Section 4:
        input layer -> hidden layer -> output layer

    This class is only used here so the Section 4 checkpoint can be loaded
    and the trained fc1 weights can be used to generate Task 5 currents.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_size: int = 128,
        time_steps: int = 10,
        dt_ms: float = 1.0,
        i_max_nA: float = 100.0,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_steps = time_steps

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.hidden_neuron = CircuitIFNeuron(dt_ms=dt_ms, I_max_nA=i_max_nA)

        self.fc2 = nn.Linear(hidden_size, 10, bias=True)
        self.output_neuron = CircuitIFNeuron(dt_ms=dt_ms, I_max_nA=i_max_nA)

    @staticmethod
    def current_normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.relu(x), 0.0, 1.0)


def load_checkpoint_args(checkpoint: dict, cli_args: argparse.Namespace) -> dict:
    saved_args = checkpoint.get("args", {})

    return {
        "hidden_size": int(saved_args.get("hidden_size", cli_args.hidden_size)),
        "time_steps": int(saved_args.get("time_steps", cli_args.time_steps)),
        "input_resolution": int(saved_args.get("input_resolution", cli_args.input_resolution)),
        "dt_ms": float(saved_args.get("dt_ms", cli_args.dt_ms)),
        "i_max_nA": float(saved_args.get("i_max_nA", cli_args.i_max_nA)),
    }


def load_test_sample(data_dir: str, input_resolution: int, sample_index: int):
    transform = transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
        ]
    )

    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    image, label = test_set[sample_index]
    return image, int(label)


def save_input_image(image: torch.Tensor, label: int, out_path: Path) -> None:
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze(0).numpy(), cmap="gray")
    plt.title(f"MNIST sample, label={label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def generate_input_spike_train(image: torch.Tensor, time_steps: int, seed: int) -> torch.Tensor:
    """
    Deterministic rate-coded MNIST spike train.
    Returns shape [T, input_size].
    """
    torch.manual_seed(seed)
    x = image.flatten()

    spikes = []
    for _ in range(time_steps):
        spikes.append((torch.rand_like(x) <= x).float())

    return torch.stack(spikes, dim=0)


def compute_hidden_current_norm(model: CircuitInformedMNISTSNN, input_spikes: torch.Tensor) -> torch.Tensor:
    """
    Uses trained fc1 weights:
        hidden_current_norm[t] = clamp(relu(fc1(input_spikes[t])), 0, 1)

    Returns shape [T, hidden_size].
    """
    model.eval()

    currents = []
    with torch.no_grad():
        for t in range(input_spikes.shape[0]):
            x_t = input_spikes[t].unsqueeze(0)
            hidden_linear = model.fc1(x_t)
            hidden_current_norm = model.current_normalize(hidden_linear)
            currents.append(hidden_current_norm.squeeze(0).cpu())

    return torch.stack(currents, dim=0)


def simulate_reference_spikes_from_currents(
    currents_A: torch.Tensor,
    dt_ms: float,
    i_max_nA: float,
) -> torch.Tensor:
    """
    Simulates Python reference spikes using the exact currents exported to Cadence.

    The CircuitIFNeuron expects normalized current:
        1.0 -> i_max_nA
    so current_norm = current_A / (i_max_nA * 1e-9).
    """
    neuron = CircuitIFNeuron(dt_ms=dt_ms, I_max_nA=i_max_nA)
    functional.reset_net(neuron)

    currents_norm = currents_A / (i_max_nA * 1e-9)

    spikes = []
    with torch.no_grad():
        for t in range(currents_norm.shape[0]):
            s_t = neuron(currents_norm[t].unsqueeze(0))
            spikes.append(s_t.squeeze(0).cpu())

    functional.reset_net(neuron)
    return torch.stack(spikes, dim=0)


def choose_neurons(
    exported_currents_A: torch.Tensor,
    reference_spikes: torch.Tensor,
    requested_indices: list[int] | None,
    count: int = 4,
) -> list[int]:
    hidden_size = exported_currents_A.shape[1]

    if requested_indices:
        selected = [int(i) for i in requested_indices]
        for i in selected:
            if i < 0 or i >= hidden_size:
                raise ValueError(f"Hidden neuron index {i} is outside 0..{hidden_size - 1}")
        if len(selected) != 4:
            raise ValueError("Task 5 needs exactly 4 neuron indices.")
        return selected

    # Prefer neurons with spikes. If tied, prefer higher average current.
    spike_counts = reference_spikes.sum(dim=0)
    mean_currents = exported_currents_A.mean(dim=0)

    score = spike_counts * 1000.0 + mean_currents
    selected = torch.argsort(score, descending=True)[:count].tolist()

    return [int(i) for i in selected]


def make_step_points_duplicate_time(currents_A: np.ndarray, dt_s: float) -> list[tuple[float, float]]:
    """
    Creates a step waveform using duplicate timestamps.

    For T = 3:
        0      I0
        dt     I0
        dt     I1
        2dt    I1
        2dt    I2
        3dt    I2
    """
    points = []

    T = len(currents_A)

    for k in range(T):
        t0 = k * dt_s
        t1 = (k + 1) * dt_s
        i_k = float(currents_A[k])

        if k == 0:
            points.append((t0, i_k))

        points.append((t1, i_k))

        if k < T - 1:
            i_next = float(currents_A[k + 1])
            points.append((t1, i_next))

    return points


def write_pwlf_txt(path: Path, points: list[tuple[float, float]]) -> None:
    # No header, exactly two columns: time(seconds), current(amps)
    with open(path, "w") as f:
        for t_s, i_A in points:
            f.write(f"{t_s:.9f}\t{i_A:.12e}\n")


def write_summary_txt(
    path: Path,
    selected: list[int],
    sample_index: int,
    label: int,
    simulation_time_s: float,
    spike_times_by_file_neuron: dict[int, list[float]],
    spike_times_by_hidden_index: dict[int, list[float]],
) -> None:
    with open(path, "w") as f:
        f.write("EE 533 Task 5 PWLF Export Summary\n")
        f.write("=================================\n\n")

        f.write("Selected hidden-layer neuron indices:\n")
        f.write(", ".join(str(i) for i in selected) + "\n\n")

        f.write("PWLF file mapping:\n")
        for file_neuron, hidden_idx in enumerate(selected):
            f.write(f"neuron_{file_neuron}_pwlf.txt -> hidden neuron {hidden_idx}\n")
        f.write("\n")

        f.write("MNIST test image used:\n")
        f.write(f"test image index {sample_index}, digit label {label}\n\n")

        f.write("Python reference spike times, by PWLF file neuron, in seconds:\n")
        for file_neuron in range(4):
            times = spike_times_by_file_neuron[file_neuron]
            if times:
                times_str = ", ".join(f"{t:.6f}s" for t in times)
            else:
                times_str = "no spikes"
            f.write(f"Neuron {file_neuron} spikes at: {times_str}\n")
        f.write("\n")

        f.write("Python reference spike times, by original hidden-layer index, in seconds:\n")
        for hidden_idx in selected:
            times = spike_times_by_hidden_index[hidden_idx]
            if times:
                times_str = ", ".join(f"{t:.6f}s" for t in times)
            else:
                times_str = "no spikes"
            f.write(f"Hidden neuron {hidden_idx} spikes at: {times_str}\n")
        f.write("\n")

        f.write("Total simulation time window:\n")
        f.write(f"{simulation_time_s:.6f} seconds ({simulation_time_s * 1000:.3f} ms)\n")


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    ckpt_args = load_checkpoint_args(checkpoint, args)

    input_size = ckpt_args["input_resolution"] * ckpt_args["input_resolution"]

    model = CircuitInformedMNISTSNN(
        input_size=input_size,
        hidden_size=ckpt_args["hidden_size"],
        time_steps=ckpt_args["time_steps"],
        dt_ms=ckpt_args["dt_ms"],
        i_max_nA=ckpt_args["i_max_nA"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print("=" * 72)
    print("Loaded Section 4 checkpoint")
    print("=" * 72)
    print(f"checkpoint       : {checkpoint_path}")
    print(f"fc1.weight shape : {tuple(model.fc1.weight.shape)}")
    print(f"fc1.bias shape   : {tuple(model.fc1.bias.shape)}")
    print(f"hidden_size      : {ckpt_args['hidden_size']}")
    print(f"time_steps       : {ckpt_args['time_steps']}")
    print(f"input resolution : {ckpt_args['input_resolution']}x{ckpt_args['input_resolution']}")
    print(f"dt_ms            : {ckpt_args['dt_ms']}")
    print(f"I_max_nA         : {ckpt_args['i_max_nA']}")
    print("=" * 72)

    image, label = load_test_sample(
        data_dir=args.data_dir,
        input_resolution=ckpt_args["input_resolution"],
        sample_index=args.sample_index,
    )

    input_spikes = generate_input_spike_train(
        image=image,
        time_steps=ckpt_args["time_steps"],
        seed=args.seed + args.sample_index,
    )

    hidden_current_norm = compute_hidden_current_norm(model, input_spikes)

    # Convert Section 4 normalized currents into amps.
    # Then force the exact Cadence requested current range: 1 nA to 100 nA.
    raw_currents_A = hidden_current_norm * ckpt_args["i_max_nA"] * 1e-9
    exported_currents_A = torch.clamp(raw_currents_A, min=args.min_current_nA * 1e-9, max=args.max_current_nA * 1e-9)

    reference_spikes_all = simulate_reference_spikes_from_currents(
        currents_A=exported_currents_A,
        dt_ms=ckpt_args["dt_ms"],
        i_max_nA=ckpt_args["i_max_nA"],
    )

    selected = choose_neurons(
        exported_currents_A=exported_currents_A,
        reference_spikes=reference_spikes_all,
        requested_indices=args.neuron_indices,
        count=4,
    )

    dt_s = ckpt_args["dt_ms"] * 1e-3
    simulation_time_s = ckpt_args["time_steps"] * dt_s

    out_dir = Path(args.output_dir) / f"sample_{args.sample_index:04d}"
    pwlf_dir = out_dir / "pwlf_files"
    out_dir.mkdir(parents=True, exist_ok=True)
    pwlf_dir.mkdir(parents=True, exist_ok=True)

    save_input_image(image, label, out_dir / "mnist_input_image.png")

    # Save selected current plot
    time_ms = np.arange(ckpt_args["time_steps"]) * ckpt_args["dt_ms"]
    plt.figure(figsize=(7, 4.5))
    for file_neuron, hidden_idx in enumerate(selected):
        y_nA = exported_currents_A[:, hidden_idx].numpy() * 1e9
        plt.step(time_ms, y_nA, where="post", label=f"file neuron {file_neuron} = hidden {hidden_idx}")
    plt.xlabel("Time [ms]")
    plt.ylabel("Current [nA]")
    plt.title("Task 5 exported PWLF currents")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "task5_exported_currents.png", dpi=200)
    plt.close()

    spike_times_by_file_neuron = {}
    spike_times_by_hidden_index = {}

    # Write four required files: neuron_0_pwlf.txt ... neuron_3_pwlf.txt
    for file_neuron, hidden_idx in enumerate(selected):
        currents_np = exported_currents_A[:, hidden_idx].numpy()
        points = make_step_points_duplicate_time(currents_np, dt_s)

        pwlf_path = pwlf_dir / f"neuron_{file_neuron}_pwlf.txt"
        write_pwlf_txt(pwlf_path, points)

        spikes = reference_spikes_all[:, hidden_idx]
        spike_steps = torch.nonzero(spikes > 0.5, as_tuple=False).flatten().tolist()
        spike_times_s = [(step + 1) * dt_s for step in spike_steps]

        spike_times_by_file_neuron[file_neuron] = spike_times_s
        spike_times_by_hidden_index[hidden_idx] = spike_times_s

    # CSV summary, useful for the report.
    with open(out_dir / "task5_selected_neurons_and_spikes.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pwlf_file",
            "file_neuron_number",
            "hidden_neuron_index",
            "mnist_test_index",
            "mnist_digit_label",
            "simulation_time_s",
            "mean_current_nA",
            "min_current_nA",
            "max_current_nA",
            "python_spike_count",
            "python_spike_times_s",
        ])

        for file_neuron, hidden_idx in enumerate(selected):
            currents_nA = exported_currents_A[:, hidden_idx].numpy() * 1e9
            times = spike_times_by_hidden_index[hidden_idx]
            writer.writerow([
                f"neuron_{file_neuron}_pwlf.txt",
                file_neuron,
                hidden_idx,
                args.sample_index,
                label,
                f"{simulation_time_s:.9f}",
                f"{currents_nA.mean():.9f}",
                f"{currents_nA.min():.9f}",
                f"{currents_nA.max():.9f}",
                len(times),
                ";".join(f"{t:.9f}" for t in times),
            ])

    # Plain-English summary to send to the Cadence teammate.
    write_summary_txt(
        path=out_dir / "SEND_TO_CADENCE_PERSON_summary.txt",
        selected=selected,
        sample_index=args.sample_index,
        label=label,
        simulation_time_s=simulation_time_s,
        spike_times_by_file_neuron=spike_times_by_file_neuron,
        spike_times_by_hidden_index=spike_times_by_hidden_index,
    )

    # Save fc1 weights for transparency.
    torch.save(
        {
            "selected_hidden_neurons": selected,
            "fc1_weight_selected": model.fc1.weight.detach().cpu()[selected],
            "fc1_bias_selected": model.fc1.bias.detach().cpu()[selected],
            "mnist_sample_index": args.sample_index,
            "mnist_label": label,
            "time_steps": ckpt_args["time_steps"],
            "dt_ms": ckpt_args["dt_ms"],
        },
        out_dir / "task5_selected_fc1_weights.pt",
    )

    print("Selected hidden neurons:", selected)
    print(f"MNIST test image index : {args.sample_index}")
    print(f"MNIST digit label      : {label}")
    print(f"Simulation time        : {simulation_time_s:.6f} s ({simulation_time_s * 1000:.3f} ms)")
    print(f"Output folder          : {out_dir}")
    print("\nRequired PWLF files:")
    for file_neuron in range(4):
        print(f"  {pwlf_dir / f'neuron_{file_neuron}_pwlf.txt'}")
    print("\nPython reference spike times:")
    for file_neuron in range(4):
        times = spike_times_by_file_neuron[file_neuron]
        if times:
            print(f"  Neuron {file_neuron}: " + ", ".join(f"{t:.6f}s" for t in times))
        else:
            print(f"  Neuron {file_neuron}: no spikes")
    print("\nSummary file:")
    print(f"  {out_dir / 'SEND_TO_CADENCE_PERSON_summary.txt'}")
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact Task 5 PWLF files for Cadence."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="validation/section4_outputs/section4_circuit_snn_best.pt",
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="validation/section5_outputs")

    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--neuron-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional exact hidden neuron indices. Must provide exactly four if used.",
    )

    # Fallbacks if checkpoint has no saved args.
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--input-resolution", type=int, default=28)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--i-max-nA", type=float, default=100.0)

    parser.add_argument("--min-current-nA", type=float, default=1.0)
    parser.add_argument("--max-current-nA", type=float, default=100.0)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
