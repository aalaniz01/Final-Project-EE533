"""
section4_train_mnist_circuit_snn.py
-----------------------------------
EE 533 Final Project - Section 4: SNN Design and Training

This script trains an MNIST SNN using the circuit-informed neuron from Task 2.

Expected repo layout:
Final-Project-EE533/
├── circuit_data/fi_curve.csv
├── neuron_model/circuit_if_neuron.py
├── neuron_model/parameter_extraction.py
├── snn/
└── validation/

Recommended location:
    snn/section4_train_mnist_circuit_snn.py

Run from the project root:
    python -m snn.section4_train_mnist_circuit_snn --hidden-size 128 --time-steps 10 --epochs 5

Quick sanity test:
    python -m snn.section4_train_mnist_circuit_snn --hidden-size 64 --time-steps 5 --epochs 1 --train-limit 5000 --test-limit 1000
"""

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from spikingjelly.activation_based import functional

from neuron_model import CircuitIFNeuron


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Model
# -----------------------------
class CircuitInformedMNISTSNN(nn.Module):
    """
    MNIST SNN with:
        input layer  -> one hidden layer -> output layer

    Input encoding:
        Pixel intensities are converted to Bernoulli spike trains.

    Hidden neuron:
        CircuitIFNeuron from the Task 2 Cadence-informed model.

    Output neuron:
        CircuitIFNeuron with 10 output neurons.
        Class prediction is based on output spike counts over time.
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

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Small positive-friendly initialization.

        The circuit-informed neuron models input as physical current.
        Current cannot be negative, so downstream currents are rectified
        before entering the neuron.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _current_normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Convert weighted sums into normalized current in [0, 1].

        This keeps the Python SNN aligned with the Task 2 assumption:
            normalized input 1.0 -> I_max_nA

        FIX: sigmoid replaces clamp(relu(x), 0, 1).
        clamp has zero gradient wherever |x| > 1 (which is most of the
        hidden layer after fc1 with Xavier init + sparse Bernoulli input),
        and relu zeroes the negative half as well — together they killed
        gradient flow to almost all neurons, causing all hidden-layer
        weights to stay identical and producing the same output for every
        neuron in Section 5.
        sigmoid maps ℝ → (0, 1) with non-zero gradient everywhere, so
        every neuron receives a useful update on every step.
        """
        return torch.sigmoid(x)

    @staticmethod
    def _poisson_encode(x: torch.Tensor) -> torch.Tensor:
        """
        Bernoulli/rate coding:
        higher pixel intensity means higher spike probability at each time step.
        """
        return (torch.rand_like(x) <= x).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            MNIST image tensor with shape [batch, 1, H, W].

        Returns
        -------
        spike_rate:
            Output spike counts divided by time steps.
            Shape: [batch, 10].
        """
        x = x.flatten(start_dim=1)

        output_spike_count = 0.0

        for _ in range(self.time_steps):
            input_spikes = self._poisson_encode(x)

            hidden_current = self._current_normalize(self.fc1(input_spikes))
            hidden_spikes = self.hidden_neuron(hidden_current)

            output_current = self._current_normalize(self.fc2(hidden_spikes))
            output_spikes = self.output_neuron(output_current)

            output_spike_count = output_spike_count + output_spikes

        return output_spike_count / self.time_steps


# -----------------------------
# Data
# -----------------------------
def make_dataloaders(
    data_dir: str,
    batch_size: int,
    input_resolution: int,
    train_limit: int | None,
    test_limit: int | None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((input_resolution, input_resolution)),
            transforms.ToTensor(),
        ]
    )

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    if train_limit is not None:
        train_set = Subset(train_set, range(min(train_limit, len(train_set))))

    if test_limit is not None:
        test_set = Subset(test_set, range(min(test_limit, len(test_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# -----------------------------
# Train / Evaluate
# -----------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            functional.reset_net(model)
            outputs = model(images)

            loss = F.cross_entropy(outputs, labels)

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.numel()
            total_loss += loss.item() * labels.numel()

            functional.reset_net(model)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_size = args.input_resolution * args.input_resolution

    train_loader, test_loader = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_resolution=args.input_resolution,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    model = CircuitInformedMNISTSNN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        time_steps=args.time_steps,
        dt_ms=args.dt_ms,
        i_max_nA=args.i_max_nA,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = output_dir / "section4_training_metrics.csv"

    print("=" * 70)
    print("EE 533 Section 4: Circuit-Informed SNN on MNIST")
    print("=" * 70)
    print(f"Device           : {device}")
    print(f"Input resolution : {args.input_resolution}x{args.input_resolution}")
    print(f"Hidden neurons   : {args.hidden_size}")
    print(f"Time steps       : {args.time_steps}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"dt_ms            : {args.dt_ms}")
    print(f"I_max_nA         : {args.i_max_nA}")
    print("=" * 70)

    rows = []

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            functional.reset_net(model)

            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.numel()
            total_loss += loss.item() * labels.numel()

            functional.reset_net(model)

            if args.print_every > 0 and batch_idx % args.print_every == 0:
                running_acc = total_correct / total_samples
                running_loss = total_loss / total_samples
                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx:04d}/{len(train_loader):04d} "
                    f"| loss={running_loss:.4f} | acc={running_acc:.4f}"
                )

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        test_loss, test_acc = evaluate(model, test_loader, device)

        # FIX: save checkpoint BEFORE updating best_test_acc.
        # Previously best_test_acc was updated first, making
        # `test_acc >= best_test_acc` always True and saving every epoch
        # instead of only the genuinely best checkpoint.
        model_path = output_dir / "section4_circuit_snn_best.pt"
        if test_acc >= best_test_acc:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "test_accuracy": test_acc,
                },
                model_path,
            )

        best_test_acc = max(best_test_acc, test_acc)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "best_test_accuracy": best_test_acc,
            "hidden_size": args.hidden_size,
            "time_steps": args.time_steps,
            "input_resolution": args.input_resolution,
            "dt_ms": args.dt_ms,
            "i_max_nA": args.i_max_nA,
        }
        rows.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
            f"| test_loss={test_loss:.4f} | test_acc={test_acc:.4f} "
            f"| best_test_acc={best_test_acc:.4f}"
        )

        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Save a simple plot after training.
    try:
        import matplotlib.pyplot as plt

        epochs = [r["epoch"] for r in rows]
        train_acc = [r["train_accuracy"] for r in rows]
        test_acc = [r["test_accuracy"] for r in rows]

        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, train_acc, marker="o", label="Train accuracy")
        plt.plot(epochs, test_acc, marker="o", label="Test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Section 4 MNIST Accuracy: Circuit-Informed SNN")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = output_dir / "section4_accuracy_curve.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

        print(f"Saved plot   : {plot_path}")
    except Exception as exc:
        print(f"Could not save plot: {exc}")

    print(f"Saved metrics: {metrics_path}")
    print(f"Best test accuracy: {best_test_acc:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Section 4 circuit-informed SNN on MNIST.")

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./validation/section4_outputs")

    parser.add_argument("--hidden-size", type=int, default=128, choices=[512, 256, 128, 64, 32])
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--input-resolution", type=int, default=28)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--i-max-nA", type=float, default=100.0)

    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print-every", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())