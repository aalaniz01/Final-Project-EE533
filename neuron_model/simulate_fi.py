"""
simulate_fi.py
--------------
Validates Task 2a: confirms the Python neuron reproduces the Cadence f-I curve.

What this script does
---------------------
1. Loads extracted circuit parameters (C_eff, V_th, V_reset) from
   parameter_extraction.py — single source of truth, no hardcoded values.
2. Implements the same non-leaky I&F integration as CircuitIFNeuron
   using pure NumPy (no torch/SpikingJelly needed to run this script).
3. For each input current in the Cadence dataset, simulates the neuron
   for a fixed duration, counts spikes, and computes firing rate.
4. Runs a finer current sweep for a smooth curve.
5. Plots and saves: Python f-I vs Cadence f-I.
6. Prints a quantitative agreement table.

Run from the project/ directory:
    python -m neuron_model.simulate_fi
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — saves plot without needing a display
import matplotlib.pyplot as plt

from neuron_model.parameter_extraction import extract_parameters, FI_CSV_PATH

# ── output path ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
PLOT_DIR     = os.path.join(_PROJECT_DIR, "validation", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── core simulation (NumPy, mirrors CircuitIFNeuron.forward exactly) ──────────

def simulate_neuron(
    I_nA       : float,
    duration_ms: float,
    dt_ms      : float,
    C_eff_F    : float,
    V_th       : float,
    V_reset    : float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a non-leaky I&F neuron for a constant input current.

    Euler integration each step:
        V_mem += (I_A * dt_s) / C_eff_F
    Spike when V_mem >= V_th, then hard-reset to V_reset.

    Parameters
    ----------
    I_nA        : input current in nanoamps
    duration_ms : total simulation time in milliseconds
    dt_ms       : time step in milliseconds
    C_eff_F     : effective membrane capacitance in Farads
    V_th        : threshold voltage [V]
    V_reset     : reset voltage [V]

    Returns
    -------
    times_ms  : array of time points [ms]
    v_trace   : membrane voltage trace [V]
    """
    I_A   = I_nA * 1e-9                    # nA → A
    dt_s  = dt_ms * 1e-3                   # ms → s
    n_steps = int(duration_ms / dt_ms)

    times_ms = np.arange(n_steps) * dt_ms
    v_trace  = np.zeros(n_steps)

    v = V_reset
    for i in range(n_steps):
        v += I_A * dt_s / C_eff_F          # integrate
        if v >= V_th:                       # threshold check
            v = V_reset                     # hard reset
            v_trace[i] = V_th              # record spike peak for plotting
        else:
            v_trace[i] = v

    return times_ms, v_trace


def count_spikes(
    I_nA       : float,
    duration_ms: float,
    dt_ms      : float,
    C_eff_F    : float,
    V_th       : float,
    V_reset    : float,
) -> int:
    """Count spikes fired during a constant-current simulation."""
    I_A     = I_nA * 1e-9
    dt_s    = dt_ms * 1e-3
    n_steps = int(duration_ms / dt_ms)

    v = V_reset
    spikes = 0
    for _ in range(n_steps):
        v += I_A * dt_s / C_eff_F
        if v >= V_th:
            v = V_reset
            spikes += 1
    return spikes


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(
    duration_ms: float = 1000.0,
    dt_ms      : float = 1.0,
) -> None:
    """
    Full f-I validation pipeline.

    Parameters
    ----------
    duration_ms : simulation duration per current level [ms] (default 1000 ms = 1 s)
    dt_ms       : time step [ms] (default 1 ms — matches Cadence transient)
    """

    # ── 1. load circuit parameters ────────────────────────────────────────────
    params  = extract_parameters(verbose=False)
    C_eff_F = params["C_eff_F"]
    V_th    = params["V_th"]
    V_reset = params["V_reset"]

    # ── 2. load Cadence data ──────────────────────────────────────────────────
    df         = pd.read_csv(FI_CSV_PATH)
    I_cadence  = df["input_current_nA"].to_numpy(float)
    f_cadence  = df["cadence_frequency_Hz"].to_numpy(float)

    # ── 3. simulate Python neuron at each Cadence current point ───────────────
    f_python = np.array([
        count_spikes(I, duration_ms, dt_ms, C_eff_F, V_th, V_reset)
        / (duration_ms * 1e-3)          # spikes / seconds = Hz
        for I in I_cadence
    ])

    # ── 4. finer sweep for smooth curve ──────────────────────────────────────
    I_sweep  = np.linspace(1, 100, 100)
    f_sweep  = np.array([
        count_spikes(I, duration_ms, dt_ms, C_eff_F, V_th, V_reset)
        / (duration_ms * 1e-3)
        for I in I_sweep
    ])

    # ── 5. quantitative agreement table ──────────────────────────────────────
    abs_err  = np.abs(f_cadence - f_python)
    rel_err  = abs_err / f_cadence * 100

    print("=" * 65)
    print("  Task 2a — Python Neuron vs Cadence f-I Validation")
    print("=" * 65)
    print(f"  Simulation: {duration_ms:.0f} ms total,  dt = {dt_ms} ms")
    print(f"  C_eff = {C_eff_F*1e12:.4f} pF  |  V_th = {V_th} V  |  V_reset = {V_reset} V")
    print("-" * 65)
    print(f"  {'I (nA)':>6}  {'f_Cadence (Hz)':>15}  {'f_Python (Hz)':>14}  {'Abs Err':>9}  {'Rel Err':>9}")
    print("-" * 65)
    for I, fc, fp, ae, re in zip(I_cadence, f_cadence, f_python, abs_err, rel_err):
        print(f"  {I:>6.0f}  {fc:>15.4f}  {fp:>14.4f}  {ae:>8.4f}  {re:>8.4f}%")
    print("-" * 65)
    print(f"  Mean absolute error : {abs_err.mean():.4f} Hz")
    print(f"  Mean relative error : {rel_err.mean():.4f} %")
    print(f"  Max relative error  : {rel_err.max():.4f} %")
    print("=" * 65)

    # ── 6. plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — left: f-I curve comparison —
    ax = axes[0]
    ax.plot(I_sweep, f_sweep, color="tomato", linewidth=2, label="Python neuron (sweep)")
    ax.scatter(I_cadence, f_cadence, color="royalblue", zorder=5, s=70, label="Cadence data")
    ax.scatter(I_cadence, f_python, color="tomato", marker="x", s=80,
               linewidths=2, zorder=6, label="Python neuron (at Cadence points)")
    ax.set_xlabel("Input Current  [nA]", fontsize=12)
    ax.set_ylabel("Firing Frequency  [Hz]", fontsize=12)
    ax.set_title("f-I Curve: Cadence vs. Python Neuron", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)

    # — right: error plot —
    ax2 = axes[1]
    ax2.bar(I_cadence, rel_err, color="steelblue", alpha=0.8, width=6)
    ax2.axhline(rel_err.mean(), color="tomato", linewidth=1.5,
                linestyle="--", label=f"Mean = {rel_err.mean():.2f}%")
    ax2.set_xlabel("Input Current  [nA]", fontsize=12)
    ax2.set_ylabel("Relative Error  [%]", fontsize=12)
    ax2.set_title("Python vs Cadence — Relative Error per Current", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35, axis="y")

    plt.suptitle("Task 2a — Circuit-Informed Python Neuron Validation", fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "task2a_fi_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {out}")

    # ── 7. bonus: membrane voltage trace at 50 nA ────────────────────────────
    _plot_voltage_trace(50.0, 50.0, dt_ms, C_eff_F, V_th, V_reset)


def _plot_voltage_trace(
    I_nA       : float,
    duration_ms: float,
    dt_ms      : float,
    C_eff_F    : float,
    V_th       : float,
    V_reset    : float,
) -> None:
    """Save a membrane voltage trace for a single current level."""
    times_ms, v_trace = simulate_neuron(
        I_nA, duration_ms, dt_ms, C_eff_F, V_th, V_reset
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(times_ms, v_trace, color="steelblue", linewidth=1.2)
    ax.axhline(V_th, color="tomato", linewidth=1.2, linestyle="--", label=f"V_th = {V_th} V")
    ax.set_xlabel("Time  [ms]", fontsize=12)
    ax.set_ylabel("Membrane Voltage  [V]", fontsize=12)
    ax.set_title(f"Membrane Voltage Trace  —  I_in = {I_nA} nA", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35)
    plt.tight_layout()

    out = os.path.join(PLOT_DIR, "task2a_voltage_trace_50nA.png")
    plt.savefig(out, dpi=150)
    print(f"  Voltage trace saved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run(duration_ms=1000.0, dt_ms=0.001)