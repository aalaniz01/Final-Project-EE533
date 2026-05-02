"""
section3_single_neuron_validation.py
-----------------------------------
EE 533 Final Project - Task 3: Single-Neuron Validation
Circuit vs Python model for:
  1) constant 10 nA input current
  2) pulsed 10 nA input current, 10 us pulse width, 20 us period

This script is designed to be placed in the repo under:
    validation/section3_single_neuron_validation.py

It uses the same circuit-informed integrate-and-fire equations as
neuron_model/circuit_if_neuron.py, but runs them with event-based timing so
microsecond pulses are resolved accurately for Section 3.

Expected repo layout:
    circuit_data/fi_curve.csv
    circuit_data/Iconst10nA.csv
    circuit_data/Ipulse10nA.csv

Run from repo root:
    python validation/section3_single_neuron_validation.py

Or run with custom CSV paths:
    python validation/section3_single_neuron_validation.py \
        --const_csv Iconst10nA.csv \
        --pulse_csv Ipulse10nA.csv \
        --fi_csv circuit_data/fi_curve.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


@dataclass
class FIParams:
    slope_hz_per_nA: float
    intercept_hz: float
    r_squared: float
    c_eff_F: float
    i_offset_nA: float
    v_th: float = 1.0
    v_reset: float = 0.0


def read_waveform_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a Cadence CSV whose first column is time and second column is vout."""
    df = pd.read_csv(path)
    time_s = df.iloc[:, 0].to_numpy(dtype=float)
    vout_V = df.iloc[:, 1].to_numpy(dtype=float)
    return time_s, vout_V


def extract_spike_times(
    time_s: np.ndarray,
    vout_V: np.ndarray,
    threshold_V: float = 1.0,
    refractory_s: float = 1e-3,
) -> np.ndarray:
    """
    Extract spike peak times from Cadence vout.
    A spike is detected when vout crosses above threshold_V. For each thresholded
    region, the peak time is used as the spike time.
    """
    above = vout_V > threshold_V
    starts = np.where(above & np.r_[True, ~above[:-1]])[0]

    spike_times = []
    for start in starts:
        stop = start
        while stop + 1 < len(vout_V) and above[stop + 1]:
            stop += 1
        peak_index = start + int(np.argmax(vout_V[start : stop + 1]))
        peak_time = float(time_s[peak_index])
        if not spike_times or peak_time - spike_times[-1] > refractory_s:
            spike_times.append(peak_time)

    return np.array(spike_times)


def fit_fi_curve(fi_csv: Path) -> FIParams:
    """Fit f = slope*I + intercept using the Cadence f-I data."""
    df = pd.read_csv(fi_csv)
    I_nA = df["input_current_nA"].to_numpy(dtype=float)
    f_Hz = df["cadence_frequency_Hz"].to_numpy(dtype=float)

    slope, intercept, r_value, _, _ = linregress(I_nA, f_Hz)
    c_eff_F = 1.0 / ((slope / 1e-9) * 1.0)  # delta_V = 1 V
    i_offset_nA = intercept / slope

    return FIParams(
        slope_hz_per_nA=float(slope),
        intercept_hz=float(intercept),
        r_squared=float(r_value**2),
        c_eff_F=float(c_eff_F),
        i_offset_nA=float(i_offset_nA),
    )


def cadence_metrics(spikes_s: np.ndarray, tstop_s: float) -> dict[str, float]:
    isi_s = np.diff(spikes_s)
    return {
        "spike_count": int(len(spikes_s)),
        "first_spike_ms": spikes_s[0] * 1e3 if len(spikes_s) else np.nan,
        "mean_isi_ms": np.mean(isi_s) * 1e3 if len(isi_s) else np.nan,
        "isi_std_ms": np.std(isi_s) * 1e3 if len(isi_s) else np.nan,
        "firing_rate_isi_Hz": 1.0 / np.mean(isi_s) if len(isi_s) else np.nan,
        "firing_rate_count_Hz": len(spikes_s) / tstop_s if tstop_s > 0 else np.nan,
    }


def simulate_constant_spikes(
    params: FIParams,
    I_nA: float,
    v_init: float,
    tstop_s: float,
) -> np.ndarray:
    """Event-based constant-current IF simulation."""
    I_eff_A = (I_nA + params.i_offset_nA) * 1e-9
    delta_v = params.v_th - params.v_reset
    q_th = params.c_eff_F * delta_v
    q_init = params.c_eff_F * max(0.0, params.v_th - v_init)

    first_spike_s = q_init / I_eff_A
    period_s = q_th / I_eff_A

    spikes = []
    t = first_spike_s
    while t <= tstop_s + 1e-15:
        spikes.append(t)
        t += period_s
    return np.array(spikes)


def simulate_pulsed_spikes(
    params: FIParams,
    I_nA: float,
    v_init: float,
    tstop_s: float,
    delay_s: float = 1e-6,
    width_s: float = 10e-6,
    period_s: float = 20e-6,
) -> np.ndarray:
    """
    Event-based IF simulation for a pulsed current source.
    The equivalent f-I offset current is added only while the input pulse is ON.
    This matches the measured behavior without adding spontaneous OFF-time charge.
    """
    I_eff_A = (I_nA + params.i_offset_nA) * 1e-9
    q_th = params.c_eff_F * (params.v_th - params.v_reset)
    q = params.c_eff_F * v_init

    spikes = []
    k = 0
    while True:
        on_start = delay_s + k * period_s
        if on_start > tstop_s:
            break
        on_end = min(on_start + width_s, tstop_s)

        q_end = q + I_eff_A * (on_end - on_start)
        if q_end >= q_th - 1e-30:
            spike_time = on_start + (q_th - q) / I_eff_A
            if spike_time <= tstop_s + 1e-15:
                spikes.append(float(spike_time))
            # Keep integrating for the remaining part of the same ON pulse after reset.
            q = I_eff_A * max(0.0, on_end - spike_time)
        else:
            q = q_end

        k += 1

    return np.array(spikes)


def compare_spikes(cadence_s: np.ndarray, python_s: np.ndarray) -> dict[str, float]:
    n = min(len(cadence_s), len(python_s))
    err_s = python_s[:n] - cadence_s[:n]

    cadence_rate = 1.0 / np.mean(np.diff(cadence_s)) if len(cadence_s) > 1 else np.nan
    python_rate = 1.0 / np.mean(np.diff(python_s)) if len(python_s) > 1 else np.nan

    return {
        "cadence_spikes": int(len(cadence_s)),
        "python_spikes": int(len(python_s)),
        "matched_spikes": int(n),
        "spike_time_MAE_ms": np.mean(np.abs(err_s)) * 1e3 if n else np.nan,
        "spike_time_RMSE_ms": np.sqrt(np.mean(err_s**2)) * 1e3 if n else np.nan,
        "spike_time_max_error_ms": np.max(np.abs(err_s)) * 1e3 if n else np.nan,
        "cadence_rate_Hz": cadence_rate,
        "python_rate_Hz": python_rate,
        "rate_error_percent": abs(python_rate - cadence_rate) / cadence_rate * 100.0,
    }


def plot_waveform_with_spikes(
    time_s: np.ndarray,
    vout_V: np.ndarray,
    cadence_spikes_s: np.ndarray,
    python_spikes_s: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(9, 4.8))
    plt.plot(time_s * 1e3, vout_V, linewidth=1.0, label="Cadence /vout")
    plt.scatter(cadence_spikes_s * 1e3, np.ones_like(cadence_spikes_s) * 1.0, s=22, label="Cadence spikes")
    plt.scatter(python_spikes_s * 1e3, np.ones_like(python_spikes_s) * 0.78, s=18, marker="x", label="Python spikes")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (V)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_spike_agreement(
    constant_cad_s: np.ndarray,
    constant_py_s: np.ndarray,
    pulse_cad_s: np.ndarray,
    pulse_py_s: np.ndarray,
    out_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    n1 = min(len(constant_cad_s), len(constant_py_s))
    n2 = min(len(pulse_cad_s), len(pulse_py_s))

    plt.scatter(constant_cad_s[:n1] * 1e3, constant_py_s[:n1] * 1e3, label="Constant 10 nA")
    plt.scatter(pulse_cad_s[:n2] * 1e3, pulse_py_s[:n2] * 1e3, label="Pulsed 10 nA")

    all_times_ms = np.concatenate([constant_cad_s[:n1], constant_py_s[:n1], pulse_cad_s[:n2], pulse_py_s[:n2]]) * 1e3
    lo = float(np.nanmin(all_times_ms)) - 2
    hi = float(np.nanmax(all_times_ms)) + 2
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, label="Ideal agreement")

    plt.xlabel("Cadence spike time (ms)")
    plt.ylabel("Python spike time (ms)")
    plt.title("Section 3 Spike-Time Agreement")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1] if "validation" in str(Path(__file__).resolve()) else Path.cwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--const_csv", type=Path, default=repo_root / "circuit_data" / "Iconst10nA.csv")
    parser.add_argument("--pulse_csv", type=Path, default=repo_root / "circuit_data" / "Ipulse10nA.csv")
    parser.add_argument("--fi_csv", type=Path, default=repo_root / "circuit_data" / "fi_curve.csv")
    parser.add_argument("--out_dir", type=Path, default=repo_root / "validation" / "plots")
    parser.add_argument("--vout_threshold", type=float, default=1.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_const, v_const = read_waveform_csv(args.const_csv)
    t_pulse, v_pulse = read_waveform_csv(args.pulse_csv)

    cadence_const = extract_spike_times(t_const, v_const, threshold_V=args.vout_threshold)
    cadence_pulse = extract_spike_times(t_pulse, v_pulse, threshold_V=args.vout_threshold)

    # FIX: guard against empty spike arrays before any indexing.
    # If no spikes are found, the most likely causes are: wrong --vout_threshold,
    # wrong CSV column order, or the simulation time being too short.
    if len(cadence_const) == 0:
        raise RuntimeError(
            "No spikes detected in the constant-current Cadence CSV.\n"
            f"  File      : {args.const_csv}\n"
            f"  Threshold : {args.vout_threshold} V\n"
            "Check that the CSV contains a vout column that actually crosses "
            "the threshold, or lower --vout_threshold."
        )
    if len(cadence_pulse) == 0:
        raise RuntimeError(
            "No spikes detected in the pulsed Cadence CSV.\n"
            f"  File      : {args.pulse_csv}\n"
            f"  Threshold : {args.vout_threshold} V\n"
            "Check that the CSV contains a vout column that actually crosses "
            "the threshold, or lower --vout_threshold."
        )

    tstop_s = max(float(t_const[-1]), float(t_pulse[-1]))
    params = fit_fi_curve(args.fi_csv)

    # FIX: warn if i_offset_nA has the wrong sign.
    # For a standard IF neuron the f-I intercept should be negative
    # (threshold current needed to fire), making i_offset_nA negative.
    # A positive i_offset_nA would artificially inflate the effective
    # current and make Python fire faster than Cadence.
    if params.i_offset_nA > 0:
        import warnings
        warnings.warn(
            f"i_offset_nA = {params.i_offset_nA:.4f} nA is positive. "
            "This means your f-I linear fit has a positive intercept, which "
            "will make the Python model fire faster than Cadence. "
            "Check your fi_curve.csv data and the fit in fit_fi_curve().",
            UserWarning,
            stacklevel=2,
        )

    # Use the first constant-current spike to estimate the Cadence initial condition.
    # This avoids unfairly penalizing Python for the simulator's nonzero startup state.
    I_active_nA = 10.0
    I_eff_A = (I_active_nA + params.i_offset_nA) * 1e-9
    v_init = params.v_th - (I_eff_A * cadence_const[0]) / params.c_eff_F
    v_init = float(np.clip(v_init, params.v_reset, params.v_th))

    python_const = simulate_constant_spikes(params, I_active_nA, v_init, tstop_s)
    python_pulse = simulate_pulsed_spikes(params, I_active_nA, v_init, tstop_s)

    cadence_summary = pd.DataFrame(
        [
            {"case": "Constant 10 nA", **cadence_metrics(cadence_const, tstop_s)},
            {"case": "Pulsed 10 nA, 10us width, 20us period", **cadence_metrics(cadence_pulse, tstop_s)},
        ]
    )

    validation_summary = pd.DataFrame(
        [
            {"case": "Constant 10 nA", **compare_spikes(cadence_const, python_const)},
            {"case": "Pulsed 10 nA, 10us width, 20us period", **compare_spikes(cadence_pulse, python_pulse)},
        ]
    )

    model_params = pd.DataFrame(
        [
            {
                "slope_Hz_per_nA": params.slope_hz_per_nA,
                "intercept_Hz": params.intercept_hz,
                "r_squared": params.r_squared,
                "C_eff_pF": params.c_eff_F * 1e12,
                "I_offset_nA": params.i_offset_nA,
                "v_init_for_validation_V": v_init,
            }
        ]
    )

    cadence_summary.to_csv(args.out_dir / "section3_cadence_metrics.csv", index=False)
    validation_summary.to_csv(args.out_dir / "section3_validation_metrics.csv", index=False)
    model_params.to_csv(args.out_dir / "section3_model_params.csv", index=False)

    plot_waveform_with_spikes(
        t_const,
        v_const,
        cadence_const,
        python_const,
        "Section 3: Constant 10 nA Input",
        args.out_dir / "section3_constant_spikes.png",
    )
    plot_waveform_with_spikes(
        t_pulse,
        v_pulse,
        cadence_pulse,
        python_pulse,
        "Section 3: Pulsed 10 nA Input, 10 us / 20 us",
        args.out_dir / "section3_pulsed_spikes.png",
    )
    plot_spike_agreement(
        cadence_const,
        python_const,
        cadence_pulse,
        python_pulse,
        args.out_dir / "section3_spike_time_agreement.png",
    )

    print("\n=== f-I extracted model parameters ===")
    print(model_params.to_string(index=False))
    print("\n=== Cadence extracted metrics ===")
    print(cadence_summary.to_string(index=False))
    print("\n=== Cadence vs Python validation ===")
    print(validation_summary.to_string(index=False))
    print(f"\nOutputs saved in: {args.out_dir}")


if __name__ == "__main__":
    main()