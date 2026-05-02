"""
parameter_extraction.py
-----------------------
Extracts circuit-informed neuron parameters from Cadence f-I curve data.

For a non-leaky integrate-and-fire neuron:
    f = I / (C_eff * delta_V)
    => slope (Hz/nA) = 1 / (C_eff * delta_V)
    => C_eff * delta_V = 1 / slope

We normalize voltage: V_th = 1.0 V, V_reset = 0.0 V  =>  delta_V = 1.0 V
This keeps the Python model dimensionless in voltage while preserving
physical current units (nA) and time (ms).

Outputs
-------
- C_EFF_F   : effective membrane capacitance in Farads
- SLOPE_HZ_PER_NA : fitted f-I slope in Hz/nA
- INTERCEPT_HZ    : fitted f-I intercept in Hz (should be ~0 for ideal I&F)
- A plot saved to validation/plots/fi_curve_fit.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ── paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
FI_CSV_PATH  = os.path.join(_PROJECT_DIR, "circuit_data", "fi_curve.csv")
PLOT_DIR     = os.path.join(_PROJECT_DIR, "validation", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── circuit voltage conventions (normalized) ──────────────────────────────────
V_TH    = 1.0   # threshold voltage  [V]  — normalized
V_RESET = 0.0   # reset voltage      [V]  — normalized
DELTA_V = V_TH - V_RESET  # = 1.0 V

# ── unit conversion helpers ───────────────────────────────────────────────────
NA_TO_A = 1e-9   # nA → A


def load_fi_data(path: str = FI_CSV_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load Cadence f-I CSV and return (currents_nA, frequencies_Hz)."""
    df = pd.read_csv(path)
    I  = df["input_current_nA"].to_numpy(dtype=float)
    f  = df["cadence_frequency_Hz"].to_numpy(dtype=float)
    return I, f


def fit_fi_curve(I_nA: np.ndarray, f_Hz: np.ndarray) -> dict:
    """
    Linear regression on f vs I.

    Returns a dict with:
        slope_Hz_per_nA  : f-I slope           [Hz/nA]
        intercept_Hz     : y-intercept          [Hz]
        r_squared        : goodness of fit
        C_eff_F          : effective capacitance [F]
        C_eff_pF         : effective capacitance [pF]
        V_th             : threshold voltage     [V]
        V_reset          : reset voltage         [V]
    """
    slope, intercept, r, _, _ = linregress(I_nA, f_Hz)
    r_squared = r ** 2

    # f = I / (C_eff * delta_V)  =>  C_eff = 1 / (slope_in_SI * delta_V)
    # slope [Hz/nA] → slope [Hz/A]:  divide by 1e-9  (since 1 nA = 1e-9 A)
    #   9.865 Hz/nA  =  9.865 / 1e-9 Hz/A  =  9.865e9 Hz/A
    slope_SI = slope / NA_TO_A          # Hz/A  (= 1/(A·s))
    C_eff_F  = 1.0 / (slope_SI * DELTA_V)

    return {
        "slope_Hz_per_nA" : slope,
        "intercept_Hz"    : intercept,
        "r_squared"       : r_squared,
        "C_eff_F"         : C_eff_F,
        "C_eff_pF"        : C_eff_F * 1e12,
        "V_th"            : V_TH,
        "V_reset"         : V_RESET,
    }


def plot_fi_curve(
    I_nA: np.ndarray,
    f_Hz: np.ndarray,
    params: dict,
    save: bool = True,
) -> None:
    """Plot Cadence data vs. linear fit and save to validation/plots/."""
    slope     = params["slope_Hz_per_nA"]
    intercept = params["intercept_Hz"]
    r2        = params["r_squared"]
    C_pF      = params["C_eff_pF"]

    I_fit = np.linspace(I_nA.min(), I_nA.max(), 200)
    f_fit = slope * I_fit + intercept

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(I_nA, f_Hz, color="royalblue", zorder=5, label="Cadence data")
    ax.plot(
        I_fit, f_fit,
        color="tomato", linewidth=2,
        label=f"Linear fit  (R² = {r2:.6f})",
    )
    ax.set_xlabel("Input Current  [nA]", fontsize=12)
    ax.set_ylabel("Firing Frequency  [Hz]", fontsize=12)
    ax.set_title("f-I Characteristic: Cadence vs. Linear Fit", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.35)

    textstr = (
        f"slope  = {slope:.4f} Hz/nA\n"
        f"C_eff  = {C_pF:.2f} pF\n"
        f"V_th   = {params['V_th']:.1f} V\n"
        f"V_reset= {params['V_reset']:.1f} V"
    )
    ax.text(
        0.97, 0.05, textstr,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    plt.tight_layout()
    if save:
        out = os.path.join(PLOT_DIR, "fi_curve_fit.png")
        plt.savefig(out, dpi=150)
        print(f"[parameter_extraction] Plot saved → {out}")
    plt.show()


def extract_parameters(verbose: bool = True) -> dict:
    """
    Full pipeline: load data → fit → (optionally) print & plot.

    Returns the parameter dict — import this function from other modules
    to share a single source of truth for neuron parameters.
    """
    I_nA, f_Hz = load_fi_data()
    params = fit_fi_curve(I_nA, f_Hz)

    if verbose:
        print("=" * 50)
        print("  Circuit Parameter Extraction  ")
        print("=" * 50)
        print(f"  f-I slope    : {params['slope_Hz_per_nA']:.4f}  Hz/nA")
        print(f"  Intercept    : {params['intercept_Hz']:.4f}  Hz")
        print(f"  R²           : {params['r_squared']:.8f}")
        print(f"  C_eff        : {params['C_eff_pF']:.4f}  pF")
        print(f"  V_th         : {params['V_th']:.2f}  V")
        print(f"  V_reset      : {params['V_reset']:.2f}  V")
        print("=" * 50)
        plot_fi_curve(I_nA, f_Hz, params)

    return params


# ── run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_parameters(verbose=True)