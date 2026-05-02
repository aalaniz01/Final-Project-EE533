"""
circuit_if_neuron.py
--------------------
SpikingJelly-compatible integrate-and-fire neuron derived from Cadence
circuit parameters.

Architecture of the unit conversion pipeline
--------------------------------------------

SpikingJelly layer input (dimensionless weighted sum)
        │
        │  × I_scale_factor  [nA per unit]
        ▼
Physical input current  I_in  [nA]
        │
        │  Euler integration:  V_mem += (I_in * dt_ms*1e-3) / C_eff_F
        ▼                      (dt_ms in ms  →  ×1e-3 gives seconds)
Membrane voltage  V_mem  [V, normalized: 0 → V_th]
        │
        │  Threshold comparison: spike if V_mem >= V_th
        ▼
Spike  s ∈ {0, 1}
        │
        │  Hard reset:  V_mem ← V_reset  (where spike == 1)
        ▼
Updated  V_mem  (persistent state across time steps)

Key design choices
------------------
- Non-leaky I&F: no leak term, matching the Cadence circuit topology.
- Physical current units (nA) are used internally so the f-I relationship
  stays identical to Cadence.
- Normalized voltage (V_th = 1 V, V_reset = 0 V) decouples voltage from
  supply rail; only the ratio matters for firing rate.
- dt_ms defaults to 1 ms (matching Cadence transient resolution), but is
  a constructor argument for flexibility.
- I_max_nA (default 100 nA) defines the scaling from SpikingJelly's
  normalized input domain [0, 1] to physical current.  Set it as a
  constructor argument so it can be tuned without touching the class.
- Surrogate gradient (ATan) is registered so the neuron is fully
  compatible with SpikingJelly's BPTT / surrogate gradient training.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import base, surrogate

from neuron_model.parameter_extraction import extract_parameters

# ── extract circuit parameters once at import time ────────────────────────────
_PARAMS = extract_parameters(verbose=False)

# Physical constants extracted from Cadence fit
C_EFF_F  : float = _PARAMS["C_eff_F"]     # effective membrane capacitance [F]
V_TH     : float = _PARAMS["V_th"]        # threshold voltage  [V]
V_RESET  : float = _PARAMS["V_reset"]     # reset voltage      [V]


class CircuitIFNeuron(base.MemoryModule):
    """
    Non-leaky integrate-and-fire neuron parameterized from Cadence simulation.

    Parameters
    ----------
    dt_ms : float
        Simulation time step in milliseconds (default: 1.0 ms).
        Must match the time resolution used in your transient simulation.
    I_max_nA : float
        Maximum physical current in nA that corresponds to a normalized
        input of 1.0 from the SpikingJelly pipeline (default: 100 nA).
        Acts as a fixed scale factor: I_physical = x * I_max_nA.
    surrogate_function : surrogate.SurrogateFunctionBase
        Surrogate gradient function for backpropagation through spikes.
        Defaults to ATan (arctangent), a standard choice in SpikingJelly.
    v_th : float
        Threshold voltage override [V]. Defaults to value from Cadence fit.
    v_reset : float
        Reset voltage override [V]. Defaults to value from Cadence fit.
    C_eff_F : float
        Effective capacitance override [F]. Defaults to value from Cadence fit.

    State
    -----
    v : torch.Tensor
        Membrane voltage [V], shape matches input. Registered as persistent
        memory so SpikingJelly resets it correctly between samples.

    Usage
    -----
    >>> neuron = CircuitIFNeuron(dt_ms=1.0, I_max_nA=100.0)
    >>> # inside a SpikingJelly network forward pass (T time steps):
    >>> for t in range(T):
    ...     spike = neuron(x[t])   # x[t]: (batch, features), values in [0,1]
    """

    def __init__(
        self,
        dt_ms      : float = 1.0,
        I_max_nA   : float = 100.0,
        surrogate_function: surrogate.SurrogateFunctionBase = None,
        v_th       : float = V_TH,
        v_reset    : float = V_RESET,
        C_eff_F    : float = C_EFF_F,
    ):
        super().__init__()

        # ── time step (convert ms → s for SI integration) ──────────────────
        self.dt_s    = dt_ms * 1e-3          # [s]
        self.dt_ms   = dt_ms                 # stored for reference / reporting

        # ── current scaling ─────────────────────────────────────────────────
        self.I_max_nA = I_max_nA             # [nA]
        self.I_max_A  = I_max_nA * 1e-9      # [A]  used in integration

        # ── circuit parameters ───────────────────────────────────────────────
        self.v_th     = v_th
        self.v_reset  = v_reset
        self.C_eff_F  = C_eff_F

        # ── surrogate gradient ───────────────────────────────────────────────
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()
        self.surrogate_function = surrogate_function

        # ── membrane voltage state ───────────────────────────────────────────
        # Registered via SpikingJelly's MemoryModule so reset() clears it.
        self.register_memory("v", 0.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _scale_to_current(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized SpikingJelly input → physical current in Amperes."""
        return x * self.I_max_A              # (batch, features)  [A]

    def _integrate(self, I_A: torch.Tensor) -> torch.Tensor:
        """
        Euler integration step.

            delta_V = I * dt / C_eff

        Returns updated membrane voltage (not yet thresholded).
        """
        return self.v + I_A * self.dt_s / self.C_eff_F

    def _fire(self, v_new: torch.Tensor) -> torch.Tensor:
        """
        Threshold comparison with surrogate gradient.

        Returns spike tensor s ∈ {0, 1} (float, for gradient flow).
        The surrogate_function wraps the Heaviside step so gradients
        can propagate during backpropagation.
        """
        return self.surrogate_function(v_new - self.v_th)

    def _reset(self, v_new: torch.Tensor, spike: torch.Tensor) -> torch.Tensor:
        """
        Hard reset: V_mem ← V_reset wherever a spike occurred.

            v_after = v_new * (1 - spike) + v_reset * spike
        """
        return v_new * (1.0 - spike) + self.v_reset * spike

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        One time-step forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Normalized input from upstream SpikingJelly layers.
            Shape: (batch_size, n_neurons) or any broadcastable shape.
            Expected range: [0, 1]  (maps to [0, I_max_nA] nA).

        Returns
        -------
        spike : torch.Tensor
            Binary spike tensor, same shape as x, values in {0, 1}.
        """
        # 1. scale normalized input to physical current [A]
        I_A = self._scale_to_current(x)

        # 2. integrate: accumulate charge on membrane capacitor
        v_new = self._integrate(I_A)

        # 3. fire: threshold comparison (differentiable via surrogate)
        spike = self._fire(v_new)

        # 4. reset: clamp voltage back to V_reset at spike sites
        self.v = self._reset(v_new, spike)

        return spike

    # ── utility ───────────────────────────────────────────────────────────────

    def expected_firing_rate(self, I_nA: float) -> float:
        """
        Theoretical firing rate [Hz] for a constant input current.

            f = I / (C_eff * delta_V)

        Useful for sanity-checking against Cadence f-I data.

        Parameters
        ----------
        I_nA : float
            Input current in nanoamps.

        Returns
        -------
        float : expected firing rate in Hz.
        """
        I_A     = I_nA * 1e-9
        delta_V = self.v_th - self.v_reset
        return float(I_A / (self.C_eff_F * delta_V))

    def __repr__(self) -> str:
        return (
            f"CircuitIFNeuron("
            f"dt_ms={self.dt_ms}, "
            f"I_max_nA={self.I_max_nA}, "
            f"C_eff_pF={self.C_eff_F * 1e12:.4f}, "
            f"V_th={self.v_th}, "
            f"V_reset={self.v_reset})"
        )