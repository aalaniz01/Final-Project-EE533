"""
neuron_model package
--------------------
Exports:
    - extract_parameters   : load Cadence f-I data, fit, return param dict
    - CircuitIFNeuron      : SpikingJelly-compatible non-leaky I&F neuron
"""

from neuron_model.parameter_extraction import extract_parameters

# CircuitIFNeuron requires torch + spikingjelly.
# Import is skipped gracefully if those are not installed,
# so parameter_extraction and simulate_fi still work standalone.
try:
    from neuron_model.circuit_if_neuron import CircuitIFNeuron
    __all__ = ["extract_parameters", "CircuitIFNeuron"]
except ImportError:
    __all__ = ["extract_parameters"]