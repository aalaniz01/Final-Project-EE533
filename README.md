# EE 533 – Neuromorphic Circuit Project

## Project Structure

```
project/
├── circuit_data/
│   └── fi_curve.csv                  # Cadence f-I curve export
│
├── neuron_model/
│   ├── __init__.py
│   ├── parameter_extraction.py       # Task 2a: fit f-I, extract C_eff
│   └── circuit_if_neuron.py          # Task 2b/2c: SpikingJelly neuron class
│
├── validation/                       # Task 3 (single-neuron validation)
│   └── plots/
│
├── snn/                              # Task 4 (MNIST SNN)
│
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Task 2 – Circuit-Informed Neuron Model

### Step 1: Extract parameters from Cadence data
```bash
cd project
python -m neuron_model.parameter_extraction
```
Outputs fitted slope, C_eff, R², and saves `validation/plots/fi_curve_fit.png`.

### Step 2: Use the neuron in your own script
```python
from neuron_model import CircuitIFNeuron
import torch

neuron = CircuitIFNeuron(dt_ms=1.0, I_max_nA=100.0)
print(neuron)

# Sanity check: expected firing rate at 50 nA
print(neuron.expected_firing_rate(50.0))  # should be ~513 Hz

# One forward step
x = torch.tensor([[0.5, 0.2, 0.8]])  # normalized input, shape (1, 3)
spike = neuron(x)
print(spike)
```

## Key Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| V_th | 1.0 V | Normalized — only ratio V_th/V_reset matters |
| V_reset | 0.0 V | Normalized |
| C_eff | extracted via linear regression on f-I data | ~101 pF |
| dt | 1.0 ms (default) | Matches Cadence transient resolution |
| I_max_nA | 100 nA (default, constructor arg) | Maps SpikingJelly input [0,1] → [0, 100 nA] |
| Surrogate | ATan | Standard SpikingJelly choice for BPTT |