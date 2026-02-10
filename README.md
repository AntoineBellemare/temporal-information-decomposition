# Temporal Information Decomposition

Analyze the temporal structure of time series using **Partial Information Decomposition (PID)** and **Integrated Information Decomposition (PhiID)**.

## Overview

This project applies information-theoretic decomposition methods to understand how information flows across time in various signals, from toy systems to real EEG data.

### Approach

**Temporal PID**: Decompose I(X_{t-lag1}, X_{t-lag2} → X_t) into:
- **Redundancy**: Information shared by both past time points
- **Synergy**: Information that requires BOTH past points together
- **Unique**: Information from each past point alone

**Temporal PhiID**: Use Takens delay embedding to construct a pseudo-bivariate system and decompose into 16 atoms capturing:
- Storage, Transfer, Copy, Erasure
- Upward and Downward causation
- Integrated information (Φ)

## Project Structure

```
temporal-information-decomposition/
├── scripts/
│   ├── pid/                    # PID analysis scripts
│   │   ├── toy_examples.py               # Toy systems (COPY, XOR, AR)
│   │   ├── kuramoto.py                   # Kuramoto oscillators
│   │   ├── eeg.py                        # Basic EEG analysis
│   │   ├── eeg_bandpass.py               # Frequency band EEG analysis
│   │   └── biosignal_comparison.py       # Cross-biosignal comparison
│   │
│   ├── phiid/                  # PhiID analysis scripts
│   │   ├── toy_examples.py               # Toy systems validation
│   │   ├── eeg_phiid_temporal.py         # EEG PhiID (frequency bands)
│   │   └── temporal_integration_index_eeg.py  # Temporal Integration Index
│   │
│   └── utils/                  # Shared utilities
│       └── estimate_tau.py               # Timescale estimation
│
├── results/
│   ├── pid/                    # PID analysis outputs
│   └── phiid/                  # PhiID analysis outputs
│
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # EEG and other data files
├── docs/                       # Documentation
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/temporal-information-decomposition.git
cd temporal-information-decomposition

# Create conda environment (recommended)
conda create -n phiid python=3.10
conda activate phiid

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run PID Analysis

```bash
# Toy systems validation
python scripts/pid/toy_examples.py

# Kuramoto oscillators
python scripts/pid/kuramoto.py

# EEG analysis (requires data in data/ folder)
python scripts/pid/eeg_bandpass.py
```

### Run PhiID Analysis

```bash
# Toy systems validation
python scripts/phiid/toy_examples.py

# EEG PhiID analysis (frequency bands with Takens embedding)
python scripts/phiid/eeg_phiid_temporal.py

# Temporal Integration Index
python scripts/phiid/temporal_integration_index_eeg.py
```

## Key Concepts

### PID (Partial Information Decomposition)
- Uses the `dit` library with MMI (Minimum Mutual Information) redundancy measure
- Analyzes 3→1 variable relationships: two lagged sources predicting a target
- 4 atoms: Redundancy, Unique₁, Unique₂, Synergy

### PhiID (Integrated Information Decomposition)
- Uses the `phyid` library with Takens delay embedding
- Constructs 4-vectors: [t, t+τ, t+2τ, t+3τ] where τ probes different timescales
- 16 atoms organized into information dynamics categories
- ACF-aware τ selection to avoid trivial autocorrelation
- Supports broadband (LRTC) and narrowband (oscillatory) analysis

### Key Findings from Toy Systems

| Process | PID Signature | PhiID Signature |
|---------|---------------|-----------------|
| COPY (x[t] = x[t-1]) | High redundancy | High storage (rtr, xtx, yty) |
| XOR (x[t] = x[t-1] ⊕ x[t-2]) | High synergy | High synergy atoms (sts, str) |
| AR(1) | Decaying redundancy | Decaying storage |
| AR(2) with complex roots | Oscillating patterns | Transfer between timescales |

## References

- Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information.
- Mediano, P. A., et al. (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition.
- Lizier, J. T. (2012). The local information dynamics of distributed computation in complex systems.
- Takens, F. (1981). Detecting strange attractors in turbulence.

## License

MIT License
