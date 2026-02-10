# Temporal Information Decomposition

Analyze the temporal structure of time series using **Partial Information Decomposition (PID)** and **Integrated Information Decomposition (PhiID)**.

## Overview

This project applies information-theoretic decomposition methods to understand how information flows across time in various signals, from toy systems to real EEG data.

### Approach

**Temporal PID**: Decompose I(X_{t-lag1}, X_{t-lag2} → X_t) into:
- **Redundancy**: Information shared by both past time points
- **Synergy**: Information that requires BOTH past points together
- **Unique**: Information from each past point alone

**Temporal PhiID**: Treat a signal as a pseudo-bivariate system (signal vs time-shifted self) and decompose into 16 atoms capturing:
- Storage, Transfer, Copy, Erasure
- Upward and Downward causation
- Integrated information (Φ)

## Project Structure

```
temporal-phiid/
├── scripts/
│   ├── pid/                    # PID analysis scripts
│   │   ├── temporal_pid_analysis.py      # Toy systems (COPY, XOR, AR)
│   │   ├── kuramoto_temporal_pid.py      # Kuramoto oscillators
│   │   ├── eeg_temporal_pid_analysis.py  # Basic EEG analysis
│   │   ├── eeg_temporal_pid_extended.py  # Extended lags (up to 1s)
│   │   ├── eeg_bandpass_temporal_pid.py  # Frequency band analysis
│   │   └── eeg_bandpass_extended.py      # Full analysis + prewhitening
│   │
│   └── phiid/                  # PhiID analysis scripts
│       ├── toy_phiid_analysis.py         # Toy systems validation
│       └── eeg_phiid_temporal.py         # EEG PhiID analysis
│
├── results/
│   ├── pid/
│   │   ├── toy_systems/        # Validation on COPY, XOR, AR
│   │   ├── kuramoto/           # Kuramoto oscillator results
│   │   ├── eeg_basic/          # Basic EEG (short lags)
│   │   ├── eeg_extended/       # Extended lags
│   │   ├── eeg_bandpass/       # Frequency band analysis
│   │   └── eeg_bandpass_extended/
│   │
│   └── phiid/
│       ├── toy_systems/        # PhiID toy validation
│       └── eeg/                # PhiID EEG results
│
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # EEG and other data files
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/antoinebellemare/temporal-phiid.git
cd temporal-phiid

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run PID Analysis

```bash
# Toy systems validation
python scripts/pid/temporal_pid_analysis.py

# Kuramoto oscillators
python scripts/pid/kuramoto_temporal_pid.py

# EEG analysis (requires data in data/ folder)
python scripts/pid/eeg_bandpass_extended.py
```

### Run PhiID Analysis

```bash
# Toy systems validation
python scripts/phiid/toy_phiid_analysis.py

# EEG PhiID analysis
python scripts/phiid/eeg_phiid_temporal.py
```

## Key Concepts

### PID (Partial Information Decomposition)
- Uses the `dit` library with MMI (Minimum Mutual Information) redundancy measure
- Analyzes 3→1 variable relationships: two lagged sources predicting a target
- 4 atoms: Redundancy, Unique₁, Unique₂, Synergy

### PhiID (Integrated Information Decomposition)
- Uses the `phyid` library
- Analyzes bivariate temporal relationships
- 16 atoms organized into information dynamics categories
- Computes IIT-related metrics (Φ, causal density, etc.)

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

## License

MIT License
