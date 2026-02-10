# Temporal Information Decomposition Scripts

This directory contains scripts for analyzing temporal information structure using two complementary approaches:
- **PID** (Partial Information Decomposition): 4-atom decomposition of I(X_{t-lag1}, X_{t-lag2} → X_t)
- **PhiID** (Integrated Information Decomposition): 16-atom decomposition using Takens delay embedding

---

## Directory Structure

```
scripts/
├── pid/                        # Partial Information Decomposition
│   ├── toy_examples.py                   # Toy systems (COPY, XOR, AR)
│   ├── kuramoto.py                       # Kuramoto oscillator analysis
│   ├── eeg.py                            # Basic EEG PID analysis
│   ├── eeg_bandpass.py                   # EEG by frequency band
│   └── biosignal_comparison.py           # Cross-biosignal comparison
│
├── phiid/                      # Integrated Information Decomposition
│   ├── toy_examples.py                   # Toy system validation
│   ├── eeg_phiid_temporal.py             # EEG PhiID (frequency bands)
│   └── temporal_integration_index_eeg.py # Temporal Integration Index
│
└── utils/                      # Shared utilities
    └── estimate_tau.py                   # Characteristic timescale estimation
```

---

## PID Scripts

**Approach**: Decompose I(X_{t-lag1}, X_{t-lag2} → X_t) into:
- **Redundancy**: Shared predictive information from both lags
- **Synergy**: Information requiring BOTH lags together
- **Unique**: Information from each lag alone

### Scripts:

1. **toy_examples.py** - Toy system validation
   - COPY process, XOR process, AR(1), AR(2)
   - Verifies expected PID patterns with known ground truth
   
2. **kuramoto.py** - Kuramoto oscillator dynamics
   - Simulates N coupled oscillators with varying coupling strength K
   - Explores how synchronization affects temporal information structure
   
3. **eeg.py** - Basic EEG analysis
   - Discretizes continuous EEG and computes PID for lag pairs
   - Compares "temporal fingerprints" across brain regions
   
4. **eeg_bandpass.py** - Frequency band analysis
   - Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
   - Bandpass filters before computing PID
   - Extended lags up to 1 second
   
5. **biosignal_comparison.py** - Cross-biosignal comparison
   - Compares EEG, ECG, and Respiration signals
   - Auto-estimates characteristic timescale (τ) for normalization
   - Generates "temporal information fingerprints"

---

## PhiID Scripts

**Approach**: Use Takens delay embedding to construct pseudo-bivariate system, then decompose mutual information into 16 atoms.

### Key Concepts:
- **Takens embedding**: Construct 4-vectors [t, t+τ, t+2τ, t+3τ]
- **16 atoms** (vs 4 in PID): finer decomposition of information flow
- **Grouped metrics**: Storage, Transfer, Copy, Erasure, Upward/Downward Causation
- **IIT metrics**: Integrated information (Φ), Causal density

### Scripts:

1. **toy_examples.py** - Toy system validation
   - Binary: IID, COPY, XOR
   - Gaussian: IID, AR(1), AR(2), Oscillation
   - Validates PhiID expectations on known processes

2. **eeg_phiid_temporal.py** - EEG PhiID by frequency band
   - **Takens embedding**: Regular sampling [t, t+τ, t+2τ, t+3τ]
   - **Frequency bands**: Broadband, Delta, Theta, Alpha, Beta, Gamma
   - **ACF-aware τ selection**: Filters out trivially autocorrelated timescales
   - **Broadband analysis**: Preserves LRTC (Long-Range Temporal Correlations)
   - **Multi-scale probing**: τ from 10ms to 16+ seconds
   - Outputs: 16 atoms, dynamics metrics, IIT measures

3. **temporal_integration_index_eeg.py** - Temporal Integration Index
   - Computes scale-invariant integration measure
   - Temporal Integration Index (TII): How information integrates across timescales
   - Surrogate comparison for significance testing
   - Segment-wise analysis for state transitions

---

## Utilities

### estimate_tau.py - Characteristic Timescale Estimation

Auto-estimate intrinsic timescale for normalized cross-signal comparisons.

**Methods available:**
- `τ_autocorr`: Time for autocorrelation to decay to 1/e
- `τ_halflife`: Time for autocorrelation to decay to 0.5
- `τ_zero`: First zero-crossing of autocorrelation
- `τ_period`: Dominant oscillation period (1/peak_frequency)
- `τ_integral`: Integral timescale (area under |ACF|)

**Usage:**
```python
from utils.estimate_tau import estimate_tau, TauEstimator, create_normalized_lags

# Quick estimate
tau = estimate_tau(signal, fs)

# Detailed estimates
estimator = TauEstimator(signal, fs)
print(estimator.summary())

# Create τ-normalized lag array
lags = create_normalized_lags(tau, fs, multiples=[0.5, 1, 2, 4, 8])
```

---

## Requirements

```bash
pip install dit phyid numpy pandas matplotlib seaborn scipy scikit-learn tqdm
```

---

## Usage Examples

```bash
# From project root

# PID: Toy system validation
python scripts/pid/toy_examples.py

# PID: EEG bandpass analysis
python scripts/pid/eeg_bandpass.py

# PID: Cross-biosignal comparison
python scripts/pid/biosignal_comparison.py

# PhiID: Toy validation
python scripts/phiid/toy_examples.py

# PhiID: EEG temporal analysis (frequency bands)
python scripts/phiid/eeg_phiid_temporal.py

# PhiID: Temporal Integration Index
python scripts/phiid/temporal_integration_index_eeg.py
```

---

## Results

All results are saved to:
```
results/
├── pid/                        # PID analysis outputs
│   └── [various subdirectories per analysis]
│
└── phiid/                      # PhiID analysis outputs
    └── [various subdirectories per analysis]
```

---

## Key References

- **PID**: Williams & Beer (2010) - Partial Information Decomposition
- **PhiID**: Mediano et al. (2021) - Integrated Information Decomposition
- **Takens embedding**: Takens (1981) - Delay embedding theorem
