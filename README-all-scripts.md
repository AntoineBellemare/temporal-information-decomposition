# Temporal Information Decomposition Scripts

This directory contains scripts for analyzing temporal information structure using two complementary approaches:
- **PID** (Partial Information Decomposition): 4-atom decomposition of I(X_{t-lag1}, X_{t-lag2} → X_t)
- **PhiID** (Integrated Information Decomposition): 16-atom decomposition of bivariate time series

---

## Directory Structure

```
scripts/
├── pid/           # Partial Information Decomposition (3→1 variable)
│   ├── temporal_pid_analysis.py           # Toy systems (COPY, XOR, AR)
│   ├── kuramoto_temporal_pid.py           # Kuramoto oscillator analysis
│   ├── eeg_temporal_pid_analysis.py       # Basic EEG (20 channels, short lags)
│   ├── eeg_temporal_pid_extended.py       # Extended EEG (up to 1s lags)
│   ├── eeg_bandpass_temporal_pid.py       # EEG by frequency band
│   ├── eeg_bandpass_extended.py           # Bandpass + extended lags + prewhitening
│   └── biosignal_temporal_pid_comparison.py  # Cross-biosignal comparison (EEG, ECG, Resp)
│
├── phiid/         # Integrated Information Decomposition (bivariate)
│   ├── toy_phiid_analysis.py              # Toy system validation (v1)
│   ├── toy_phiid_analysis_v2.py           # Improved toy validation (fixes edge cases)
│   ├── eeg_phiid_temporal.py              # EEG temporal PhiID analysis
│   ├── eeg_lag_divergence_fast.py         # Fast EEG lag divergence (optimized)
│   ├── phiid_lag_divergence_analysis.py   # Lag divergence exploration
│   ├── temporal_integration_analysis.py   # Advanced integration metrics (v1)
│   ├── temporal_integration_v2.py         # Improved: normalization, surrogates (v2)
│   └── temporal_integration_v3.py         # Takens embedding approach (v3)
│
└── utils/         # Shared utilities
    └── estimate_tau.py                    # Characteristic timescale estimation
```

---

## PID Scripts

**Approach**: Decompose I(X_{t-lag1}, X_{t-lag2} → X_t) into:
- **Redundancy**: Shared predictive information from both lags
- **Synergy**: Information requiring BOTH lags together
- **Unique**: Information from each lag alone

### Scripts (in order of complexity):

1. **temporal_pid_analysis.py** - Toy system validation
   - COPY process, XOR process, AR(1), AR(2)
   - Verifies expected PID patterns with known ground truth
   
2. **kuramoto_temporal_pid.py** - Kuramoto oscillator dynamics
   - Simulates N coupled oscillators with varying coupling strength K
   - Predictions: Low K = high redundancy; Near-critical = enhanced synergy; High K = synchronized
   
3. **eeg_temporal_pid_analysis.py** - Basic EEG analysis
   - Discretizes continuous EEG and computes PID for lag pairs
   - Compares "temporal fingerprints" across brain regions
   
4. **eeg_temporal_pid_extended.py** - Extended lags (up to 1s)
   - Multiple binning strategies compared
   - Single-lag mutual information alongside PID
   - Downsampling option for slow dynamics

5. **eeg_bandpass_temporal_pid.py** - Frequency band analysis
   - Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-50 Hz)
   - Bandpass filters before computing PID
   
6. **eeg_bandpass_extended.py** - Full analysis pipeline
   - Bandpass filtering + extended lags up to 1 second
   - Prewhitening to isolate nonlinear structure
   
7. **biosignal_temporal_pid_comparison.py** - Cross-biosignal comparison
   - Compares EEG, ECG, and Respiration signals
   - Auto-estimates characteristic timescale (τ) for normalization
   - Generates "temporal information fingerprints" for fair cross-signal comparison

---

## PhiID Scripts

**Approach**: Decompose mutual information between two time series into 16 atoms.
For temporal analysis, uses signal vs time-shifted self as pseudo-bivariate system.

### Key Concepts:
- **16 atoms** (vs 4 in PID): finer decomposition of information flow
- **Grouped metrics**: Storage, Transfer, Copy, Erasure, Upward/Downward Causation
- **IIT metrics**: Integrated information (Φ), Causal density, Information storage

### Scripts (in order of complexity):

1. **toy_phiid_analysis.py** - Toy system validation (v1)
   - Binary: IID, COPY, XOR
   - Gaussian: IID, AR(1), AR(2), Oscillation
   - Validates PhiID expectations on known processes
   
2. **toy_phiid_analysis_v2.py** - Improved toy validation (v2)
   - Fixes: AND/OR collapse, XOR period-3 cycles, temporal overlap issues
   - Ensures `extra_lag > tau` to avoid spurious correlations

3. **eeg_phiid_temporal.py** - EEG temporal PhiID
   - Uses `calc_PhiID(src, tgt, tau)` from phyid library
   - `extra_lag` = timescale to probe; `tau` = embedding delay
   - Analyzes all 16 atoms across channels and brain regions

4. **eeg_lag_divergence_fast.py** - Fast EEG divergence analysis
   - Optimized for real-world EEG (300 Hz DSI-24)
   - Uses subset of data for speed with progress tracking
   
5. **phiid_lag_divergence_analysis.py** - Lag divergence exploration
   - When do metrics at different lags DIVERGE vs CONVERGE?
   - Reveals moments of high vs low temporal integration
   - Identifies state transitions and scale-invariant dynamics

6. **temporal_integration_analysis.py** - Advanced integration metrics (v1)
   - Temporal Integration Index (TII): scalar measure of scale-invariance
   - Dominant timescale detection
   - Cross-lag coupling analysis
   - Integration state segmentation via clustering
   - Complexity metrics (entropy of multi-scale dynamics)

7. **temporal_integration_v2.py** - Improved analysis (v2)
   - Normalizes metrics by lag to remove trivial scaling
   - Adds surrogate/baseline comparison (phase shuffle)
   - Z-score based TII formulation
   - Frequency band decomposition
   - Cross-channel synchrony analysis
   - Artifact detection

8. **temporal_integration_v3.py** - Takens embedding approach (v3)
   - Uses proper Takens delay embedding with direct 4-vector construction
   - **Key fix**: Regular sampling (τ, τ, τ) vs irregular (τ, extra_lag-τ, τ)
   - Timeline: t, t+τ, t+2τ, t+3τ (perfectly regular)
   - Bypasses `calc_PhiID` to use internal PhiID functions directly

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
python scripts/pid/temporal_pid_analysis.py

# PID: Full EEG bandpass analysis
python scripts/pid/eeg_bandpass_extended.py

# PID: Cross-biosignal comparison
python scripts/pid/biosignal_temporal_pid_comparison.py

# PhiID: Toy validation
python scripts/phiid/toy_phiid_analysis_v2.py

# PhiID: EEG temporal analysis  
python scripts/phiid/eeg_phiid_temporal.py

# PhiID: Temporal integration (latest version)
python scripts/phiid/temporal_integration_v3.py
```

---

## Results

All results are saved to:
```
results/
├── pid/
│   ├── toy_systems/           # Toy system validation
│   ├── kuramoto/              # Kuramoto oscillator results
│   ├── eeg_basic/             # Basic EEG analysis
│   ├── eeg_extended/          # Extended lag analysis
│   ├── eeg_bandpass/          # Frequency band analysis
│   ├── eeg_bandpass_extended/ # Full bandpass + extended
│   └── biosignal_comparison/  # Cross-biosignal comparison
│
└── phiid/
    ├── toy_systems/           # PhiID toy validation (v1)
    ├── toy_systems_v2/        # PhiID toy validation (v2)
    ├── eeg/                   # EEG PhiID results
    ├── eeg_divergence/        # EEG lag divergence
    ├── lag_divergence/        # Lag divergence analysis
    ├── temporal_integration/  # Integration analysis (v1)
    ├── temporal_integration_v2/  # Integration analysis (v2)
    └── temporal_integration_v3/  # Integration analysis (v3)
```

---

## Key References

- **PID**: Williams & Beer (2010) - Partial Information Decomposition
- **PhiID**: Mediano et al. (2021) - Integrated Information Decomposition
- **Temporal embedding**: Takens (1981) - Delay embedding theorem
