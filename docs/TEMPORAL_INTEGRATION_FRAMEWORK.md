# Temporal Integration Framework for PhiID Analysis

## Overview

This framework analyzes **temporal integration** in biosignals by examining how Integrated Information Decomposition (PhiID) metrics behave across different time lags. Rather than computing PhiID at a single temporal scale, we embed the signal at multiple lag values and ask: *Do information dynamics converge (scale-invariant) or diverge (multi-scale) across timescales?*

---

## 1. Core Concept: Temporal Integration Index (TII)

### Definition

The **Temporal Integration Index** measures how unified or fragmented a signal's information dynamics are across timescales:

$$TII(t) = \frac{1}{1 + D_{norm}(t)}$$

Where:
- $D_{norm}(t)$ = normalized divergence across lags at time $t$
- TII ∈ [0, 1]: Higher = more integrated (scale-invariant)

### Interpretation

| TII Value | Interpretation |
|-----------|----------------|
| **~1.0** | Scale-invariant: same information dynamics at all lags |
| **~0.5** | Mixed: some scale-specific processing |
| **~0.0** | Multi-scale: completely different dynamics at each lag |

---

## 2. Information Dynamics Metrics (from PhiID)

PhiID decomposes temporal mutual information into 16 atoms. We aggregate these into 6 **Information Dynamics** metrics:

| Metric | PhiID Atoms | Interpretation |
|--------|-------------|----------------|
| **Storage** | rtr + xtx + yty + sts | Information maintained over time |
| **Copy** | xtx + yty | Information copied from past to future |
| **Transfer** | xty + ytx | Information transferred between processes |
| **Erasure** | rtx + rty | Information lost/overwritten |
| **Upward Causation** | xts + yts + rts | Micro→Macro: parts influence whole |
| **Downward Causation** | stx + sty + str | Macro→Micro: whole influences parts |

### PhiID Atom Notation
- **r** = Redundancy (shared information)
- **x, y** = Unique information in X or Y
- **s** = Synergy (emergent information)
- **First letter** = Source type, **Second letter** = Target type

---

## 3. Multi-Lag Embedding

### Standard PhiID vs Temporal Self-PhiID

**Standard PhiID** analyzes two different signals (X, Y):
```
PhiID(X, Y, tau) → How do X and Y share/transfer information?
```

**Temporal Self-PhiID** (our approach) analyzes one signal at two time points:
```
PhiID(signal_early, signal_late, tau) → How does the signal relate to itself across time?
```

### Our Implementation

The `calc_PhiID(src, trg, tau)` function internally creates 4 vectors:

```python
# Inside calc_PhiID:
src_past   = src[:-tau]     # p1
trg_past   = trg[:-tau]     # p2  
src_future = src[tau:]      # t1
trg_future = trg[tau:]      # t2
```

**Our calling code:**
```python
segment = signal[start:end]
src = segment[:-extra_lag]    # Signal from t=0 to t=window
tgt = segment[extra_lag:]     # Signal from t=extra_lag to t=window+extra_lag

calc_PhiID(src, tgt, tau)
```

This creates the following temporal structure:

```
Timeline:
|----tau----|          |----tau----|
[src_past]  [src_fut]  [tgt_past]  [tgt_fut]
     ↑          ↑          ↑          ↑
  t-tau        t      t+lag-tau    t+lag

Where:
- src_past   = signal[t - tau]         (X_past)
- src_future = signal[t]               (X_future) 
- tgt_past   = signal[t + lag - tau]   (Y_past)
- tgt_future = signal[t + lag]         (Y_future)
```

### Why This Is Valid

1. **PhiID doesn't require X≠Y**: The math works for any two time series, including shifted versions of the same signal

2. **Temporal self-information is meaningful**: 
   - Storage: How much does signal(t) predict signal(t)?
   - Transfer: How does signal(t-lag) influence signal(t)?
   - Synergy/Redundancy: Emergent vs shared temporal structure

3. **Lag controls temporal scale**:
   - Small lag (10ms): Fine temporal structure
   - Large lag (200ms): Coarse temporal structure

### Alternative Approaches

| Approach | src | tgt | Measures |
|----------|-----|-----|----------|
| **Our method** | signal[0:T] | signal[lag:T+lag] | Temporal self-structure at scale `lag` |
| **Bivariate** | channel_X | channel_Y | Cross-channel information sharing |
| **Delay embedding** | signal[t] | signal[t-τ:t-nτ] | Attractor reconstruction |
| **Multiscale** | filtered(slow) | filtered(fast) | Cross-frequency coupling |

### Validation

Our approach is similar to **Temporal Information Dynamics** (Lizier et al.), which decomposes:
- **Active Information Storage**: How past predicts future (our Storage)
- **Transfer Entropy**: How one process predicts another (our Transfer)

The key insight: When src and tgt are shifted versions of the same signal, PhiID measures **temporal autocorrelation structure** decomposed into redundancy, unique, and synergistic components.

### Divergence Computation

1. **Z-normalize** each lag's timeseries (removes amplitude differences)
2. **Compute std** across lags at each timepoint = divergence
3. **Convert to TII**: TII = 1 / (1 + divergence)

---

## 4. Cross-Scale Analysis Methods

### 4.1 Cross-Scale Prediction (Correlation-based)

**Question**: Can short-lag dynamics predict long-lag dynamics?

**Method**: 
- Take shortest lag (10ms) and longest lag (200ms) timeseries
- Compute lagged cross-correlation at offsets [-10, +10] windows
- Find offset with maximum correlation

**Output**:
- `best_offset`: Temporal lead/lag in windows
- `best_correlation`: Strength of relationship

**Interpretation**:
| Offset | Meaning |
|--------|---------|
| 0 | Synchronous - timescales coupled |
| Negative | Long-lag leads (top-down, slow→fast) |
| Positive | Short-lag leads (bottom-up, fast→slow) |

---

### 4.2 Granger Causality

**Question**: Does knowing the *history* of one timescale improve prediction of the other?

**Method**:
- Autoregressive model: predict Y(t) from Y(t-1), ..., Y(t-k)
- Unrestricted model: add X(t-1), ..., X(t-k)
- F-test: does unrestricted model significantly reduce error?

**Output**:
- `short_to_long`: F-statistic for fast→slow causation
- `long_to_short`: F-statistic for slow→fast causation
- `direction`: Which timescale "leads" causally

**Formula**:
$$F = \frac{(RSS_{restricted} - RSS_{unrestricted}) / df_1}{RSS_{unrestricted} / df_2}$$

**Interpretation**:
| Result | Meaning |
|--------|---------|
| High F(short→long) | Fast dynamics drive slow dynamics (bottom-up) |
| High F(long→short) | Slow dynamics drive fast dynamics (top-down) |
| Both high | Bidirectional coupling |

---

### 4.3 Phase Coupling (PLV)

**Question**: Are the oscillations of different timescales phase-locked?

**Method**:
1. Apply Hilbert transform to extract instantaneous phase
2. Compute Phase Locking Value (PLV):
$$PLV = \left| \frac{1}{N} \sum_{t=1}^{N} e^{i(\phi_{short}(t) - \phi_{long}(t))} \right|$$
3. Compute mean phase difference

**Output**:
- `plv`: Phase locking value (0-1)
- `mean_phase_diff_deg`: Average phase difference in degrees
- `phase_direction`: Which timescale leads in phase

**Interpretation**:
| PLV | Meaning |
|-----|---------|
| ~1.0 | Perfect phase synchrony |
| ~0.5 | Partial coupling |
| ~0.0 | No phase relationship |

| Phase Diff | Meaning |
|------------|---------|
| 0° | In-phase (synchronous) |
| 90° | Quadrature (one leads by ¼ cycle) |
| 180° | Anti-phase (opposite) |

---

## 5. Surrogate Baseline Comparison

### Purpose
Determine if observed integration is *real* or just an artifact of spectral properties.

### Method: Phase Shuffling
1. FFT the signal
2. Randomize phases while preserving magnitude spectrum
3. Inverse FFT
4. Repeat N times (typically 3-10 surrogates)

### Significance Testing
- Compute divergence for real data and surrogates
- Z-score: $z = \frac{D_{real} - \mu_{surrogate}}{\sigma_{surrogate}}$

| Z-score | Interpretation |
|---------|----------------|
| z < -2 | **Significantly integrated** (more unified than chance) |
| -2 < z < 2 | Not significantly different from null |
| z > 2 | **Significantly fragmented** (more divergent than chance) |

---

## 6. Implementation Details

### File Structure
```
scripts/phiid/temporal_integration_v2.py
├── DYNAMICS_GROUPS          # 6 metric definitions
├── compute_phiid_for_lag()  # PhiID at single lag
├── compute_normalized_divergence()  # TII computation
├── compute_cross_scale_prediction() # Lagged correlation
├── compute_granger_causality()      # F-test causality
├── compute_phase_coupling()         # PLV analysis
├── create_surrogate()       # Phase-shuffle null model
├── analyze_with_surrogate() # Full analysis pipeline
└── plot_advanced_analysis() # 12-panel visualization
```

### Parameters
```python
FS = 300                    # Sampling rate (Hz)
LAGS_SAMPLES = [3, 9, 15, 30, 60]  # ~10, 30, 50, 100, 200 ms
WINDOW_SAMPLES = 150        # 500ms analysis window
STEP_SAMPLES = 30           # 100ms step (80% overlap)
TAU = 1                     # Embedding delay (samples)
N_SURROGATES = 3            # Surrogate count
```

### Output: 12-Panel Figure (per metric, per channel)

| Panel | Content |
|-------|---------|
| 1 | TII timeseries with 3-state clustering |
| 2 | TII distribution |
| 3 | Divergence Z-score vs surrogate |
| 4 | **Granger causality F-statistics** |
| 5 | Cross-lag correlation matrix |
| 6 | **Phase Locking Value timeseries** |
| 7 | Cross-scale prediction (original) |
| 8 | **Phase difference histogram** |
| 9 | Lag gradient distribution |
| 10 | Raw metric by lag |
| 11 | Real vs surrogate distribution |
| 12 | Summary statistics |

---

## 7. What's Novel About This Framework

### 7.1 Multi-Scale PhiID
Traditional PhiID uses a single temporal lag. We extend this to **multiple lags simultaneously**, revealing how information decomposition varies across timescales.

### 7.2 Temporal Integration Index
The TII provides a single number (0-1) summarizing how "unified" a signal's dynamics are across time. This is analogous to IIT's Φ but for temporal rather than spatial integration.

### 7.3 Three Complementary Cross-Scale Perspectives

| Method | Measures | Strength |
|--------|----------|----------|
| Cross-Scale Prediction | Linear correlation | Simple, interpretable |
| Granger Causality | Causal direction | Determines driver vs receiver |
| Phase Coupling | Oscillatory synchrony | Captures non-amplitude relationships |

### 7.4 Surrogate-Validated Significance
By comparing to phase-shuffled surrogates, we ensure that observed integration/fragmentation is not just an artifact of the signal's power spectrum.

### 7.5 Metric-Specific Temporal Fingerprints
Different metrics show different temporal patterns:
- **Storage/Copy**: Often scale-invariant (high TII)
- **Transfer/Erasure**: Often scale-specific (low TII)
- **Upward/Downward Causation**: Show asymmetric cross-scale interactions

---

## 8. Theoretical Considerations & Caveats

### 8.1 Temporal Self-PhiID: Is It Valid?

**Concern**: PhiID was designed for two distinct processes. Using shifted versions of the same signal might violate assumptions.

**Response**: The approach is mathematically valid because:
1. PhiID only requires two time series with shared temporal samples
2. The decomposition measures information relationships, regardless of source
3. This is analogous to **autocorrelation** but for information structure

**Interpretation shift**: 
- Standard PhiID: "How do X and Y share information?"
- Temporal Self-PhiID: "How does the signal's information structure change over lag?"

### 8.2 What the Atoms Mean in Temporal Context

| Standard PhiID | Temporal Self-PhiID |
|----------------|---------------------|
| Redundancy (X,Y share) | Temporal stability (persists across lag) |
| Unique X | Information specific to early timepoints |
| Unique Y | Information specific to late timepoints |
| Synergy | Emergent temporal patterns (not in either alone) |

### 8.3 Known Limitations

1. **Gaussian assumption**: We use `kind='gaussian'`, assuming linear relationships. Non-linear dynamics may be missed.

2. **Stationarity**: PhiID assumes stationary statistics within each window. Fast transients may violate this.

3. **Lag interpretation**: Our "lag" is the temporal offset between src and tgt, NOT the same as `tau` (embedding delay within each).

4. **Sample size**: Short windows (500ms = 150 samples) may give noisy estimates.

### 8.4 Alternative Approaches to Consider

| Alternative | Pros | Cons |
|-------------|------|------|
| **Bandpass then PhiID** | Clean frequency separation | Loses transients |
| **Wavelet + PhiID** | Time-frequency resolution | Computationally expensive |
| **Bivariate (cross-channel)** | Standard interpretation | Doesn't capture temporal self-structure |
| **Delay embedding (Takens)** | Attractor reconstruction | Different theoretical framework |

### 8.5 When Our Approach Works Best

✅ **Good for**:
- Signals with rich temporal autocorrelation (EEG, heartbeat)
- Questions about scale-invariance vs multi-scale structure
- Comparing integration across conditions/channels

⚠️ **Caution with**:
- White noise or near-random signals (no structure to decompose)
- Very short recordings (< 10 seconds)
- Highly non-stationary signals (use shorter windows)

---

## 9. Interpretation Guide

### High TII + High Coupling + Negative Z-score
→ **Integrated dynamics**: The signal maintains consistent information structure across timescales, more than expected by chance.

### Low TII + Low Coupling + Positive Z-score
→ **Fragmented dynamics**: Different timescales process information independently.

### Asymmetric Granger + Non-zero Phase Difference
→ **Hierarchical processing**: One timescale drives the other, suggesting top-down or bottom-up causation.

### Storage high TII, Transfer low TII
→ **Memory is scale-invariant, flow is scale-specific**: The system maintains memory across timescales but processes information transfers at each scale independently.

---

## 10. Example Results

From DSI-24 EEG analysis:

| Metric | Mean TII | Cross-Lag Coupling | Dominant Direction |
|--------|----------|-------------------|-------------------|
| Storage | 0.80 | 0.89 | Synchronous (offset ~0) |
| Copy | 0.76 | 0.81 | Synchronous |
| Transfer | 0.60 | 0.13 | Bottom-up (offset +6) |
| Erasure | 0.60 | 0.13 | Bottom-up (offset +6) |
| Upward | 0.67 | 0.53 | Top-down (offset -3) |
| Downward | 0.67 | 0.54 | Bottom-up (offset +10) |

**Key finding**: Memory processes (Storage, Copy) are scale-invariant, while information flow (Transfer, Erasure) and causal processes show scale-specific, asymmetric dynamics.

---

## 11. References

- Mediano, P.A.M., et al. (2021). *Integrated information decomposition*
- Lizier, J.T. (2014). *JIDT: An information-theoretic toolkit*
- Schreiber, T. (2000). *Measuring information transfer*
- Lachaux, J.P., et al. (1999). *Measuring phase synchrony in brain signals*

---

## Citation

If you use this framework, please cite:
```
Temporal Integration Framework for PhiID Analysis
https://github.com/[your-repo]/temporal-phiid
```

---

*Document generated: February 2026*
*Framework version: 2.0*
