# Alternative Approaches for Temporal PhiID Analysis

## Overview

This document details 5 alternative approaches for computing PhiID on temporal information dynamics within a single signal. Each approach addresses different aspects of temporal structure and has distinct advantages.

---

## Current Approach: Shifted Segment PhiID

### Implementation
```python
segment = signal[start:end]
src = segment[:-extra_lag]    # Early segment
tgt = segment[extra_lag:]     # Late segment (shifted by extra_lag)
calc_PhiID(src, tgt, tau)
```

### Resulting Temporal Structure
```
Timeline positions of the 4 PhiID vectors:

src_past:   t
src_future: t + tau
tgt_past:   t + extra_lag  
tgt_future: t + extra_lag + tau

Example with tau=1, extra_lag=30:
Positions: [t, t+1, t+30, t+31]
Spacing:   1, 29, 1  ← IRREGULAR!
```

### The Two-Lag Problem

**Issue**: We have TWO temporal parameters:
1. `tau` - PhiID's internal embedding delay (past→future within each process)
2. `extra_lag` - Our external offset between "src" and "tgt" processes

**Consequences**:
- Irregular temporal sampling: gaps of τ, then (lag-τ), then τ
- Confounded interpretation: which lag drives which effect?
- If τ << extra_lag, the two "processes" are almost independent windows
- If τ ≈ extra_lag, there's temporal overlap causing redundancy

**When it's OK**:
- When we're explicitly asking "how do two temporal windows relate?"
- When τ is very small (τ=1) and just creates past/future within each window
- When we only compare ACROSS different extra_lag values (relative comparison)

**When it's problematic**:
- Trying to interpret absolute atom values
- Comparing to standard bivariate PhiID literature
- When τ and extra_lag are similar magnitude

---

## Approach 1: Cross-Frequency PhiID

### Concept
Decompose signal into frequency bands, then compute PhiID between bands. This tests how different frequency components share/transfer information.

### Implementation
```python
from scipy.signal import butter, sosfilt

def bandpass(signal, low, high, fs, order=4):
    """Bandpass filter using Butterworth filter."""
    nyq = fs / 2
    sos = butter(order, [low/nyq, high/nyq], btype='band', output='sos')
    return sosfilt(sos, signal)

# Define canonical frequency bands
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (12, 30),
    'gamma': (30, 50)
}

# Extract bands
signal_delta = bandpass(signal, 1, 4, fs)
signal_alpha = bandpass(signal, 8, 12, fs)
signal_gamma = bandpass(signal, 30, 50, fs)

# PhiID between frequency bands
# Example: How do slow oscillations (delta) relate to fast (gamma)?
atoms_delta_gamma, _ = calc_PhiID(signal_delta, signal_gamma, tau)

# Or: alpha-gamma coupling (often studied in neuroscience)
atoms_alpha_gamma, _ = calc_PhiID(signal_alpha, signal_gamma, tau)
```

### What It Measures
- **Cross-frequency coupling (CFC)** decomposed into information atoms
- Traditional CFC only measures coupling strength; PhiID decomposes it into:
  - Redundancy: Both bands carry same information
  - Unique slow: Information only in slow oscillations
  - Unique fast: Information only in fast oscillations
  - Synergy: Emergent patterns requiring both (true CFC)

### Interpretation of Atoms

| Atom | Cross-Frequency Meaning |
|------|------------------------|
| **rtr** | Redundant information stable in both bands |
| **xtx** | Slow-band specific information preserved |
| **yty** | Fast-band specific information preserved |
| **sts** | Synergistic cross-frequency pattern |
| **xty + ytx** | Transfer between frequency bands |
| **xts + yts** | Parts (bands) → Emergent pattern |
| **stx + sty** | Emergent pattern → Modulates bands |

### Advantages
- Clean frequency separation (no irregular time sampling)
- Directly interpretable in terms of neural oscillations
- Relates to established literature (PAC, CFC)
- Single `tau` parameter (no extra_lag confusion)

### Limitations
- Filtering introduces edge effects and phase distortion
- Sharp frequency boundaries are artificial
- Loses transient, broadband events
- Requires sufficient signal length for low frequencies

### Multi-Scale Extension
```python
# Compute PhiID for all band pairs
band_names = list(bands.keys())
results = {}

for i, (name1, (lo1, hi1)) in enumerate(bands.items()):
    for j, (name2, (lo2, hi2)) in enumerate(bands.items()):
        if i < j:  # Only compute each pair once
            band1 = bandpass(signal, lo1, hi1, fs)
            band2 = bandpass(signal, lo2, hi2, fs)
            atoms, _ = calc_PhiID(band1, band2, tau)
            results[f'{name1}_{name2}'] = atoms

# Visualize as matrix
# rows/cols = frequency bands
# cell color = synergy (or other atom)
```

---

## Approach 2: Derivative PhiID (State-Change Dynamics)

### Concept
Compare signal with its temporal derivatives. This measures how the current state relates to the rate of change, capturing the signal's dynamical structure.

### Implementation
```python
def compute_derivatives(signal, fs):
    """Compute signal derivatives."""
    dt = 1 / fs
    velocity = np.gradient(signal, dt)           # First derivative
    acceleration = np.gradient(velocity, dt)     # Second derivative
    jerk = np.gradient(acceleration, dt)         # Third derivative
    return velocity, acceleration, jerk

# Compute derivatives
velocity, acceleration, jerk = compute_derivatives(signal, fs)

# Align lengths (gradient preserves length, but derivatives are noisier at edges)
n = len(signal) - 4  # Trim edges
position = signal[2:-2]
velocity = velocity[2:-2]
acceleration = acceleration[2:-2]

# PhiID: Position vs Velocity
atoms_pos_vel, _ = calc_PhiID(position, velocity, tau)

# PhiID: Velocity vs Acceleration
atoms_vel_acc, _ = calc_PhiID(velocity, acceleration, tau)

# PhiID: Position vs Acceleration (skip one derivative level)
atoms_pos_acc, _ = calc_PhiID(position, acceleration, tau)
```

### What It Measures
- How well current state predicts future change
- Whether dynamics are smooth (velocity predictable from position)
- Phase space structure without explicit reconstruction

### Interpretation of Atoms

| Atom | State-Change Meaning |
|------|---------------------|
| **rtr** | Redundancy: Position and velocity carry same info (smooth, predictable) |
| **xtx** | State memory: Position-specific information preserved |
| **yty** | Rate memory: Velocity-specific information preserved |
| **sts** | Dynamical synergy: Complex dynamics (neither alone sufficient) |
| **xty** | Position drives velocity (smooth acceleration) |
| **ytx** | Velocity drives position (inertial dynamics) |
| **xts + yts** | State/rate → Emergent pattern |
| **stx + sty** | Emergent → Modulates state/rate |

### Physical Interpretation
For a harmonic oscillator: x(t) = A·sin(ωt)
- Position and velocity are 90° out of phase
- High redundancy expected (one fully determines other)
- Zero synergy (simple dynamics)

For chaotic dynamics:
- Position-velocity relationship is complex
- High synergy expected
- Low predictability

### Advantages
- Single tau parameter (clean interpretation)
- Captures dynamical structure
- Related to phase space analysis
- Interpretable in physical terms

### Limitations
- Derivatives amplify noise
- Edge effects
- Requires smooth signals (not good for spiky data)
- Second derivative very noisy for discrete data

### Noise Mitigation
```python
from scipy.ndimage import gaussian_filter1d

# Smooth before differentiating
sigma = 2  # Gaussian smoothing width in samples
signal_smooth = gaussian_filter1d(signal, sigma)
velocity_smooth = np.gradient(signal_smooth, 1/fs)

# Or use Savitzky-Golay for derivative estimation
from scipy.signal import savgol_filter
velocity_sg = savgol_filter(signal, window_length=11, polyorder=2, deriv=1, delta=1/fs)
```

---

## Approach 3: Coarse-Grained Multi-Resolution PhiID

### Concept
Compare the signal at different temporal resolutions by averaging (coarse-graining). This reveals what information is preserved vs lost across scales.

### Implementation
```python
def coarse_grain(signal, factor):
    """
    Reduce temporal resolution by averaging every 'factor' samples.
    Similar to downsampling but uses averaging instead of decimation.
    """
    n = len(signal) // factor * factor  # Ensure divisible
    signal_trimmed = signal[:n]
    # Reshape and average
    return signal_trimmed.reshape(-1, factor).mean(axis=1)

def upsample_to_match(coarse, fine_length, factor):
    """Upsample coarse signal to match fine signal length."""
    # Repeat each coarse value 'factor' times
    return np.repeat(coarse, factor)[:fine_length]

# Create multi-resolution versions
scales = [1, 2, 4, 8, 16]  # Coarse-graining factors
signals_at_scale = {}

for scale in scales:
    if scale == 1:
        signals_at_scale[scale] = signal
    else:
        coarse = coarse_grain(signal, scale)
        # Upsample back to original length for PhiID computation
        signals_at_scale[scale] = upsample_to_match(coarse, len(signal), scale)

# PhiID between resolutions
# Fine (scale=1) vs Coarse (scale=8)
fine = signals_at_scale[1]
coarse = signals_at_scale[8]

# Trim to equal length
min_len = min(len(fine), len(coarse))
atoms_fine_coarse, _ = calc_PhiID(fine[:min_len], coarse[:min_len], tau)
```

### What It Measures
- Scale-invariance of information content
- What information is preserved at macro scales?
- What information exists only at fine scales?
- Emergent macro-scale patterns

### Interpretation of Atoms

| Atom | Multi-Resolution Meaning |
|------|-------------------------|
| **rtr** | Scale-invariant information (same at all resolutions) |
| **xtx** | Fine-scale memory (detail preserved) |
| **yty** | Coarse-scale memory (macro patterns preserved) |
| **sts** | Multi-scale synergy (requires both resolutions) |
| **xty** | Fine details predict macro patterns |
| **ytx** | Macro patterns modulate fine details |
| **xts + yts** | Scale-specific → Emergent |
| **stx + sty** | Emergent → Scale-specific |

### Connection to Renormalization
This is related to renormalization group analysis:
- Coarse-graining = spatial/temporal averaging
- Fixed points = scale-invariant structure
- High redundancy across scales = self-similarity (fractal-like)

### Advantages
- Direct measure of scale-invariance
- Connects to renormalization theory
- Interpretable in terms of "what's lost when we blur?"
- No filtering artifacts (just averaging)

### Limitations
- Upsampling introduces artificial correlations
- Integer factors only (2, 4, 8...)
- Edge effects from trimming
- Coarse-grained signal has "staircase" structure

### Alternative: Wavelet Multi-Resolution
```python
import pywt

def wavelet_multiresolution(signal, wavelet='db4', level=4):
    """Decompose signal into approximation (coarse) and details (fine)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] = approximation (coarsest)
    # coeffs[1:] = details (fine to coarse)
    
    # Reconstruct at each level
    reconstructions = {}
    for i in range(level + 1):
        # Zero out all but one level
        c = [np.zeros_like(c) for c in coeffs]
        c[i] = coeffs[i]
        reconstructions[f'level_{i}'] = pywt.waverec(c, wavelet)[:len(signal)]
    
    return reconstructions, coeffs

# Wavelet decomposition
recons, _ = wavelet_multiresolution(signal, level=4)

# PhiID between wavelet levels
atoms_approx_detail, _ = calc_PhiID(recons['level_0'], recons['level_3'], tau)
```

---

## Approach 4: Takens Delay Embedding PhiID

### Concept
Use Takens' theorem to reconstruct the attractor, then compute PhiID on the embedding dimensions. This captures the dynamical system's true state space structure.

### Theory Background
Takens' theorem: A 1D time series can reconstruct the full attractor of a dynamical system if embedded in sufficient dimensions with appropriate delay.

```
Embedding: [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
```

For PhiID, we need exactly 4 vectors. We can map the embedding to PhiID's structure.

### Implementation
```python
def takens_embed(signal, dim, tau_embed):
    """
    Create Takens delay embedding.
    
    Parameters
    ----------
    signal : array
        1D time series
    dim : int
        Embedding dimension
    tau_embed : int
        Embedding delay in samples
    
    Returns
    -------
    embedded : array of shape (dim, N)
        Embedded time series, each row is one delay
    """
    N = len(signal) - (dim - 1) * tau_embed
    embedded = np.zeros((dim, N))
    for i in range(dim):
        start = (dim - 1 - i) * tau_embed
        end = start + N
        embedded[i] = signal[start:end]
    return embedded

# Estimate optimal tau using mutual information
def estimate_embedding_tau(signal, max_tau=50):
    """Estimate embedding delay using first minimum of mutual information."""
    from sklearn.metrics import mutual_info_score
    
    mi_values = []
    for tau in range(1, max_tau):
        # Discretize for MI calculation
        x = np.digitize(signal[:-tau], bins=np.linspace(signal.min(), signal.max(), 20))
        y = np.digitize(signal[tau:], bins=np.linspace(signal.min(), signal.max(), 20))
        mi = mutual_info_score(x, y)
        mi_values.append(mi)
    
    # Find first local minimum
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i-1] and mi_values[i] < mi_values[i+1]:
            return i + 1
    
    return np.argmin(mi_values) + 1

# Optimal tau estimation
tau_embed = estimate_embedding_tau(signal)
print(f"Optimal embedding tau: {tau_embed} samples")

# Create 4D embedding for PhiID
embedded = takens_embed(signal, dim=4, tau_embed=tau_embed)
# Shape: (4, N) where rows are [x(t-3τ), x(t-2τ), x(t-τ), x(t)]

# Map to PhiID's 4 vectors
# Option A: First half = "past process", second half = "future process"
src_embedded = embedded[:2].mean(axis=0)  # Average of x(t-3τ), x(t-2τ)
tgt_embedded = embedded[2:].mean(axis=0)  # Average of x(t-τ), x(t)

atoms_takens, _ = calc_PhiID(src_embedded, tgt_embedded, tau=1)

# Option B: Alternating assignment
src_b = embedded[::2].mean(axis=0)   # x(t-3τ), x(t-τ)
tgt_b = embedded[1::2].mean(axis=0)  # x(t-2τ), x(t)

atoms_takens_b, _ = calc_PhiID(src_b, tgt_b, tau=1)
```

### ⚠️ WARNING: Options A and B Still Have Two Lags!

**The problem**: `calc_PhiID()` internally slices src and tgt again with its own `tau`:
```python
# Inside calc_PhiID - ALWAYS happens:
src_past, src_future = src[:-tau], src[tau:]
trg_past, trg_future = trg[:-tau], trg[tau:]
```

So even with `tau=1`, Options A and B create:
- tau_embed from Takens embedding
- tau=1 from calc_PhiID's internal slicing

**Result**: Still irregular temporal structure!

---

### ✅ Option C: Direct 4-Vector Construction (TRUE Single-Lag Solution)

To achieve genuinely regular temporal sampling with ONE parameter, we must **bypass calc_PhiID** and construct the 4 vectors directly:

```python
def takens_phiid_direct(signal, tau_embed, kind="gaussian", redundancy="MMI"):
    """
    Compute PhiID on Takens embedding with TRUE SINGLE lag parameter.
    
    Bypasses calc_PhiID to avoid the two-lag problem.
    Creates perfectly regular temporal sampling: t, t+τ, t+2τ, t+3τ
    
    Parameters
    ----------
    signal : array
        1D time series
    tau_embed : int
        Embedding delay in samples (the ONLY temporal parameter)
    kind : str
        'gaussian' or 'discrete'
    redundancy : str
        'MMI' or 'CCS'
    
    Returns
    -------
    atoms_res : dict
        PhiID atoms
    calc_res : dict
        Intermediate calculations
    """
    from phyid.calculate import (
        _get_entropy_four_vec, 
        _get_coinfo_four_vec,
        _get_redundancy_four_vec, 
        _get_double_redundancy_four_vec,
        _get_atoms_four_vec
    )
    from phyid.utils import PhiID_atoms_abbr
    
    # Create 4D Takens embedding with PERFECTLY REGULAR spacing
    N = len(signal) - 3 * tau_embed
    
    X = np.zeros((4, N))
    X[0] = signal[0:N]                          # p1 = x(t)       - "src_past"
    X[1] = signal[tau_embed:N+tau_embed]        # p2 = x(t+τ)     - "tgt_past"
    X[2] = signal[2*tau_embed:N+2*tau_embed]    # t1 = x(t+2τ)    - "src_future"
    X[3] = signal[3*tau_embed:N+3*tau_embed]    # t2 = x(t+3τ)    - "tgt_future"
    
    # Timeline: t, t+τ, t+2τ, t+3τ
    # Spacing:  [τ], [τ], [τ]  ← PERFECTLY REGULAR!
    
    # Normalize (same as calc_PhiID does)
    if kind == "gaussian":
        X_norm = X / np.std(X, axis=1, ddof=1, keepdims=True)
        X_input = X_norm
    elif kind == "discrete":
        from phyid.utils import _binarize
        X_input = np.array([_binarize(X[i]) for i in range(4)])
    else:
        raise ValueError("kind must be 'gaussian' or 'discrete'")
    
    # Run PhiID computation pipeline directly
    h_res = _get_entropy_four_vec(X_input, kind=kind)
    I_res = _get_coinfo_four_vec(h_res)
    R_res = _get_redundancy_four_vec(redundancy, I_res)
    
    calc_res = {
        "h_res": h_res,
        "I_res": I_res,
        "R_res": R_res
    }
    
    rtr = _get_double_redundancy_four_vec(redundancy, calc_res)
    calc_res["rtr"] = rtr
    
    atoms_res = _get_atoms_four_vec(calc_res)
    
    return atoms_res, calc_res


# Usage example
tau_embed = estimate_embedding_tau(signal)  # e.g., 10 samples
atoms, calc_res = takens_phiid_direct(signal, tau_embed)

# Multi-scale analysis: vary tau_embed
tau_values = [5, 10, 20, 40, 80]  # Different temporal scales
results_by_tau = {}
for tau in tau_values:
    atoms, _ = takens_phiid_direct(signal, tau)
    results_by_tau[tau] = atoms
```

### Comparison of All Options

| Option | How it works | # Lags | Temporal Structure | Recommended? |
|--------|--------------|--------|-------------------|--------------|
| **A** | Average dims, call calc_PhiID | **2** | Irregular | ❌ No |
| **B** | Alternating dims, call calc_PhiID | **2** | Irregular | ❌ No |
| **C** | Direct 4-vector, bypass calc_PhiID | **1** | t, t+τ, t+2τ, t+3τ (regular) | ✅ Yes |

### Option C Temporal Structure

```
Timeline with tau_embed = 10 samples:

Sample:     0    10    20    30
            |     |     |     |
Vector:    p1    p2    t1    t2
           (src  (tgt  (src  (tgt
           past) past) fut)  fut)

Spacing: [10] [10] [10]  ← PERFECTLY REGULAR!
```

Compare to current approach with tau=1, extra_lag=30:
```
Sample:     0     1    30    31
            |     |     |     |
Spacing:  [1]  [29]  [1]  ← IRREGULAR!
```
```

### Difference from Current Approach

| Aspect | Current (Shifted Segment) | Takens Embedding |
|--------|--------------------------|------------------|
| Temporal sampling | Irregular: τ, lag-τ, τ | Regular: τ, τ, τ |
| Parameters | Two: tau AND extra_lag | One: tau_embed |
| Time points | t, t+τ, t+lag, t+lag+τ | t-3τ, t-2τ, t-τ, t |
| Interpretation | Two temporal windows | Single attractor trajectory |
| Theoretical basis | Ad-hoc | Takens' theorem |

### What It Measures
- Information structure within the reconstructed attractor
- Predictability along the trajectory
- Determinism vs stochasticity
- Attractor complexity

### Interpretation of Atoms

| Atom | Attractor Meaning |
|------|-------------------|
| **rtr** | Redundant trajectory information (predictable dynamics) |
| **xtx, yty** | Dimension-specific information preserved |
| **sts** | Emergent attractor structure (not in individual dimensions) |
| **xty, ytx** | Cross-prediction along trajectory |
| **Synergy terms** | Complex, high-dimensional dynamics |

### Advantages
- Theoretically grounded (Takens' theorem)
- Single clean parameter (tau_embed)
- Regular temporal sampling
- Captures true dynamical structure

### Limitations
- Optimal tau estimation can be tricky
- Assumes deterministic dynamics (not ideal for stochastic signals)
- Requires sufficient embedding dimension (we're fixed at 4)
- Sensitive to noise

### Choosing tau_embed
```python
# Method 1: First minimum of mutual information (standard)
tau_mi = estimate_embedding_tau(signal)

# Method 2: First zero of autocorrelation
def estimate_tau_autocorr(signal):
    acf = np.correlate(signal - signal.mean(), signal - signal.mean(), mode='full')
    acf = acf[len(acf)//2:] / acf[len(acf)//2]
    # Find first zero crossing
    zero_crossings = np.where(np.diff(np.sign(acf)))[0]
    return zero_crossings[0] if len(zero_crossings) > 0 else len(acf) // 4

tau_acf = estimate_tau_autocorr(signal)

# Method 3: 1/4 of dominant period
def estimate_tau_period(signal, fs):
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    peak_freq = freqs[np.argmax(fft[1:]) + 1]
    period_samples = int(fs / peak_freq)
    return period_samples // 4

tau_period = estimate_tau_period(signal, fs)
```

---

## Approach 5: Amplitude-Phase Decomposition PhiID

### Concept
Use Hilbert transform to decompose signal into instantaneous amplitude (envelope) and phase, then compute PhiID between these components. This directly measures amplitude-phase coupling.

### Implementation
```python
from scipy.signal import hilbert

def extract_amplitude_phase(signal):
    """Extract instantaneous amplitude and phase using Hilbert transform."""
    analytic = hilbert(signal)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    phase_unwrapped = np.unwrap(phase)
    return amplitude, phase, phase_unwrapped

def extract_inst_frequency(phase_unwrapped, fs):
    """Compute instantaneous frequency from unwrapped phase."""
    inst_freq = np.gradient(phase_unwrapped) * fs / (2 * np.pi)
    return inst_freq

# Extract components
amplitude, phase, phase_unwrap = extract_amplitude_phase(signal)
inst_freq = extract_inst_frequency(phase_unwrap, fs)

# PhiID: Amplitude vs Phase
# Note: phase is circular, so we use sin/cos projection
phase_sin = np.sin(phase)
phase_cos = np.cos(phase)

atoms_amp_phase_sin, _ = calc_PhiID(amplitude[:-1], phase_sin[:-1], tau)
atoms_amp_phase_cos, _ = calc_PhiID(amplitude[:-1], phase_cos[:-1], tau)

# PhiID: Amplitude vs Instantaneous Frequency
atoms_amp_freq, _ = calc_PhiID(amplitude[:-1], inst_freq, tau)

# For narrowband signals, first filter then extract
def phiid_amp_phase_narrowband(signal, band_low, band_high, fs, tau):
    """PhiID on amplitude-phase of a specific frequency band."""
    # Bandpass filter
    filtered = bandpass(signal, band_low, band_high, fs)
    
    # Extract amplitude and phase
    amp, phase, _ = extract_amplitude_phase(filtered)
    phase_sin = np.sin(phase)
    
    # PhiID
    atoms, _ = calc_PhiID(amp, phase_sin, tau)
    return atoms

# Alpha band amplitude-phase coupling
atoms_alpha_ap = phiid_amp_phase_narrowband(signal, 8, 12, fs, tau)
```

### What It Measures
- How amplitude modulations relate to phase dynamics
- Traditional PAC asks "does slow phase modulate fast amplitude?"
- PhiID asks "what's the full information relationship?"

### Interpretation of Atoms

| Atom | Amplitude-Phase Meaning |
|------|------------------------|
| **rtr** | Common information (locked oscillation) |
| **xtx** | Amplitude memory (envelope dynamics independent of phase) |
| **yty** | Phase memory (phase dynamics independent of amplitude) |
| **sts** | Amplitude-phase synergy (complex modulation pattern) |
| **xty** | Amplitude predicts phase (FM from AM) |
| **ytx** | Phase predicts amplitude (AM from FM) |
| **xts + yts** | Components → Emergent pattern |
| **stx + sty** | Emergent → Modulates components |

### Physical Examples

**Simple oscillator** (x = A·sin(ωt)):
- Constant amplitude, linear phase
- Zero coupling (A and φ are independent)

**AM signal** (x = (1 + m·sin(ωm·t))·sin(ωc·t)):
- Amplitude modulated
- High ytx: phase carries info about amplitude modulation

**FM signal** (x = sin(ωc·t + β·sin(ωm·t))):
- Frequency (phase derivative) modulated
- High xty: amplitude dynamics encode frequency changes

**PAC** (phase-amplitude coupling):
- Slow phase modulates fast amplitude
- Requires cross-frequency version (see combined approach)

### Advantages
- Directly measures amplitude-phase relationships
- Connects to PAC literature
- Clean interpretation for oscillatory signals
- Single tau parameter

### Limitations
- Phase is circular (need sin/cos projection)
- Broadband signals: phase not well-defined
- Hilbert transform has edge effects
- Instantaneous frequency can be negative (non-physical)

### Cross-Frequency Amplitude-Phase Coupling
```python
def phiid_cross_freq_pac(signal, slow_band, fast_band, fs, tau):
    """
    PhiID for cross-frequency phase-amplitude coupling.
    Tests: Does slow-band phase modulate fast-band amplitude?
    """
    # Extract slow phase
    slow_filtered = bandpass(signal, slow_band[0], slow_band[1], fs)
    _, slow_phase, _ = extract_amplitude_phase(slow_filtered)
    slow_phase_sin = np.sin(slow_phase)
    
    # Extract fast amplitude
    fast_filtered = bandpass(signal, fast_band[0], fast_band[1], fs)
    fast_amp, _, _ = extract_amplitude_phase(fast_filtered)
    
    # PhiID: slow phase → fast amplitude
    atoms, _ = calc_PhiID(slow_phase_sin, fast_amp, tau)
    return atoms

# Theta-gamma PAC (classic example)
atoms_theta_gamma_pac = phiid_cross_freq_pac(
    signal, 
    slow_band=(4, 8),   # Theta
    fast_band=(30, 50), # Gamma
    fs=fs, 
    tau=tau
)
```

---

## Comparison Summary

### The Two-Lag Problem

Most approaches that use `calc_PhiID()` inherit its internal `tau` parameter, creating a **two-lag problem**:

| Approach | Uses calc_PhiID? | # Lag Parameters | True Single-Lag? |
|----------|------------------|------------------|------------------|
| **Current** | Yes | 2 (tau + extra_lag) | ❌ No |
| **1. Cross-Freq** | Yes | 2 (tau + implicit freq) | ❌ No |
| **2. Derivative** | Yes | 2 (tau + derivative order) | ❌ No |
| **3. Coarse-Grain** | Yes | 2 (tau + scale factor) | ❌ No |
| **4. Takens (Option C)** | **No (bypass)** | **1 (tau_embed only)** | ✅ **Yes** |
| **5. Amp-Phase** | Yes | 2 (tau + Hilbert) | ❌ No |

**Key insight**: Only **Approach 4 with Option C** (direct 4-vector construction) achieves true single-lag, regular temporal sampling.

### Feature Comparison

| Approach | Temporal Structure | Best For | Complexity |
|----------|-------------------|----------|------------|
| **Current** | Irregular: τ, lag-τ, τ | Quick relative comparisons | Low |
| **1. Cross-Freq** | Within-sample (filtered) | Neural oscillations, CFC | Medium |
| **2. Derivative** | Adjacent samples | Smooth dynamics, phase space | Low |
| **3. Coarse-Grain** | Multi-resolution | Scale-invariance, fractals | Medium |
| **4. Takens (C)** | **Regular: τ, τ, τ** | **Rigorous analysis, publication** | Medium |
| **5. Amp-Phase** | Within-sample (Hilbert) | Oscillations, PAC | Medium |

---

## Recommendations

### For Rigorous Publication
**Use Takens Embedding with Option C (Direct 4-Vector)**:
- ✅ Single clean parameter
- ✅ Theoretical justification (Takens' theorem)
- ✅ Perfectly regular temporal sampling
- ✅ Connects to dynamical systems literature
- ✅ No interpretation ambiguity

### For EEG/Neural Data
1. **Primary**: Takens Option C - rigorous temporal analysis
2. **Add**: Cross-Frequency (Approach 1) - for oscillation-specific questions
3. **Add**: Amp-Phase (Approach 5) - for PAC analysis

### For Cardiovascular (HRV, BP)
1. **Primary**: Takens Option C - attractor structure
2. **Add**: Derivative (Approach 2) - regulatory dynamics
3. **Add**: Coarse-Grain (Approach 3) - multi-scale HRV

### For Quick Exploration
- Current approach is fine for **relative comparisons** across lags
- Don't over-interpret absolute atom values
- Use as preliminary analysis before rigorous methods

---

## Implementation Priority

1. **Approach 4 Option C (Takens Direct)** - Highest priority, fixes two-lag problem
2. **Approach 2 (Derivative)** - Easy to add, complements Takens
3. **Approach 1 (Cross-Freq)** - High value for neural data
4. **Approach 3 (Coarse-Grain)** - Unique multi-scale insight
5. **Approach 5 (Amp-Phase)** - Specialized for oscillatory signals

---

## Summary: The Right Way Forward

For theoretically sound temporal PhiID analysis:

```python
# ✅ CORRECT: Takens with direct 4-vector (Option C)
atoms = takens_phiid_direct(signal, tau_embed=10)
# Timeline: t, t+10, t+20, t+30 (regular spacing)

# ❌ AVOID: Current approach (or any using calc_PhiID for temporal)
atoms = calc_PhiID(signal[:-30], signal[30:], tau=1)
# Timeline: t, t+1, t+30, t+31 (irregular spacing)
```

---

*Document generated: February 2026*
*Framework version: 2.0*
