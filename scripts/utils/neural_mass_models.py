"""
Neural Mass Model Utilities for Temporal PID Analysis
======================================================

Provides simulators for neural mass models with known dynamics,
enabling ground-truth validation of temporal information decomposition.

Models implemented:
1. Single Population with Delayed Feedback
2. Excitatory-Inhibitory (Wilson-Cowan style)

Each model includes:
- Simulator function
- Theoretical predictions for PID
- Validation tests
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Dict, Optional, List
import warnings


# =============================================================================
# ACTIVATION FUNCTIONS
# =============================================================================

def sigmoid(x: np.ndarray, gain: float = 1.0, threshold: float = 0.0) -> np.ndarray:
    """Standard sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-gain * (x - threshold)))


def tanh_activation(x: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Tanh activation (symmetric around 0)."""
    return np.tanh(gain * x)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


# =============================================================================
# MODEL 1: SINGLE POPULATION WITH DELAYED FEEDBACK
# =============================================================================

def simulate_single_population_delayed(
    n_samples: int = 10000,
    fs: float = 1000.0,
    delay_ms: float = 10.0,
    weight: float = 0.8,
    noise_std: float = 0.1,
    activation: str = 'tanh',
    gain: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate a single neural population with delayed self-feedback.
    
    Model:
        x(t) = f(w * x(t - τ)) + noise
    
    Where:
        - f is the activation function
        - w is the self-feedback weight
        - τ is the delay in samples
        - noise is Gaussian
    
    Parameters
    ----------
    n_samples : int
        Number of time points to simulate
    fs : float
        Sampling frequency (Hz)
    delay_ms : float
        Feedback delay in milliseconds
    weight : float
        Self-feedback weight (|w| < 1 for stability)
    noise_std : float
        Standard deviation of noise
    activation : str
        'tanh', 'sigmoid', 'relu', or 'linear'
    gain : float
        Activation function gain
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    x : np.ndarray
        Simulated time series
    params : dict
        Dictionary of simulation parameters
    
    Theoretical PID Predictions
    ---------------------------
    For I(x(t-τ₁), x(t-τ₂) → x(t)):
    
    1. When τ₁ = delay: HIGH UNIQUE₁
       - x(t-delay) directly predicts x(t)
       
    2. When τ₁ = τ₂ ≈ delay: HIGH REDUNDANCY
       - Both lags capture the same feedback loop
       
    3. When τ₁ = delay, τ₂ = 2*delay: POTENTIAL SYNERGY
       - x(t-2τ) provides context that modifies prediction
       - Especially with nonlinear activation
       
    4. When τ₁, τ₂ >> delay: LOW INFORMATION
       - Signal decorrelates beyond feedback memory
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert delay to samples
    delay_samples = int(delay_ms * fs / 1000)
    
    # Select activation function
    if activation == 'tanh':
        f = lambda x: tanh_activation(x, gain)
    elif activation == 'sigmoid':
        f = lambda x: sigmoid(x, gain)
    elif activation == 'relu':
        f = relu
    elif activation == 'linear':
        f = lambda x: x
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    # Initialize
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * noise_std
    
    # Warm-up with noise
    x[:delay_samples] = noise[:delay_samples]
    
    # Simulate
    for t in range(delay_samples, n_samples):
        x[t] = f(weight * x[t - delay_samples]) + noise[t]
    
    params = {
        'model': 'single_population_delayed',
        'n_samples': n_samples,
        'fs': fs,
        'delay_ms': delay_ms,
        'delay_samples': delay_samples,
        'weight': weight,
        'noise_std': noise_std,
        'activation': activation,
        'gain': gain,
        'seed': seed
    }
    
    return x, params


def simulate_multi_delay_population(
    n_samples: int = 10000,
    fs: float = 1000.0,
    delays_ms: List[float] = [10.0, 25.0, 50.0],
    weights: List[float] = [0.5, 0.3, 0.2],
    noise_std: float = 0.1,
    activation: str = 'tanh',
    gain: float = 3.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Single population with MULTIPLE feedback delays - creates temporal integration.
    
    Model:
        x(t) = f(Σᵢ wᵢ * x(t - τᵢ)) + noise
    
    This creates synergy because information from multiple timescales combines.
    
    Parameters
    ----------
    delays_ms : list of float
        Multiple feedback delays
    weights : list of float
        Weight for each delay (should sum to < 1 for stability)
    gain : float
        Nonlinearity gain (higher = more synergy expected)
    
    Theoretical PID Predictions
    ---------------------------
    1. At each delay τᵢ: HIGH UNIQUE from that lag
    2. At pairs (τᵢ, τⱼ): SYNERGY because both contribute
    3. Higher gain → more synergy (nonlinear mixing)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Convert delays to samples
    delay_samples = [int(d * fs / 1000) for d in delays_ms]
    max_delay = max(delay_samples)
    
    # Select activation
    if activation == 'tanh':
        f = lambda x: tanh_activation(x, gain)
    elif activation == 'sigmoid':
        f = lambda x: sigmoid(x, gain, threshold=0.5)
    else:
        f = lambda x: x
    
    # Initialize
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * noise_std
    x[:max_delay] = noise[:max_delay]
    
    # Simulate
    for t in range(max_delay, n_samples):
        weighted_sum = sum(w * x[t - d] for w, d in zip(weights, delay_samples))
        x[t] = f(weighted_sum) + noise[t]
    
    params = {
        'model': 'multi_delay_population',
        'n_samples': n_samples,
        'fs': fs,
        'delays_ms': delays_ms,
        'delay_samples': delay_samples,
        'weights': weights,
        'noise_std': noise_std,
        'activation': activation,
        'gain': gain,
        'seed': seed
    }
    
    return x, params


def simulate_xor_timescales(
    n_samples: int = 10000,
    fs: float = 1000.0,
    tau1_ms: float = 10.0,
    tau2_ms: float = 50.0,
    mix_prob: float = 0.7,
    noise_prob: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Discrete process with XOR-like interaction between timescales.
    
    Creates GENUINE SYNERGY: you need BOTH timescales to predict.
    
    Model:
        With prob mix_prob: x(t) = x(t-τ1) XOR x(t-τ2)
        With prob noise_prob: x(t) = random
        Otherwise: x(t) = x(t-τ1)
    
    This creates:
    - Synergy at (τ1, τ2) lag pair: XOR requires both
    - Some redundancy at τ1: copy operation
    - Unique info at each delay
    """
    if seed is not None:
        np.random.seed(seed)
    
    tau1 = int(tau1_ms * fs / 1000)
    tau2 = int(tau2_ms * fs / 1000)
    max_tau = max(tau1, tau2)
    
    x = np.zeros(n_samples, dtype=int)
    x[:max_tau] = np.random.randint(0, 2, max_tau)
    
    for t in range(max_tau, n_samples):
        r = np.random.rand()
        if r < mix_prob:
            # XOR of two timescales - requires BOTH to predict
            x[t] = x[t - tau1] ^ x[t - tau2]
        elif r < mix_prob + noise_prob:
            x[t] = np.random.randint(2)
        else:
            x[t] = x[t - tau1]
    
    params = {
        'model': 'xor_timescales',
        'n_samples': n_samples,
        'fs': fs,
        'tau1_ms': tau1_ms,
        'tau2_ms': tau2_ms,
        'tau1_samples': tau1,
        'tau2_samples': tau2,
        'mix_prob': mix_prob,
        'noise_prob': noise_prob,
        'seed': seed
    }
    
    return x, params


def simulate_conditional_timescales(
    n_samples: int = 10000,
    fs: float = 1000.0,
    tau_fast_ms: float = 10.0,
    tau_slow_ms: float = 100.0,
    noise_std: float = 0.3,
    gain: float = 5.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Process where fast dynamics are GATED by slow dynamics.
    
    Model:
        slow(t) = low-pass filtered noise (intrinsic slow fluctuations)
        x(t) = f(w_fast * x(t-τ_fast) * g(slow(t))) + noise
    
    Where g(slow) modulates the fast feedback:
    - When slow is high: fast dynamics dominate (high autocorr at τ_fast)
    - When slow is low: noise dominates (low autocorr)
    
    This creates synergy because:
    - x(t-τ_fast) alone doesn't predict x(t) well (depends on slow state)
    - x(t-τ_slow) alone doesn't predict x(t) well (fast timescale)
    - BOTH together predict much better
    """
    if seed is not None:
        np.random.seed(seed)
    
    tau_fast = int(tau_fast_ms * fs / 1000)
    tau_slow = int(tau_slow_ms * fs / 1000)
    max_tau = max(tau_fast, tau_slow)
    
    # Generate slow modulation (low-pass filtered noise)
    from scipy.signal import butter, filtfilt
    slow_raw = np.random.randn(n_samples)
    b, a = butter(2, 2 * (1000 / tau_slow_ms) / fs, btype='low')
    slow = filtfilt(b, a, slow_raw)
    slow = (slow - slow.mean()) / slow.std()  # Normalize
    
    # Gating function: maps slow to [0.2, 1.0]
    gate = 0.2 + 0.8 * sigmoid(slow, gain=2.0, threshold=0.0)
    
    # Simulate x with gated fast dynamics
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * noise_std
    x[:max_tau] = noise[:max_tau]
    
    f = lambda v: tanh_activation(v, gain)
    
    for t in range(max_tau, n_samples):
        # Fast feedback, gated by slow state
        fast_input = 0.8 * x[t - tau_fast] * gate[t]
        x[t] = f(fast_input) + noise[t]
    
    params = {
        'model': 'conditional_timescales',
        'n_samples': n_samples,
        'fs': fs,
        'tau_fast_ms': tau_fast_ms,
        'tau_slow_ms': tau_slow_ms,
        'tau_fast_samples': tau_fast,
        'tau_slow_samples': tau_slow,
        'noise_std': noise_std,
        'gain': gain,
        'seed': seed
    }
    
    return x, slow, params


def simulate_hierarchical_timescales(
    n_samples: int = 10000,
    fs: float = 1000.0,
    tau_fast_ms: float = 5.0,
    tau_slow_ms: float = 50.0,
    delay_ms: float = 10.0,
    w_fast: float = 0.7,
    w_slow: float = 0.5,
    w_cross_up: float = 0.3,
    w_cross_down: float = 0.2,
    noise_std: float = 0.1,
    gain: float = 2.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Two-population model with different intrinsic timescales.
    
    Creates hierarchical temporal integration:
    - Fast population (short memory)
    - Slow population (long memory)  
    - Bidirectional coupling creates cross-scale integration
    
    Model:
        τ_fast * dx_fast/dt = -x_fast + f(w_fast*x_fast + w_cross_down*x_slow) + noise
        τ_slow * dx_slow/dt = -x_slow + f(w_slow*x_slow + w_cross_up*x_fast) + noise
    
    Theoretical PID Predictions
    ---------------------------
    Within fast: High info at short lags, decays quickly
    Within slow: High info extends to long lags
    Cross (fast, slow → fast): SYNERGY at intermediate lags
    Cross (fast, slow → slow): REDUNDANCY (slow integrates fast)
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1000 / fs  # ms per sample
    delay_samples = int(delay_ms * fs / 1000)
    
    f = lambda x: tanh_activation(x, gain)
    
    # Initialize
    x_fast = np.zeros(n_samples)
    x_slow = np.zeros(n_samples)
    
    noise_fast = np.random.randn(n_samples) * noise_std
    noise_slow = np.random.randn(n_samples) * noise_std
    
    # Warm-up
    x_fast[:delay_samples] = 0.1 * noise_fast[:delay_samples]
    x_slow[:delay_samples] = 0.1 * noise_slow[:delay_samples]
    
    # Simulate
    for t in range(delay_samples, n_samples):
        # Delayed inputs
        xf_d = x_fast[t - delay_samples]
        xs_d = x_slow[t - delay_samples]
        
        # Euler integration
        dx_fast = (-x_fast[t-1] + f(w_fast * xf_d + w_cross_down * xs_d)) * dt / tau_fast_ms
        dx_slow = (-x_slow[t-1] + f(w_slow * xs_d + w_cross_up * xf_d)) * dt / tau_slow_ms
        
        x_fast[t] = x_fast[t-1] + dx_fast + noise_fast[t] * np.sqrt(dt / tau_fast_ms)
        x_slow[t] = x_slow[t-1] + dx_slow + noise_slow[t] * np.sqrt(dt / tau_slow_ms)
    
    params = {
        'model': 'hierarchical_timescales',
        'n_samples': n_samples,
        'fs': fs,
        'tau_fast_ms': tau_fast_ms,
        'tau_slow_ms': tau_slow_ms,
        'delay_ms': delay_ms,
        'w_fast': w_fast,
        'w_slow': w_slow,
        'w_cross_up': w_cross_up,
        'w_cross_down': w_cross_down,
        'noise_std': noise_std,
        'gain': gain,
        'seed': seed
    }
    
    return x_fast, x_slow, params


def get_single_population_predictions(delay_ms: float, fs: float) -> Dict:
    """
    Get theoretical PID predictions for single population model.
    
    Returns lag pairs and expected dominant PID components.
    """
    delay_samples = int(delay_ms * fs / 1000)
    
    predictions = {
        'optimal_lag': delay_samples,
        'optimal_lag_ms': delay_ms,
        'lag_pairs': [
            {
                'tau1': delay_samples,
                'tau2': delay_samples,
                'expected': 'HIGH_REDUNDANCY',
                'reason': 'Both lags at feedback delay capture same info'
            },
            {
                'tau1': delay_samples,
                'tau2': 2 * delay_samples,
                'expected': 'UNIQUE1_DOMINANT',
                'reason': 'τ₁ at delay, τ₂ captures grandparent'
            },
            {
                'tau1': delay_samples,
                'tau2': delay_samples // 2 if delay_samples > 1 else 1,
                'expected': 'UNIQUE1_DOMINANT',
                'reason': 'τ₁ at delay, τ₂ too short to add info'
            },
            {
                'tau1': 3 * delay_samples,
                'tau2': 4 * delay_samples,
                'expected': 'LOW_TOTAL_INFO',
                'reason': 'Both lags beyond effective memory'
            }
        ],
        'nonlinearity_effect': 'Higher gain increases synergy at (τ, 2τ) pairs'
    }
    
    return predictions


def test_single_population_model(verbose: bool = True) -> bool:
    """
    Test the single population model against theoretical predictions.
    
    Returns True if basic sanity checks pass.
    """
    from scipy.stats import pearsonr
    
    tests_passed = True
    
    # Test 1: Autocorrelation peak at delay
    x, params = simulate_single_population_delayed(
        n_samples=50000, delay_ms=20, weight=0.9, noise_std=0.1, seed=42
    )
    delay = params['delay_samples']
    
    # Compute autocorrelation
    acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
    acf = acf[len(acf)//2:]  # Take positive lags
    acf = acf / acf[0]  # Normalize
    
    # Check that ACF has significant value at delay
    acf_at_delay = acf[delay]
    if acf_at_delay > 0.3:
        if verbose:
            print(f"✓ Test 1 PASSED: ACF at delay = {acf_at_delay:.3f} > 0.3")
    else:
        if verbose:
            print(f"✗ Test 1 FAILED: ACF at delay = {acf_at_delay:.3f} < 0.3")
        tests_passed = False
    
    # Test 2: Weight < 1 gives stable dynamics
    x_stable, _ = simulate_single_population_delayed(
        n_samples=10000, weight=0.8, seed=42
    )
    if np.all(np.isfinite(x_stable)) and np.std(x_stable) < 10:
        if verbose:
            print(f"✓ Test 2 PASSED: Stable dynamics (std = {np.std(x_stable):.3f})")
    else:
        if verbose:
            print("✗ Test 2 FAILED: Unstable dynamics")
        tests_passed = False
    
    # Test 3: x(t) correlated with x(t-delay)
    corr, _ = pearsonr(x[delay:], x[:-delay])
    if corr > 0.5:
        if verbose:
            print(f"✓ Test 3 PASSED: Correlation(x(t), x(t-τ)) = {corr:.3f} > 0.5")
    else:
        if verbose:
            print(f"✗ Test 3 FAILED: Correlation = {corr:.3f} < 0.5")
        tests_passed = False
    
    # Test 4: Noise matters - different seeds give different series
    x1, _ = simulate_single_population_delayed(n_samples=1000, seed=1)
    x2, _ = simulate_single_population_delayed(n_samples=1000, seed=2)
    if np.corrcoef(x1, x2)[0, 1] < 0.5:
        if verbose:
            print("✓ Test 4 PASSED: Different seeds give different series")
    else:
        if verbose:
            print("✗ Test 4 FAILED: Series too similar with different seeds")
        tests_passed = False
    
    return tests_passed


# =============================================================================
# MODEL 2: EXCITATORY-INHIBITORY (WILSON-COWAN STYLE)
# =============================================================================

def simulate_ei_population(
    n_samples: int = 10000,
    fs: float = 1000.0,
    delay_ms: float = 5.0,
    wEE: float = 1.2,
    wEI: float = 1.0,
    wIE: float = 1.0,
    wII: float = 0.5,
    tau_E: float = 10.0,
    tau_I: float = 20.0,
    input_E: float = 0.5,
    input_I: float = 0.0,
    noise_std: float = 0.1,
    gain: float = 1.0,
    threshold: float = 2.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simulate coupled Excitatory-Inhibitory populations (Wilson-Cowan style).
    
    Model (continuous-time, discretized with Euler):
        τ_E * dE/dt = -E + f(wEE*E - wEI*I + input_E) + noise
        τ_I * dI/dt = -I + f(wIE*E - wII*I + input_I) + noise
    
    With synaptic delay applied to coupling terms.
    
    Parameters
    ----------
    n_samples : int
        Number of time points
    fs : float
        Sampling frequency (Hz)
    delay_ms : float
        Synaptic delay in milliseconds
    wEE, wEI, wIE, wII : float
        Connection weights (E→E, I→E, E→I, I→I)
    tau_E, tau_I : float
        Time constants in ms
    input_E, input_I : float
        External input to each population
    noise_std : float
        Noise standard deviation
    gain : float
        Sigmoid gain
    threshold : float
        Sigmoid threshold
    seed : int, optional
        Random seed
    
    Returns
    -------
    E : np.ndarray
        Excitatory population activity
    I : np.ndarray
        Inhibitory population activity
    params : dict
        Simulation parameters
    
    Theoretical PID Predictions
    ---------------------------
    For I(E(t-τ₁), E(t-τ₂) → E(t)):
    
    1. τ₁ ≈ delay: HIGH UNIQUE₁
       - Direct recurrence captured
       
    2. τ₁ ≈ oscillation_period/2, τ₂ ≈ delay: SYNERGY
       - E-I oscillation creates synergistic structure
       
    For I(E(t-τ), I(t-τ) → E(t)):
    
    1. When balanced (wEI ≈ wEE): HIGH SYNERGY
       - Need BOTH E and I to predict
       
    2. When E-dominant (wEE >> wEI): HIGH UNIQUE_E
       
    3. At oscillation period: HIGH REDUNDANCY
       - E and I anti-phase, both predict via oscillation
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1000 / fs  # Time step in ms
    delay_samples = int(delay_ms * fs / 1000)
    
    # Initialize
    E = np.zeros(n_samples)
    I = np.zeros(n_samples)
    
    # Initial conditions
    E[:delay_samples] = 0.2 + np.random.randn(delay_samples) * 0.01
    I[:delay_samples] = 0.2 + np.random.randn(delay_samples) * 0.01
    
    # Activation function
    f = lambda x: sigmoid(x, gain, threshold)
    
    # Noise
    noise_E = np.random.randn(n_samples) * noise_std
    noise_I = np.random.randn(n_samples) * noise_std
    
    # Simulate using Euler integration
    for t in range(delay_samples, n_samples):
        # Delayed inputs
        E_delayed = E[t - delay_samples]
        I_delayed = I[t - delay_samples]
        
        # Currents
        curr_E = wEE * E_delayed - wEI * I_delayed + input_E
        curr_I = wIE * E_delayed - wII * I_delayed + input_I
        
        # Euler step
        dE = (-E[t-1] + f(curr_E)) * dt / tau_E + noise_E[t] * np.sqrt(dt)
        dI = (-I[t-1] + f(curr_I)) * dt / tau_I + noise_I[t] * np.sqrt(dt)
        
        E[t] = E[t-1] + dE
        I[t] = I[t-1] + dI
        
        # Clamp to valid range
        E[t] = np.clip(E[t], 0, 1)
        I[t] = np.clip(I[t], 0, 1)
    
    params = {
        'model': 'ei_population',
        'n_samples': n_samples,
        'fs': fs,
        'delay_ms': delay_ms,
        'delay_samples': delay_samples,
        'wEE': wEE,
        'wEI': wEI,
        'wIE': wIE,
        'wII': wII,
        'tau_E': tau_E,
        'tau_I': tau_I,
        'input_E': input_E,
        'input_I': input_I,
        'noise_std': noise_std,
        'gain': gain,
        'threshold': threshold,
        'seed': seed
    }
    
    return E, I, params


def simulate_ei_oscillatory(
    n_samples: int = 10000,
    fs: float = 1000.0,
    target_freq_hz: float = 25.0,
    noise_std: float = 0.05,
    coupling_strength: float = 0.3,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    E-I model with coupled but partially independent oscillators.
    
    Uses two coupled phase oscillators where E and I have:
    - Shared oscillation frequency (set by target_freq_hz)
    - Partial coupling (not perfectly phase-locked)
    - Independent noise → creates synergistic information
    
    This is better for testing phase-based predictions because
    E and I carry BOTH redundant (shared phase) and unique info.
    
    Parameters
    ----------
    target_freq_hz : float
        Target oscillation frequency
    coupling_strength : float
        How strongly E and I are coupled (0=independent, 1=locked)
    
    Returns
    -------
    E, I : np.ndarray
        Excitatory and inhibitory activity
    params : dict
        Simulation parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    dt = 1.0 / fs  # seconds
    omega = 2 * np.pi * target_freq_hz
    
    # Two coupled oscillators with independent phase noise
    phase_E = np.zeros(n_samples)
    phase_I = np.zeros(n_samples)
    
    # Initial phases
    phase_E[0] = 0.0
    phase_I[0] = -np.pi / 2  # I starts 90 degrees behind E
    
    # Independent phase noise for each
    noise_E = np.random.randn(n_samples) * noise_std * 2
    noise_I = np.random.randn(n_samples) * noise_std * 2
    
    # Kuramoto-style coupled oscillators
    for t in range(1, n_samples):
        # Phase difference
        phase_diff = phase_I[t-1] - phase_E[t-1]
        
        # E tries to pull I forward, I tries to pull E back
        dphase_E = omega + coupling_strength * np.sin(phase_diff) + noise_E[t]
        dphase_I = omega - coupling_strength * np.sin(phase_diff) + noise_I[t]
        
        phase_E[t] = phase_E[t-1] + dphase_E * dt
        phase_I[t] = phase_I[t-1] + dphase_I * dt
    
    # Convert phases to activity (with some amplitude variation)
    amp_noise_E = 1.0 + 0.1 * np.random.randn(n_samples)
    amp_noise_I = 1.0 + 0.1 * np.random.randn(n_samples)
    
    E = 0.5 + 0.4 * amp_noise_E * np.cos(phase_E)
    I = 0.5 + 0.4 * amp_noise_I * np.cos(phase_I)
    
    # Clip
    E = np.clip(E, 0, 1)
    I = np.clip(I, 0, 1)
    
    # Estimate actual frequency
    _, actual_freq = estimate_ei_oscillation_period(E[1000:], I[1000:], fs)
    
    # Estimate phase coherence
    phase_diff_mean = np.mean(np.abs(np.sin(phase_E[1000:] - phase_I[1000:])))
    
    params = {
        'model': 'ei_oscillatory_kuramoto',
        'n_samples': n_samples,
        'fs': fs,
        'target_freq_hz': target_freq_hz,
        'actual_freq_hz': actual_freq,
        'coupling_strength': coupling_strength,
        'phase_coherence': 1 - phase_diff_mean,  # 1=locked, 0=independent
        'noise_std': noise_std,
        'seed': seed
    }
    
    return E, I, params


def estimate_ei_oscillation_period(E: np.ndarray, I: np.ndarray, fs: float) -> float:
    """
    Estimate the dominant oscillation period of E-I dynamics.
    
    Returns period in samples.
    """
    from scipy.signal import welch
    from scipy.ndimage import gaussian_filter1d
    
    # Use E for frequency estimation
    freqs, psd = welch(E - E.mean(), fs=fs, nperseg=min(len(E)//4, 1024))
    
    # Smooth and find peak (excluding DC)
    psd_smooth = gaussian_filter1d(psd, sigma=2)
    peak_idx = np.argmax(psd_smooth[1:]) + 1  # Skip DC
    peak_freq = freqs[peak_idx]
    
    if peak_freq > 0:
        period_samples = int(fs / peak_freq)
    else:
        period_samples = 100  # Default fallback
    
    return period_samples, peak_freq


def get_ei_predictions(delay_ms: float, fs: float, oscillation_period_ms: Optional[float] = None) -> Dict:
    """
    Get theoretical PID predictions for E-I model.
    """
    delay_samples = int(delay_ms * fs / 1000)
    
    if oscillation_period_ms is not None:
        osc_samples = int(oscillation_period_ms * fs / 1000)
    else:
        # Typical for E-I: ~20-50 Hz oscillation
        osc_samples = int(fs / 30)  # ~33 ms at 30 Hz
    
    predictions = {
        'synaptic_delay': delay_samples,
        'synaptic_delay_ms': delay_ms,
        'expected_oscillation_period': osc_samples,
        
        'within_E_predictions': [
            {
                'tau1': delay_samples,
                'tau2': delay_samples,
                'expected': 'HIGH_REDUNDANCY',
                'reason': 'Direct recurrence at synaptic delay'
            },
            {
                'tau1': osc_samples // 2,
                'tau2': osc_samples,
                'expected': 'SYNERGY',
                'reason': 'Half-period and full-period phases'
            },
            {
                'tau1': osc_samples,
                'tau2': osc_samples,
                'expected': 'HIGH_REDUNDANCY',
                'reason': 'Same phase of oscillation'
            }
        ],
        
        'cross_EI_predictions': [
            {
                'description': 'E(t-τ) and I(t-τ) → E(t)',
                'balanced': 'HIGH_SYNERGY - need both to predict',
                'e_dominant': 'UNIQUE_E dominant',
                'at_oscillation': 'REDUNDANCY - anti-phase predicts same info'
            }
        ],
        
        'parameter_effects': {
            'wEI_increase': 'Shifts toward synergy (I becomes essential)',
            'wEE_increase': 'Shifts toward unique_E (self-recurrence dominates)',
            'noise_increase': 'Reduces total information, increases relative synergy'
        }
    }
    
    return predictions


def test_ei_model(verbose: bool = True) -> bool:
    """
    Test the E-I model against theoretical predictions.
    """
    from scipy.stats import pearsonr
    
    tests_passed = True
    
    # Test 1: Model produces oscillations
    E, I, params = simulate_ei_population(
        n_samples=20000, wEE=1.5, wEI=1.2, wIE=1.0, wII=0.3, seed=42
    )
    
    period_samples, peak_freq = estimate_ei_oscillation_period(E, I, params['fs'])
    
    if 5 < peak_freq < 100:  # Reasonable oscillation range
        if verbose:
            print(f"✓ Test 1 PASSED: Oscillation detected at {peak_freq:.1f} Hz")
    else:
        if verbose:
            print(f"✗ Test 1 FAILED: No clear oscillation (peak = {peak_freq:.1f} Hz)")
        tests_passed = False
    
    # Test 2: E and I are anti-correlated at zero lag (signature of E-I dynamics)
    corr_zero, _ = pearsonr(E[1000:], I[1000:])  # Skip transient
    if corr_zero < 0.5:  # Could be anti-correlated or phase-shifted
        if verbose:
            print(f"✓ Test 2 PASSED: E-I correlation = {corr_zero:.3f} (not strongly positive)")
    else:
        if verbose:
            print(f"? Test 2 NOTE: E-I correlation = {corr_zero:.3f} (positive, may indicate strong drive)")
    
    # Test 3: E and I both have temporal structure
    acf_E = np.correlate(E - E.mean(), E - E.mean(), mode='full')
    acf_E = acf_E[len(acf_E)//2:len(acf_E)//2 + 100] / acf_E[len(acf_E)//2]
    
    if np.min(acf_E) < 0.5:  # ACF should decay or oscillate
        if verbose:
            print("✓ Test 3 PASSED: E has temporal structure (ACF decays/oscillates)")
    else:
        if verbose:
            print("✗ Test 3 FAILED: E lacks temporal structure")
        tests_passed = False
    
    # Test 4: Stability check
    if np.all(np.isfinite(E)) and np.all(np.isfinite(I)):
        if verbose:
            print("✓ Test 4 PASSED: Dynamics stable (no NaN/Inf)")
    else:
        if verbose:
            print("✗ Test 4 FAILED: Unstable dynamics")
        tests_passed = False
    
    # Test 5: Bounded activity
    if 0 <= E.min() and E.max() <= 1 and 0 <= I.min() and I.max() <= 1:
        if verbose:
            print("✓ Test 5 PASSED: Activity bounded in [0, 1]")
    else:
        if verbose:
            print(f"✗ Test 5 FAILED: Activity out of bounds (E: [{E.min():.2f}, {E.max():.2f}], I: [{I.min():.2f}, {I.max():.2f}])")
        tests_passed = False
    
    return tests_passed


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def discretize_for_pid(signal: np.ndarray, n_bins: int = 8, method: str = 'quantile') -> np.ndarray:
    """
    Discretize continuous signal for PID computation.
    
    Parameters
    ----------
    signal : np.ndarray
        Continuous signal
    n_bins : int
        Number of discrete states
    method : str
        'quantile' (equal count) or 'uniform' (equal width)
    
    Returns
    -------
    discrete : np.ndarray
        Integer array with values 0 to n_bins-1
    """
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(signal, percentiles)
        discrete = np.digitize(signal, bin_edges[1:-1])
    elif method == 'uniform':
        bin_edges = np.linspace(signal.min(), signal.max() + 1e-10, n_bins + 1)
        discrete = np.digitize(signal, bin_edges[1:-1])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return discrete


def extract_lagged_windows(signal: np.ndarray, lag1: int, lag2: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract aligned windows for PID: (x_{t-lag1}, x_{t-lag2}) → x_t
    
    Returns
    -------
    source1, source2, target : aligned arrays
    """
    max_lag = max(lag1, lag2)
    n = len(signal) - max_lag
    
    target = signal[max_lag:]
    source1 = signal[max_lag - lag1:max_lag - lag1 + n]
    source2 = signal[max_lag - lag2:max_lag - lag2 + n]
    
    return source1, source2, target


def extract_cross_lagged_windows(
    signal1: np.ndarray, 
    signal2: np.ndarray, 
    lag: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract windows for cross-signal PID: (signal1_{t-lag}, signal2_{t-lag}) → signal1_t
    
    Returns
    -------
    source1, source2, target : aligned arrays
    """
    target = signal1[lag:]
    source1 = signal1[:-lag]
    source2 = signal2[:-lag]
    
    return source1, source2, target


# =============================================================================
# MAIN: RUN TESTS
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("NEURAL MASS MODEL TESTS")
    print("=" * 60)
    
    print("\n--- Single Population with Delayed Feedback ---")
    test1_passed = test_single_population_model(verbose=True)
    
    print("\n--- Excitatory-Inhibitory Population ---")
    test2_passed = test_ei_model(verbose=True)
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
