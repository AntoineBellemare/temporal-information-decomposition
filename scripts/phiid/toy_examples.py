"""
Toy PhiID Analysis - Version 3 (Takens Embedding)
==================================================

Uses TRUE Takens delay embedding with a SINGLE lag parameter (tau_embed).
This solves the two-lag problem from v1/v2.

KEY IMPROVEMENT:
- v2: Irregular sampling with tau + extra_lag (two parameters, confusing)
- v3: Regular sampling: t, t+τ, t+2τ, t+3τ (one parameter, clean)

Timeline comparison:
- v2: t-tau, t, t+extra_lag-tau, t+extra_lag  (irregular spacing)
- v3: t, t+τ, t+2τ, t+3τ                      (perfectly regular!)

Direct PhiID computation:
- Bypasses calc_PhiID() to use internal functions
- Constructs 4-vector embedding directly
- Cleaner interpretation: tau IS the timescale being probed

Toy Processes with PREDICTIONS:
================================

BASELINE (no temporal structure):
- IID_Binary:  All atoms ≈ 0
- IID_Gauss:   All atoms ≈ 0

MEMORY/STORAGE (redundant predictability):
- COPY:        High rtr (redundant transfer), high xtx/yty (self-storage)
- AR(1):       Similar to COPY, scales with coefficient phi
- Markov:      Storage proportional to persistence (p_stay)

SYNERGISTIC (requires multiple timepoints):
- XOR:         High synergy atoms (str, sts), requires t-1 AND t-2
- AR(2):       Mix of redundancy + some synergy from 2-step dependence

OSCILLATORY (periodic structure):
- Sine wave:   Periodic patterns in atoms, peak at period/4
- Damped:      Decaying memory structure

NONLINEAR:
- Logistic:    Chaotic dynamics, complex atom patterns
- Threshold:   Nonlinear transformation of AR process

Usage:
    python toy_phiid_analysis_v3.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import PhiID internal functions (bypass calc_PhiID)
from phyid.calculate import (
    _get_entropy_four_vec,
    _get_coinfo_four_vec,
    _get_redundancy_four_vec,
    _get_double_redundancy_four_vec,
    _get_atoms_four_vec
)
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "toy_systems_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plot quality - use 300 DPI for publication-quality large figures
FIGURE_DPI = 300

# PhiID atom names (16 atoms in 4x4 grid)
ATOM_NAMES = ['rtr', 'rtx', 'rty', 'rts',   # Redundancy source row
              'xtr', 'xtx', 'xty', 'xts',   # X-unique source row
              'ytr', 'ytx', 'yty', 'yts',   # Y-unique source row
              'str', 'stx', 'sty', 'sts']   # Synergy source row

# Information Dynamics groupings
DYNAMICS_GROUPS = {
    'Storage': ['rtr', 'xtx', 'yty', 'sts'],      # Info preserved over time
    'Copy': ['xtx', 'yty'],                        # Within-process continuity
    'Transfer': ['xty', 'ytx'],                    # Cross-process flow
    'Erasure': ['rtx', 'rty'],                     # Lost information
    'Upward_causation': ['xts', 'yts', 'rts'],    # Parts → synergy
    'Downward_causation': ['stx', 'sty', 'str'],  # Synergy → parts
}

# IIT-inspired aggregate metrics
IIT_METRICS = {
    'Phi_R': ['rtr'],                              # Double redundancy (integration core)
    'Phi_S': ['sts'],                              # Synergistic storage
    'Total_redundancy': ['rtr', 'xtr', 'ytr', 'str', 'rtx', 'rty', 'rts'],
    'Total_synergy': ['str', 'stx', 'sty', 'sts', 'xts', 'yts', 'rts'],
}


# =============================================================================
# DATA CLASSES FOR ORGANIZATION
# =============================================================================

@dataclass
class ToyProcess:
    """Container for a toy process with metadata."""
    name: str
    signal: np.ndarray
    category: str           # 'baseline', 'memory', 'synergistic', 'oscillatory', 'nonlinear'
    kind: str               # 'gaussian' or 'discrete'
    description: str
    expected_pattern: str   # What we expect to see in PhiID atoms
    color: str = '#333333'  # For plotting


# =============================================================================
# TOY PROCESS GENERATORS
# =============================================================================

# --- BASELINE PROCESSES (no temporal structure) ---

def generate_iid_binary(n_samples=10000, seed=42) -> np.ndarray:
    """IID binary - no temporal structure at all."""
    np.random.seed(seed)
    return np.random.randint(0, 2, n_samples).astype(float)


def generate_iid_gaussian(n_samples=10000, seed=42) -> np.ndarray:
    """IID Gaussian white noise - no temporal structure."""
    np.random.seed(seed)
    return np.random.randn(n_samples)


# --- MEMORY/STORAGE PROCESSES (redundant temporal structure) ---

def generate_copy_process(n_samples=10000, seed=42) -> np.ndarray:
    """
    Perfect COPY: x[t] = x[t-1]
    All information is redundantly carried forward.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.randn()
    for t in range(1, n_samples):
        x[t] = x[t-1]
    return x


def generate_noisy_copy(n_samples=10000, noise_level=0.1, seed=42) -> np.ndarray:
    """
    Noisy COPY (random walk): x[t] = x[t-1] + noise
    Information gradually accumulated and preserved.
    """
    np.random.seed(seed)
    return np.cumsum(np.random.randn(n_samples) * noise_level)


def generate_ar1(n_samples=10000, phi=0.9, seed=42) -> np.ndarray:
    """
    AR(1): x[t] = phi * x[t-1] + noise
    
    Memory decays exponentially with timescale ~ 1/(1-phi).
    Higher phi = more redundancy.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise_std = np.sqrt(1 - phi**2)  # Stationary variance = 1
    noise = np.random.randn(n_samples) * noise_std
    x[0] = np.random.randn()
    for t in range(1, n_samples):
        x[t] = phi * x[t-1] + noise[t]
    return x


def generate_markov_chain(n_samples=10000, p_stay=0.9, seed=42) -> np.ndarray:
    """
    Binary Markov chain: P(x[t]=x[t-1]) = p_stay
    
    Higher p_stay = more memory/redundancy.
    This doesn't collapse like COPY (has randomness).
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    for t in range(1, n_samples):
        if np.random.rand() < p_stay:
            x[t] = x[t-1]
        else:
            x[t] = 1 - x[t-1]
    return x.astype(float)


# --- SYNERGISTIC PROCESSES (require multiple timepoints) ---

def generate_xor_process(n_samples=10000, noise_prob=0.05, seed=42) -> np.ndarray:
    """
    XOR: x[t] = x[t-1] XOR x[t-2] (with small noise)
    
    SYNERGISTIC: knowing just t-1 OR t-2 gives NO info about t.
    Must know BOTH to predict. This is the canonical synergy example.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        if np.random.rand() < noise_prob:
            x[t] = np.random.randint(0, 2)
        else:
            x[t] = x[t-1] ^ x[t-2]
    return x.astype(float)


def generate_ar2(n_samples=10000, phi1=0.5, phi2=0.3, seed=42) -> np.ndarray:
    """
    AR(2): x[t] = phi1*x[t-1] + phi2*x[t-2] + noise
    
    Has BOTH redundancy (from phi1) and synergy (from phi2).
    The 2-step dependence creates synergistic structure.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise_var = max(0.1, 1 - phi1**2 - phi2**2 - 2*phi1**2*phi2/(1-phi2))
    noise = np.random.randn(n_samples) * np.sqrt(noise_var)
    x[0], x[1] = np.random.randn(2)
    for t in range(2, n_samples):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + noise[t]
    return x


def generate_parity_3(n_samples=10000, noise_prob=0.05, seed=42) -> np.ndarray:
    """
    3-way parity: x[t] = x[t-1] XOR x[t-2] XOR x[t-3]
    
    Even more synergistic - requires 3 past values.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0:3] = np.random.randint(0, 2, 3)
    for t in range(3, n_samples):
        if np.random.rand() < noise_prob:
            x[t] = np.random.randint(0, 2)
        else:
            x[t] = x[t-1] ^ x[t-2] ^ x[t-3]
    return x.astype(float)


# --- OSCILLATORY PROCESSES (periodic structure) ---

def generate_sine_wave(n_samples=10000, period=20, noise_level=0.2, seed=42) -> np.ndarray:
    """
    Noisy sine wave with period T.
    
    Atoms should show periodic patterns with tau_embed.
    Peak information at tau_embed ≈ period/4 (quadrature).
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    return np.sin(2 * np.pi * t / period) + noise_level * np.random.randn(n_samples)


def generate_damped_oscillation(n_samples=10000, period=20, decay=0.01, seed=42) -> np.ndarray:
    """
    Damped oscillation: decaying sine wave.
    
    Memory decreases over time as oscillation dies out.
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    envelope = np.exp(-decay * t)
    return envelope * np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(n_samples)


def generate_ar2_oscillatory(n_samples=10000, freq=0.1, damping=0.95, seed=42) -> np.ndarray:
    """
    AR(2) tuned to oscillate at specified frequency.
    
    phi1 = 2*damping*cos(2*pi*freq), phi2 = -damping^2
    """
    np.random.seed(seed)
    phi1 = 2 * damping * np.cos(2 * np.pi * freq)
    phi2 = -damping**2
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * 0.3
    x[0], x[1] = np.random.randn(2)
    for t in range(2, n_samples):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + noise[t]
    return x


# --- NONLINEAR PROCESSES ---

def generate_logistic_map(n_samples=10000, r=3.9, seed=42) -> np.ndarray:
    """
    Logistic map: x[t+1] = r * x[t] * (1 - x[t])
    
    At r=3.9, this is chaotic - deterministic but unpredictable.
    Complex temporal structure with nonlinear dependencies.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.rand() * 0.5 + 0.25  # Start in [0.25, 0.75]
    for t in range(1, n_samples):
        x[t] = r * x[t-1] * (1 - x[t-1])
    return x


def generate_threshold_ar(n_samples=10000, phi=0.7, threshold=0.5, seed=42) -> np.ndarray:
    """
    Threshold AR: x[t] = phi*x[t-1] + noise, then threshold.
    
    Nonlinear transformation creates complex dependencies.
    """
    np.random.seed(seed)
    x_cont = generate_ar1(n_samples, phi, seed)
    # Threshold at median
    med = np.median(x_cont)
    return (x_cont > med).astype(float)


def generate_henon_map(n_samples=10000, a=1.4, b=0.3, seed=42) -> np.ndarray:
    """
    Hénon map (x-component): x[t+1] = 1 - a*x[t]^2 + y[t], y[t+1] = b*x[t]
    
    2D chaotic system - use x-component only.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    y = np.zeros(n_samples)
    x[0] = np.random.rand() * 0.2
    y[0] = np.random.rand() * 0.2
    for t in range(1, n_samples):
        x[t] = 1 - a * x[t-1]**2 + y[t-1]
        y[t] = b * x[t-1]
    return x


# =============================================================================
# TAKENS EMBEDDING PhiID (Single lag parameter!)
# =============================================================================

def takens_phiid(signal: np.ndarray, tau_embed: int, kind: str = 'gaussian',
                  redundancy: str = 'MMI') -> Optional[Dict[str, float]]:
    """
    Compute PhiID using TRUE Takens delay embedding.
    
    Creates 4 vectors with PERFECTLY REGULAR spacing:
        p1 = x(t)           "X past"
        p2 = x(t + τ)       "Y past"  
        t1 = x(t + 2τ)      "X future"
        t2 = x(t + 3τ)      "Y future"
    
    Timeline: t → t+τ → t+2τ → t+3τ
    Spacing:    [τ]   [τ]    [τ]   ← PERFECTLY REGULAR!
    
    Parameters
    ----------
    signal : array
        1D time series
    tau_embed : int
        Embedding delay (the ONLY temporal parameter!)
    kind : str
        'gaussian' for continuous, 'discrete' for binary
    redundancy : str
        Redundancy measure ('MMI' or 'CCS')
    
    Returns
    -------
    atoms : dict
        16 PhiID atoms (scalar means)
    """
    N = len(signal) - 3 * tau_embed
    
    if N < 50:  # Need minimum samples for reliable estimation
        return None
    
    # Construct 4D Takens embedding
    X = np.zeros((4, N))
    X[0] = signal[0:N]                          # p1 = x(t)
    X[1] = signal[tau_embed:N+tau_embed]        # p2 = x(t+τ)
    X[2] = signal[2*tau_embed:N+2*tau_embed]    # t1 = x(t+2τ)
    X[3] = signal[3*tau_embed:N+3*tau_embed]    # t2 = x(t+3τ)
    
    # Normalize for Gaussian
    if kind == 'gaussian':
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, ddof=1, keepdims=True)
        stds = np.maximum(stds, 1e-10)  # Avoid division by zero
        X = (X - means) / stds
    
    try:
        # PhiID computation pipeline
        h_res = _get_entropy_four_vec(X, kind=kind)
        I_res = _get_coinfo_four_vec(h_res)
        R_res = _get_redundancy_four_vec(redundancy, I_res)
        
        calc_res = {"h_res": h_res, "I_res": I_res, "R_res": R_res}
        
        rtr = _get_double_redundancy_four_vec(redundancy, calc_res)
        calc_res["rtr"] = rtr
        
        atoms_res = _get_atoms_four_vec(calc_res)
        
        # Convert to scalar means
        atoms = {}
        for name in ATOM_NAMES:
            if name in atoms_res:
                val = atoms_res[name]
                if isinstance(val, np.ndarray):
                    val = np.nanmean(val)
                atoms[name] = float(val) if np.isfinite(val) else 0.0
            else:
                atoms[name] = 0.0
        
        return atoms
        
    except Exception as e:
        print(f"      PhiID computation error: {e}")
        return None


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_process(process: ToyProcess, tau_values: List[int]) -> pd.DataFrame:
    """Analyze a process across multiple tau values."""
    results = []
    
    for tau in tau_values:
        atoms = takens_phiid(process.signal, tau, kind=process.kind)
        if atoms is None:
            continue
        
        # Add metadata
        atoms['tau_embed'] = tau
        atoms['process'] = process.name
        atoms['category'] = process.category
        atoms['kind'] = process.kind
        results.append(atoms)
    
    return pd.DataFrame(results)


def compute_dynamics(row: pd.Series) -> Dict[str, float]:
    """Compute aggregate dynamics metrics for a row."""
    dynamics = {}
    for metric, atom_list in DYNAMICS_GROUPS.items():
        dynamics[metric] = sum(row.get(a, 0) for a in atom_list)
    for metric, atom_list in IIT_METRICS.items():
        dynamics[metric] = sum(row.get(a, 0) for a in atom_list)
    return dynamics


def theoretical_ar1_autocorr(phi: float, tau: int) -> float:
    """Theoretical autocorrelation for AR(1) at lag tau: rho = phi^tau"""
    return phi ** tau


def plot_theoretical_comparison(df: pd.DataFrame, save_path: Path):
    """
    Compare actual PhiID results with theoretical predictions for AR(1) processes.
    
    For AR(1) with coefficient φ:
    - Autocorrelation at lag τ: ρ(τ) = φ^τ
    - Expected: rtr should scale with correlation strength
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Get AR1 processes
    ar1_procs = [p for p in df['process'].unique() if 'AR1' in p]
    
    if not ar1_procs:
        plt.close()
        return
    
    # Panel 1: Theoretical autocorrelation vs tau
    ax1 = axes[0, 0]
    tau_range = np.arange(1, 50)
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(ar1_procs)))
    
    phi_map = {'AR1_0.5': 0.5, 'AR1_0.9': 0.9, 'AR1_0.95': 0.95, 'AR1_0.99': 0.99}
    
    for i, proc in enumerate(ar1_procs):
        if proc in phi_map:
            phi = phi_map[proc]
            autocorr = [theoretical_ar1_autocorr(phi, t) for t in tau_range]
            ax1.plot(tau_range, autocorr, '-', color=colors[i], 
                    label=f'{proc} (φ={phi})', linewidth=2)
    
    ax1.set_xlabel('τ (lag in samples)')
    ax1.set_ylabel('Autocorrelation ρ(τ)')
    ax1.set_title('Theoretical Autocorrelation: ρ(τ) = φ^τ', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Panel 2: Actual rtr vs tau for AR1 processes
    ax2 = axes[0, 1]
    for i, proc in enumerate(ar1_procs):
        df_proc = df[df['process'] == proc].sort_values('tau_embed')
        ax2.plot(df_proc['tau_embed'], df_proc['rtr'], 'o-', 
                color=colors[i], label=proc, linewidth=2, markersize=6)
    
    ax2.set_xlabel('τ (lag in samples)')
    ax2.set_ylabel('rtr (double redundancy, bits)')
    ax2.set_title('Actual PhiID: rtr (Double Redundancy)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Panel 3: rtr vs theoretical autocorrelation (should correlate)
    ax3 = axes[1, 0]
    for i, proc in enumerate(ar1_procs):
        if proc in phi_map:
            phi = phi_map[proc]
            df_proc = df[df['process'] == proc]
            theoretical_rho = [phi ** t for t in df_proc['tau_embed']]
            ax3.scatter(theoretical_rho, df_proc['rtr'], 
                       color=colors[i], label=proc, s=60, alpha=0.8)
    
    ax3.set_xlabel('Theoretical ρ(τ) = φ^τ')
    ax3.set_ylabel('Actual rtr (bits)')
    ax3.set_title('Correlation: Theory vs Actual', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Panel 4: Memory timescale comparison
    ax4 = axes[1, 1]
    
    # For each AR1 process, find tau where rtr drops to 50% of peak
    summary_text = "Memory Timescales:\n\n"
    summary_text += "Process    | Theoretical τ_mem | Observed rtr decay\n"
    summary_text += "-" * 50 + "\n"
    
    for proc in ar1_procs:
        if proc in phi_map:
            phi = phi_map[proc]
            # Theoretical: τ_mem ≈ -1/ln(φ) (time constant)
            tau_theoretical = -1 / np.log(phi) if phi < 1 else np.inf
            
            df_proc = df[df['process'] == proc].sort_values('tau_embed')
            rtr_peak = df_proc['rtr'].max()
            
            # Find where rtr drops to 50%
            half_peak = rtr_peak * 0.5
            below_half = df_proc[df_proc['rtr'] < half_peak]
            if len(below_half) > 0 and rtr_peak > 0.01:
                tau_observed = below_half['tau_embed'].iloc[0]
                summary_text += f"{proc:10} | {tau_theoretical:8.1f}        | ~{tau_observed} samples\n"
            else:
                summary_text += f"{proc:10} | {tau_theoretical:8.1f}        | not reached\n"
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
             fontfamily='monospace', verticalalignment='center')
    ax4.axis('off')
    ax4.set_title('Memory Timescale Comparison', fontweight='bold')
    
    plt.suptitle('AR(1) Theoretical vs Actual PhiID', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_prediction_validation(df: pd.DataFrame, save_path: Path):
    """
    Create a summary figure showing whether predictions match observations.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    
    # Get summary at tau=1 and tau=5
    tau1 = df[df['tau_embed'] == 1]
    tau5 = df[df['tau_embed'] == 5]
    
    text = """
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                              PhiID v3 PREDICTION VALIDATION SUMMARY                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                  ║
    ║  CATEGORY        PROCESS         PREDICTION                           RESULT AT τ=1       RESULT AT τ=5  STATUS ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
"""
    
    # Define predictions and checks
    predictions = [
        ('baseline', 'IID_Binary', 'All atoms ≈ 0', 'rtr', lambda x: abs(x) < 0.01),
        ('baseline', 'IID_Gauss', 'All atoms ≈ 0', 'rtr', lambda x: abs(x) < 0.01),
        ('memory', 'AR1_0.9', 'rtr > 0, decays with τ', 'rtr', lambda x: x > 0.02),
        ('memory', 'AR1_0.99', 'High rtr at all τ', 'rtr', lambda x: x > 0.5),
        ('memory', 'NoisyCopy', 'Very high rtr', 'rtr', lambda x: x > 1.0),
        ('synergistic', 'XOR', 'High synergy (sts)', 'sts', lambda x: x > 0.1),
        ('synergistic', 'Parity3', 'Higher synergy', 'sts', lambda x: x > 0.1),
        ('oscillatory', 'Sine_T20', 'Pattern at τ=5 (T/4)', 'sts', lambda x: x > 0.5),
    ]
    
    for cat, proc, pred, atom, check_fn in predictions:
        val1 = tau1[tau1['process'] == proc][atom].values[0] if proc in tau1['process'].values else None
        val5 = tau5[tau5['process'] == proc][atom].values[0] if proc in tau5['process'].values else None
        
        val1_str = f"{val1:.3f}" if val1 is not None else "N/A"
        val5_str = f"{val5:.3f}" if val5 is not None else "N/A"
        
        # Check if prediction matches at tau=5 (main test)
        if val5 is not None:
            status = "✓" if check_fn(val5) else "✗"
        else:
            status = "?"
        
        text += f"    ║  {cat:12}  {proc:15}  {pred:35}  {atom}={val1_str:8}  {atom}={val5_str:8}  {status}    ║\n"
    
    text += """    ║                                                                                                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  INTERPRETATION KEY:                                                                                             ║
    ║                                                                                                                  ║
    ║  • rtr (double redundancy): Information that is shared in past AND future of both X and Y                       ║
    ║  • sts (synergistic storage): Information that requires combining X and Y in past, preserved to future          ║
    ║  • Values at τ=1 capture fast dynamics; values at τ=5 capture slower dynamics                                   ║
    ║  • AR(1) rtr DECAYS with τ because autocorrelation = φ^τ (exponential decay)                                    ║
    ║  • XOR/Parity show synergy because output requires combining multiple past values                               ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
            fontfamily='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_atoms_grid(df: pd.DataFrame, title: str, save_path: Path):
    """
    4x4 grid showing all 16 PhiID atoms across processes.
    """
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    
    processes = df['process'].unique()
    
    # Use a colormap to assign unique colors to each process
    cmap = plt.cm.get_cmap('tab10') if len(processes) <= 10 else plt.cm.get_cmap('tab20')
    process_colors = {proc: cmap(i % cmap.N) for i, proc in enumerate(processes)}
    
    for idx, atom in enumerate(ATOM_NAMES):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        
        for proc in processes:
            df_proc = df[df['process'] == proc]
            color = process_colors[proc]
            ax.plot(df_proc['tau_embed'], df_proc[atom], 'o-',
                   color=color, label=proc, linewidth=1.5, markersize=4, alpha=0.8)
        
        ax.set_xlabel('τ (samples)')
        ax.set_ylabel('bits')
        ax.set_title(atom, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Only show legend on first plot
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right', ncol=2)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_dynamics_summary(df: pd.DataFrame, title: str, save_path: Path):
    """
    2x3 grid of aggregate information dynamics metrics.
    """
    # Compute dynamics for each row
    dynamics_rows = []
    for _, row in df.iterrows():
        dyn = compute_dynamics(row)
        dyn['tau_embed'] = row['tau_embed']
        dyn['process'] = row['process']
        dyn['category'] = row['category']
        dynamics_rows.append(dyn)
    
    df_dyn = pd.DataFrame(dynamics_rows)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics = list(DYNAMICS_GROUPS.keys())
    processes = df_dyn['process'].unique()
    
    # Use a colormap to assign unique colors to each process
    cmap = plt.cm.get_cmap('tab10') if len(processes) <= 10 else plt.cm.get_cmap('tab20')
    process_colors = {proc: cmap(i % cmap.N) for i, proc in enumerate(processes)}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for proc in processes:
            df_proc = df_dyn[df_dyn['process'] == proc]
            color = process_colors[proc]
            ax.plot(df_proc['tau_embed'], df_proc[metric], 'o-',
                   color=color, label=proc, linewidth=2, markersize=5, alpha=0.8)
        
        ax.set_xlabel('τ (samples)')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right', ncol=2)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_category_comparison(df: pd.DataFrame, save_path: Path):
    """
    Compare categories: averaged atoms by category at each tau.
    """
    # Average within category
    df_cat = df.groupby(['category', 'tau_embed'])[ATOM_NAMES].mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    category_colors = {
        'baseline': '#888888',
        'memory': '#1f77b4',
        'synergistic': '#d62728',
        'oscillatory': '#2ca02c',
        'nonlinear': '#9467bd'
    }
    
    # Key atoms to compare
    key_atoms = [
        ('rtr', 'Double Redundancy (rtr)\n"Shared past → shared future"'),
        ('sts', 'Synergistic Storage (sts)\n"Synergy preserved"'),
        ('xtx', 'X Self-Copy (xtx)\n"X past → X future"'),
        ('str', 'Synergy → Redundancy (str)\n"Synergy collapses"'),
    ]
    
    for idx, (atom, atom_title) in enumerate(key_atoms):
        ax = axes[idx // 2, idx % 2]
        
        for cat in df_cat['category'].unique():
            df_c = df_cat[df_cat['category'] == cat]
            color = category_colors.get(cat, '#333333')
            ax.plot(df_c['tau_embed'], df_c[atom], 'o-', color=color,
                   label=cat.title(), linewidth=2.5, markersize=7)
        
        ax.set_xlabel('τ (samples)', fontsize=11)
        ax.set_ylabel('bits', fontsize=11)
        ax.set_title(atom_title, fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.suptitle('Category Comparison: Key PhiID Atoms', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_heatmap_at_tau(df: pd.DataFrame, tau_value: int, save_path: Path):
    """
    Heatmap of all atoms for all processes at a specific tau.
    """
    df_tau = df[df['tau_embed'] == tau_value].copy()
    if len(df_tau) == 0:
        return
    
    # Set process as index
    df_tau = df_tau.set_index('process')[ATOM_NAMES]
    
    # Sort by category
    category_order = ['baseline', 'memory', 'synergistic', 'oscillatory', 'nonlinear']
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use 3 decimal places to show small but non-zero values (like AR1)
    sns.heatmap(df_tau, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax, cbar_kws={'label': 'bits'}, annot_kws={'size': 7})
    
    ax.set_title(f'PhiID Atoms at τ = {tau_value}\n(Takens Embedding)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('PhiID Atom', fontsize=11)
    ax.set_ylabel('Process', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_synergy_vs_redundancy(df: pd.DataFrame, save_path: Path):
    """
    Scatter plot: total synergy vs total redundancy, colored by category.
    """
    # Compute totals
    plot_data = []
    for _, row in df.iterrows():
        total_syn = sum(row.get(a, 0) for a in ['str', 'stx', 'sty', 'sts', 'xts', 'yts', 'rts'])
        total_red = sum(row.get(a, 0) for a in ['rtr', 'xtr', 'ytr', 'rtx', 'rty'])
        plot_data.append({
            'process': row['process'],
            'category': row['category'],
            'tau_embed': row['tau_embed'],
            'total_synergy': total_syn,
            'total_redundancy': total_red
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    category_colors = {
        'baseline': '#888888',
        'memory': '#1f77b4',
        'synergistic': '#d62728',
        'oscillatory': '#2ca02c',
        'nonlinear': '#9467bd'
    }
    
    for cat in df_plot['category'].unique():
        df_c = df_plot[df_plot['category'] == cat]
        ax.scatter(df_c['total_redundancy'], df_c['total_synergy'],
                  c=category_colors.get(cat, '#333333'), label=cat.title(),
                  s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Total Redundancy (bits)', fontsize=12)
    ax.set_ylabel('Total Synergy (bits)', fontsize=12)
    ax.set_title('Synergy vs Redundancy Across Processes and Lags', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, title='Category')
    ax.grid(alpha=0.3)
    
    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_tau_decay_profiles(df: pd.DataFrame, save_path: Path):
    """
    Show how key atoms decay with tau for each process.
    Normalized to peak value for shape comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    key_metrics = [
        ('Storage', ['rtr', 'xtx', 'yty', 'sts']),
        ('Transfer', ['xty', 'ytx']),
        ('Upward_causation', ['xts', 'yts', 'rts']),
        ('Downward_causation', ['stx', 'sty', 'str']),
    ]
    
    category_colors = {
        'baseline': '#888888',
        'memory': '#1f77b4',
        'synergistic': '#d62728',
        'oscillatory': '#2ca02c',
        'nonlinear': '#9467bd'
    }
    
    for idx, (metric_name, atoms) in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for proc in df['process'].unique():
            df_proc = df[df['process'] == proc].sort_values('tau_embed')
            cat = df_proc['category'].iloc[0]
            color = category_colors.get(cat, '#333333')
            
            # Sum atoms for this metric
            values = df_proc[atoms].sum(axis=1).values
            
            # Normalize to peak (if any signal)
            peak = np.max(np.abs(values))
            if peak > 0.01:
                values_norm = values / peak
            else:
                values_norm = values
            
            ax.plot(df_proc['tau_embed'], values_norm, '-',
                   color=color, label=proc, linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('τ (samples)')
        ax.set_ylabel('Normalized value')
        ax.set_title(f'{metric_name} Decay Profile', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
        
        if idx == 0:
            ax.legend(fontsize=6, loc='upper right', ncol=2)
    
    plt.suptitle('Temporal Decay Profiles (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path.name}")


def create_interpretation_guide(save_path: Path):
    """Create comprehensive interpretation guide figure."""
    fig = plt.figure(figsize=(18, 22))
    
    # Use gridspec for layout
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1.2, 1.2, 1], hspace=0.3, wspace=0.2)
    
    # --- Panel 1: Takens Embedding Diagram ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('Takens Embedding (v3 Approach)', fontsize=14, fontweight='bold', loc='left')
    
    diagram_text = """
    ┌─────────────────────────────────────────────────────────────┐
    │  TAKENS EMBEDDING: 4 vectors with SINGLE lag parameter τ   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Timeline:   t ────τ────> t+τ ────τ────> t+2τ ────τ────> t+3τ │
    │              │           │            │            │        │
    │              ▼           ▼            ▼            ▼        │
    │            p1=x(t)    p2=x(t+τ)    t1=x(t+2τ)  t2=x(t+3τ)  │
    │            "X past"   "Y past"    "X future"  "Y future"   │
    │                                                             │
    │  Spacing: [τ] [τ] [τ]  ← PERFECTLY REGULAR!                │
    │                                                             │
    │  vs v2: t-τ, t, t+lag-τ, t+lag  ← IRREGULAR (two params)  │
    └─────────────────────────────────────────────────────────────┘
    """
    ax1.text(0.05, 0.95, diagram_text, transform=ax1.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')
    
    # --- Panel 2: Atom Grid Explanation ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('16 PhiID Atoms (4×4 Grid)', fontsize=14, fontweight='bold', loc='left')
    
    grid_text = """
    ┌─────────────────────────────────────────────────────────────┐
    │  Rows = SOURCE (past), Columns = TARGET (future)           │
    ├─────────────────────────────────────────────────────────────┤
    │            │ Redundancy │ X-unique  │ Y-unique  │ Synergy   │
    │            │   target   │  target   │  target   │  target   │
    │  ──────────┼────────────┼───────────┼───────────┼───────────│
    │  Redundancy│    rtr     │    rtx    │    rty    │    rts    │
    │   source   │  (shared→  │ (shared→  │ (shared→  │ (shared→  │
    │            │   shared)  │  X only)  │  Y only)  │  synergy) │
    │  ──────────┼────────────┼───────────┼───────────┼───────────│
    │  X-unique  │    xtr     │    xtx    │    xty    │    xts    │
    │   source   │ (X→shared) │ (X→X)     │ (X→Y)     │ (X→syn)   │
    │  ──────────┼────────────┼───────────┼───────────┼───────────│
    │  Y-unique  │    ytr     │    ytx    │    yty    │    yts    │
    │   source   │ (Y→shared) │ (Y→X)     │ (Y→Y)     │ (Y→syn)   │
    │  ──────────┼────────────┼───────────┼───────────┼───────────│
    │  Synergy   │    str     │    stx    │    sty    │    sts    │
    │   source   │ (syn→share)│ (syn→X)   │ (syn→Y)   │ (syn→syn) │
    └─────────────────────────────────────────────────────────────┘
    """
    ax2.text(0.02, 0.95, grid_text, transform=ax2.transAxes, fontsize=8,
             fontfamily='monospace', verticalalignment='top')
    
    # --- Panel 3: Process Categories and Predictions ---
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    ax3.set_title('Toy Processes: Categories and Predictions', fontsize=14, fontweight='bold', loc='left')
    
    predictions_text = """
    ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║ CATEGORY        │ PROCESS         │ EQUATION                │ EXPECTED PhiID PATTERN                                     ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ BASELINE        │ IID Binary      │ x[t] ~ Bernoulli(0.5)   │ All atoms ≈ 0 (no temporal structure)                      ║
    ║ (no structure)  │ IID Gaussian    │ x[t] ~ N(0,1)           │ All atoms ≈ 0 (baseline for comparison)                    ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ MEMORY          │ COPY            │ x[t] = x[t-1]           │ High rtr, xtx, yty (perfect redundancy)                    ║
    ║ (redundant)     │ AR(1) φ=0.5     │ x[t] = 0.5·x[t-1] + ε   │ Moderate rtr, xtx, yty; decays with τ                      ║
    ║                 │ AR(1) φ=0.9     │ x[t] = 0.9·x[t-1] + ε   │ High rtr, xtx, yty; slow decay with τ                      ║
    ║                 │ Markov p=0.9    │ P(x[t]=x[t-1]) = 0.9    │ Similar to AR(1) but discrete                              ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ SYNERGISTIC     │ XOR             │ x[t] = x[t-1] ⊕ x[t-2]  │ High str, sts, stx, sty (synergy dominates!)               ║
    ║ (non-redundant) │ Parity-3        │ x[t] = ⊕(t-1,t-2,t-3)   │ Even higher synergy (needs 3 past values)                  ║
    ║                 │ AR(2)           │ x[t] = φ₁x[t-1]+φ₂x[t-2]│ Mix: some rtr from φ₁, some synergy from φ₂                ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ OSCILLATORY     │ Sine (T=20)     │ sin(2πt/20) + noise     │ Periodic atoms; peak at τ≈T/4 (quadrature)                 ║
    ║ (periodic)      │ AR(2) osc       │ Tuned for oscillation   │ Similar to sine but stochastic                             ║
    ║                 │ Damped          │ exp(-αt)·sin(ωt)        │ Atoms decay as oscillation dies                            ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║ NONLINEAR       │ Logistic r=3.9  │ x[t+1] = r·x(1-x)       │ Chaotic: complex mix of atoms, sensitive to τ              ║
    ║ (complex)       │ Threshold AR    │ AR(1) + threshold       │ Nonlinear transform creates unique patterns                ║
    ║                 │ Hénon           │ x[t+1] = 1-ax²+y        │ 2D chaos: rich temporal structure                          ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax3.text(0.01, 0.95, predictions_text, transform=ax3.transAxes, fontsize=7.5,
             fontfamily='monospace', verticalalignment='top')
    
    # --- Panel 4: Information Dynamics Groupings ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    ax4.set_title('Information Dynamics Groupings', fontsize=14, fontweight='bold', loc='left')
    
    dynamics_text = """
    ┌────────────────────────────────────────────────────────┐
    │  STORAGE = rtr + xtx + yty + sts                       │
    │    → Information preserved over time                   │
    │    → High for: COPY, AR(1), Markov                     │
    │                                                        │
    │  COPY = xtx + yty                                      │
    │    → Self-prediction (X past → X future, Y→Y)          │
    │    → Measures within-process continuity                │
    │                                                        │
    │  TRANSFER = xty + ytx                                  │
    │    → Cross-prediction (X→Y and Y→X)                    │
    │    → Information flow between processes                │
    │                                                        │
    │  ERASURE = rtx + rty                                   │
    │    → Shared information that gets lost                 │
    │    → Redundancy in past → unique in future             │
    │                                                        │
    │  UPWARD CAUSATION = xts + yts + rts                    │
    │    → Parts → whole (unique/redundant → synergy)        │
    │    → Information integration over time                 │
    │                                                        │
    │  DOWNWARD CAUSATION = stx + sty + str                  │
    │    → Whole → parts (synergy → unique/redundant)        │
    │    → Synergy "breaks apart" into components            │
    └────────────────────────────────────────────────────────┘
    """
    ax4.text(0.02, 0.95, dynamics_text, transform=ax4.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')
    
    # --- Panel 5: Key Insights ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    ax5.set_title('Key Insights for Interpretation', fontsize=14, fontweight='bold', loc='left')
    
    insights_text = """
    ┌────────────────────────────────────────────────────────┐
    │  REDUNDANCY vs SYNERGY                                 │
    │  ━━━━━━━━━━━━━━━━━━━━━━                                 │
    │  • High rtr = past points carry SAME info about future │
    │  • High sts = future requires COMBINING past info      │
    │  • XOR has high synergy because x[t-1] alone tells     │
    │    NOTHING about x[t], but with x[t-2] → perfect info  │
    │                                                        │
    │  τ SCALING                                             │
    │  ━━━━━━━━━━                                             │
    │  • Atoms typically decay with τ (memory fades)         │
    │  • Decay rate depends on process timescale             │
    │  • AR(1) τ_memory ~ 1/(1-φ) samples                    │
    │  • Oscillations peak at τ ≈ period/4                   │
    │                                                        │
    │  WHAT TO LOOK FOR                                      │
    │  ━━━━━━━━━━━━━━━━                                       │
    │  • Baseline (IID): All near zero ✓                     │
    │  • Memory: High rtr, xtx, yty; low synergy ✓           │
    │  • Synergy: High str, sts; low redundancy ✓            │
    │  • Oscillation: Periodic patterns in all atoms ✓       │
    │  • Chaos: Complex, irregular atom profiles ✓           │
    └────────────────────────────────────────────────────────┘
    """
    ax5.text(0.02, 0.95, insights_text, transform=ax5.transAxes, fontsize=9,
             fontfamily='monospace', verticalalignment='top')
    
    # --- Panel 6: Color Legend ---
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    ax6.set_title('Color Coding in Figures', fontsize=14, fontweight='bold', loc='left')
    
    # Draw colored boxes
    categories = [
        ('Baseline', '#888888', 'IID noise - no temporal structure'),
        ('Memory', '#1f77b4', 'Redundant processes - COPY, AR(1), Markov'),
        ('Synergistic', '#d62728', 'Synergistic processes - XOR, Parity'),
        ('Oscillatory', '#2ca02c', 'Periodic processes - Sine, Damped'),
        ('Nonlinear', '#9467bd', 'Chaotic/nonlinear - Logistic, Hénon'),
    ]
    
    for i, (cat_name, color, desc) in enumerate(categories):
        x = 0.05 + (i % 3) * 0.32
        y = 0.6 - (i // 3) * 0.35
        ax6.add_patch(plt.Rectangle((x, y), 0.04, 0.15, facecolor=color, 
                                     transform=ax6.transAxes, edgecolor='black'))
        ax6.text(x + 0.06, y + 0.07, f'{cat_name}: {desc}', transform=ax6.transAxes,
                fontsize=10, verticalalignment='center')
    
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TOY PhiID ANALYSIS v3 - TAKENS EMBEDDING APPROACH")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Parameters
    N_SAMPLES = 12000
    TAU_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
    
    print(f"\nParameters:")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  τ values:  {TAU_VALUES}")
    print(f"  Method:    Takens embedding (single τ, regular spacing)")
    
    # ==========================================================================
    # GENERATE ALL TOY PROCESSES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING TOY PROCESSES")
    print("=" * 80)
    
    processes: List[ToyProcess] = []
    
    # --- BASELINE ---
    print("\n[BASELINE] No temporal structure...")
    processes.append(ToyProcess(
        name="IID_Binary",
        signal=generate_iid_binary(N_SAMPLES),
        category="baseline",
        kind="discrete",
        description="IID Bernoulli(0.5)",
        expected_pattern="All atoms ≈ 0"
    ))
    processes.append(ToyProcess(
        name="IID_Gauss",
        signal=generate_iid_gaussian(N_SAMPLES),
        category="baseline",
        kind="gaussian",
        description="IID N(0,1)",
        expected_pattern="All atoms ≈ 0"
    ))
    
    # --- MEMORY ---
    print("[MEMORY] Redundant temporal structure...")
    processes.append(ToyProcess(
        name="Markov_0.9",
        signal=generate_markov_chain(N_SAMPLES, p_stay=0.9),
        category="memory",
        kind="discrete",
        description="P(stay) = 0.9",
        expected_pattern="High rtr, xtx, yty"
    ))
    processes.append(ToyProcess(
        name="AR1_0.5",
        signal=generate_ar1(N_SAMPLES, phi=0.5),
        category="memory",
        kind="gaussian",
        description="φ = 0.5",
        expected_pattern="Moderate redundancy, fast decay"
    ))
    processes.append(ToyProcess(
        name="AR1_0.9",
        signal=generate_ar1(N_SAMPLES, phi=0.9),
        category="memory",
        kind="gaussian",
        description="φ = 0.9",
        expected_pattern="High redundancy, slow decay"
    ))
    processes.append(ToyProcess(
        name="AR1_0.95",
        signal=generate_ar1(N_SAMPLES, phi=0.95),
        category="memory",
        kind="gaussian",
        description="φ = 0.95",
        expected_pattern="Very high redundancy, very slow decay"
    ))
    processes.append(ToyProcess(
        name="AR1_0.99",
        signal=generate_ar1(N_SAMPLES, phi=0.99),
        category="memory",
        kind="gaussian",
        description="φ = 0.99 (near random walk)",
        expected_pattern="Near-perfect redundancy at all τ"
    ))
    processes.append(ToyProcess(
        name="NoisyCopy",
        signal=generate_noisy_copy(N_SAMPLES, noise_level=0.1),
        category="memory",
        kind="gaussian",
        description="Random walk",
        expected_pattern="Very high rtr (accumulating memory)"
    ))
    
    # --- SYNERGISTIC ---
    print("[SYNERGISTIC] Non-redundant temporal structure...")
    processes.append(ToyProcess(
        name="XOR",
        signal=generate_xor_process(N_SAMPLES, noise_prob=0.05),
        category="synergistic",
        kind="discrete",
        description="x[t] = x[t-1] ⊕ x[t-2]",
        expected_pattern="High str, sts, stx, sty"
    ))
    processes.append(ToyProcess(
        name="Parity3",
        signal=generate_parity_3(N_SAMPLES, noise_prob=0.05),
        category="synergistic",
        kind="discrete",
        description="3-way XOR",
        expected_pattern="Even higher synergy"
    ))
    processes.append(ToyProcess(
        name="AR2",
        signal=generate_ar2(N_SAMPLES, phi1=0.5, phi2=0.3),
        category="synergistic",
        kind="gaussian",
        description="φ₁=0.5, φ₂=0.3",
        expected_pattern="Mix of redundancy and synergy"
    ))
    
    # --- OSCILLATORY ---
    print("[OSCILLATORY] Periodic temporal structure...")
    processes.append(ToyProcess(
        name="Sine_T20",
        signal=generate_sine_wave(N_SAMPLES, period=20, noise_level=0.2),
        category="oscillatory",
        kind="gaussian",
        description="Period=20, noisy",
        expected_pattern="Periodic atoms, peak at τ≈5"
    ))
    processes.append(ToyProcess(
        name="AR2_Osc",
        signal=generate_ar2_oscillatory(N_SAMPLES, freq=0.1, damping=0.95),
        category="oscillatory",
        kind="gaussian",
        description="Stochastic oscillator",
        expected_pattern="Similar to sine but noisier"
    ))
    processes.append(ToyProcess(
        name="Damped",
        signal=generate_damped_oscillation(N_SAMPLES, period=20, decay=0.002),
        category="oscillatory",
        kind="gaussian",
        description="Decaying sine",
        expected_pattern="Atoms decay with signal"
    ))
    
    # --- NONLINEAR ---
    print("[NONLINEAR] Complex dynamics...")
    processes.append(ToyProcess(
        name="Logistic",
        signal=generate_logistic_map(N_SAMPLES, r=3.9),
        category="nonlinear",
        kind="gaussian",
        description="r=3.9 (chaotic)",
        expected_pattern="Complex, irregular patterns"
    ))
    processes.append(ToyProcess(
        name="Henon",
        signal=generate_henon_map(N_SAMPLES, a=1.4, b=0.3),
        category="nonlinear",
        kind="gaussian",
        description="2D chaotic map",
        expected_pattern="Rich temporal structure"
    ))
    processes.append(ToyProcess(
        name="Threshold_AR",
        signal=generate_threshold_ar(N_SAMPLES, phi=0.7),
        category="nonlinear",
        kind="discrete",
        description="Thresholded AR(1)",
        expected_pattern="Nonlinear mix"
    ))
    
    print(f"\nGenerated {len(processes)} toy processes")
    
    # ==========================================================================
    # ANALYZE ALL PROCESSES
    # ==========================================================================
    print("\n" + "=" * 80)
    print("COMPUTING PhiID (Takens Embedding)")
    print("=" * 80)
    
    all_results = []
    
    for proc in processes:
        print(f"\n  Analyzing: {proc.name} ({proc.category}, {proc.kind})")
        print(f"    Description: {proc.description}")
        print(f"    Expected: {proc.expected_pattern}")
        
        df = analyze_process(proc, TAU_VALUES)
        
        if len(df) > 0:
            all_results.append(df)
            print(f"    ✓ Completed {len(df)} τ values")
        else:
            print(f"    ✗ Failed!")
    
    # Combine all results
    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(RESULTS_DIR / "all_toy_results_v3.csv", index=False)
    print(f"\n  Results saved to: all_toy_results_v3.csv")
    
    # ==========================================================================
    # GENERATE VISUALIZATIONS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # 1. Full atom grid (all processes)
    print("\n1. Full 16-atom grid...")
    plot_atoms_grid(all_results, "PhiID Atoms: All Toy Processes (v3 Takens)", 
                    RESULTS_DIR / "atoms_all_v3.png")
    
    # 2. By category
    for cat in ['baseline', 'memory', 'synergistic', 'oscillatory', 'nonlinear']:
        df_cat = all_results[all_results['category'] == cat]
        if len(df_cat) > 0:
            print(f"2. Atom grid: {cat}...")
            plot_atoms_grid(df_cat, f"PhiID Atoms: {cat.title()} Processes",
                           RESULTS_DIR / f"atoms_{cat}_v3.png")
    
    # 3. Dynamics summary
    print("3. Information dynamics summary...")
    plot_dynamics_summary(all_results, "Information Dynamics: All Processes (v3)",
                         RESULTS_DIR / "dynamics_all_v3.png")
    
    # 4. Category comparison
    print("4. Category comparison (key atoms)...")
    plot_category_comparison(all_results, RESULTS_DIR / "category_comparison_v3.png")
    
    # 5. Heatmaps at specific τ values - including τ=1 for fast dynamics
    for tau in [1, 2, 5, 10, 20]:
        if tau in all_results['tau_embed'].values:
            print(f"5. Heatmap at τ={tau}...")
            plot_heatmap_at_tau(all_results, tau, RESULTS_DIR / f"heatmap_tau{tau}_v3.png")
    
    # 6. Synergy vs Redundancy scatter
    print("6. Synergy vs Redundancy scatter...")
    plot_synergy_vs_redundancy(all_results, RESULTS_DIR / "synergy_vs_redundancy_v3.png")
    
    # 7. Decay profiles
    print("7. Temporal decay profiles...")
    plot_tau_decay_profiles(all_results, RESULTS_DIR / "decay_profiles_v3.png")
    
    # 8. Interpretation guide
    print("8. Interpretation guide...")
    create_interpretation_guide(RESULTS_DIR / "interpretation_guide_v3.png")
    
    # 9. Theoretical comparison (predictions vs actual)
    print("9. Theoretical comparison...")
    plot_theoretical_comparison(all_results, RESULTS_DIR / "theoretical_comparison_v3.png")
    
    # 10. Prediction validation matrix
    print("10. Prediction validation...")
    plot_prediction_validation(all_results, RESULTS_DIR / "prediction_validation_v3.png")
    
    # ==========================================================================
    # SUMMARY STATISTICS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Mean atoms at τ = 5")
    print("=" * 80)
    
    tau5 = all_results[all_results['tau_embed'] == 5]
    if len(tau5) > 0:
        summary = tau5.groupby('process')[ATOM_NAMES].mean()
        
        # Compute key metrics
        summary['Total_Red'] = summary[['rtr', 'xtr', 'ytr', 'rtx', 'rty']].sum(axis=1)
        summary['Total_Syn'] = summary[['str', 'stx', 'sty', 'sts', 'xts', 'yts', 'rts']].sum(axis=1)
        summary['Storage'] = summary[['rtr', 'xtx', 'yty', 'sts']].sum(axis=1)
        
        print("\nKey atoms (rounded to 3 decimals):")
        print(summary[['rtr', 'xtx', 'sts', 'str', 'Total_Red', 'Total_Syn', 'Storage']].round(3).to_string())
    
    # Verification
    print("\n" + "=" * 80)
    print("VERIFICATION: Do results match predictions?")
    print("=" * 80)
    
    if len(tau5) > 0:
        tau5_indexed = tau5.set_index('process')
        
        checks = []
        
        # IID should have near-zero atoms
        for proc in ['IID_Binary', 'IID_Gauss']:
            if proc in tau5_indexed.index:
                total = tau5_indexed.loc[proc, ATOM_NAMES].abs().sum()
                checks.append(f"  {proc}: Total |atoms| = {total:.3f} (expected ≈ 0) {'✓' if total < 0.5 else '?'}")
        
        # AR1 should have high rtr
        for proc in ['AR1_0.5', 'AR1_0.9']:
            if proc in tau5_indexed.index:
                rtr = tau5_indexed.loc[proc, 'rtr']
                checks.append(f"  {proc}: rtr = {rtr:.3f} (expected > 0) {'✓' if rtr > 0.1 else '?'}")
        
        # XOR should have high synergy
        if 'XOR' in tau5_indexed.index:
            sts = tau5_indexed.loc['XOR', 'sts']
            str_val = tau5_indexed.loc['XOR', 'str']
            checks.append(f"  XOR: sts = {sts:.3f}, str = {str_val:.3f} (expected high) {'✓' if sts > 0.1 or str_val > 0.1 else '?'}")
        
        print("\n".join(checks))
    
    print(f"\n\nAll results saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
