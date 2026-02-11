"""
Comprehensive Temporal PhiID Exploration of Neural Mass Models
===============================================================

This script provides systematic exploration of PhiID across:
1. Multiple lags (τ sweep from 5ms to 200ms)
2. All 16 PhiID atoms
3. High-level derived measures (Copy, Transfer, Erasure, Causation, etc.)
4. Different neural mass models
5. Parameter sweeps for each model

Key difference from basic analysis:
- Full τ sweep (not just a few points)
- High-level information dynamics measures
- Comprehensive visualization of all atoms
- Parameter exploration for each model

Usage:
    python neural_mass_phiid_exploration.py [--bins 8]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import sys
import warnings
warnings.filterwarnings('ignore')
import argparse

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'integrated-info-decomp'))

# Import neural mass models from utils
from scripts.utils.neural_mass_models import (
    simulate_single_population_delayed,
    simulate_hierarchical_timescales,
    simulate_ei_population,
    simulate_ei_oscillatory,
    simulate_xor_timescales,
    discretize_for_pid,
)

# Import PhiID internal functions (bypassing calc_PhiID to avoid double-lag issue)
try:
    from phyid.calculate import (
        _get_entropy_four_vec,
        _get_coinfo_four_vec,
        _get_redundancy_four_vec,
        _get_double_redundancy_four_vec,
        _get_atoms_four_vec
    )
except ImportError:
    raise ImportError("phyid library required. Ensure integrated-info-decomp is in path.")


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = project_root / 'results' / 'phiid' / 'neural_mass_exploration'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PhiID atom names (16 atoms)
ATOM_NAMES = [
    "rtr", "rtx", "rty", "rts",  # Redundancy → (redundancy, X, Y, synergy)
    "xtr", "xtx", "xty", "xts",  # X-unique → (redundancy, X, Y, synergy)
    "ytr", "ytx", "yty", "yts",  # Y-unique → (redundancy, X, Y, synergy)
    "str", "stx", "sty", "sts",  # Synergy → (redundancy, X, Y, synergy)
]

# High-level information dynamics measures
INFORMATION_DYNAMICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Transfer_X_to_Y": ["xty"],
    "Transfer_Y_to_X": ["ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward_causation": ["sty", "stx", "str"],
    "Upward_causation": ["xts", "yts", "rts"],
    "Integrated_info": ["rts", "xts", "sts", "sty", "str", "yts", "stx"],
    "Synergy_total": ["sts", "rts", "xts", "yts", "str", "stx", "sty"],
    "Redundancy_total": ["rtr", "rtx", "rty", "rts"],
}


# =============================================================================
# PhiID COMPUTATION (Using Takens embedding - avoiding calc_PhiID double-lag issue)
# =============================================================================

def compute_phiid_4vec(X_4d: np.ndarray, kind: str = 'gaussian', redundancy: str = 'MMI') -> dict:
    """
    Compute PhiID from a pre-constructed 4-vector embedding.
    
    Parameters
    ----------
    X_4d : np.ndarray
        Shape (4, N) where:
        X_4d[0] = p1 (past 1)
        X_4d[1] = p2 (past 2)  
        X_4d[2] = t1 (future 1)
        X_4d[3] = t2 (future 2)
    kind : str
        'gaussian' for continuous, 'discrete' for discrete
    redundancy : str
        'MMI' or 'CCS'
    
    Returns
    -------
    atoms : dict
        Dictionary with all 16 PhiID atom values
    """
    try:
        # Normalize for Gaussian estimation
        if kind == 'gaussian':
            means = np.mean(X_4d, axis=1, keepdims=True)
            stds = np.std(X_4d, axis=1, ddof=1, keepdims=True)
            stds = np.maximum(stds, 1e-10)
            X_4d = (X_4d - means) / stds
        
        # PhiID computation pipeline (direct, bypassing calc_PhiID)
        h_res = _get_entropy_four_vec(X_4d, kind=kind)
        I_res = _get_coinfo_four_vec(h_res)
        R_res = _get_redundancy_four_vec(redundancy, I_res)
        
        calc_res = {"h_res": h_res, "I_res": I_res, "R_res": R_res}
        
        rtr = _get_double_redundancy_four_vec(redundancy, calc_res)
        calc_res["rtr"] = rtr
        
        atoms_res = _get_atoms_four_vec(calc_res)
        
        # Convert to scalar means with numerical cleanup
        atoms = {}
        diagonal_atoms = ['rtr', 'xtx', 'yty', 'sts']  # These should be non-negative
        
        for name in ATOM_NAMES:
            if name in atoms_res:
                val = atoms_res[name]
                if isinstance(val, np.ndarray):
                    val = np.nanmean(val)
                val = float(val) if np.isfinite(val) else 0.0
                
                # Clamp near-zero negative diagonal atoms to zero (numerical precision fix)
                if name in diagonal_atoms and val < 0 and val > -1e-6:
                    val = 0.0
                
                atoms[name] = val
            else:
                atoms[name] = 0.0
        
        return atoms
        
    except Exception as e:
        print(f"    PhiID computation failed: {e}")
        return {name: 0.0 for name in ATOM_NAMES}


def compute_bivariate_phiid_at_lag(
    X: np.ndarray,
    Y: np.ndarray,
    tau: int,
    kind: str = 'gaussian',
    min_samples: int = 100
) -> dict:
    """
    Compute PhiID for two signals (X, Y) at a specific lag using Takens embedding.
    
    Creates 4 vectors with uniform spacing τ:
        p1 = X(t)           "X past"
        p2 = Y(t)           "Y past"  
        t1 = X(t + τ)       "X future"
        t2 = Y(t + τ)       "Y future"
    
    This is the BIVARIATE case: how X and Y together predict their joint future.
    """
    n = min(len(X), len(Y))
    N = n - tau
    
    if N < min_samples:  # Need minimum samples
        return {name: 0.0 for name in ATOM_NAMES}
    
    if N < 500:
        warnings.warn(f"Sample size {N} may be insufficient for reliable PhiID estimation")
    
    # Construct 4D embedding: past(X,Y) → future(X,Y)
    X_4d = np.zeros((4, N))
    X_4d[0] = X[:N]          # p1 = X(t)
    X_4d[1] = Y[:N]          # p2 = Y(t)
    X_4d[2] = X[tau:tau+N]   # t1 = X(t+τ)
    X_4d[3] = Y[tau:tau+N]   # t2 = Y(t+τ)
    
    return compute_phiid_4vec(X_4d, kind=kind)


def compute_takens_phiid_at_lag(
    signal: np.ndarray,
    tau: int,
    kind: str = 'gaussian',
    min_samples: int = 100
) -> dict:
    """
    Compute PhiID using TRUE Takens embedding of a SINGLE signal.
    
    Creates 4 vectors with PERFECTLY REGULAR spacing τ:
        p1 = x(t)           "X past"
        p2 = x(t + τ)       "Y past"  
        t1 = x(t + 2τ)      "X future"
        t2 = x(t + 3τ)      "Y future"
    
    Timeline: t → t+τ → t+2τ → t+3τ (uniform spacing!)
    
    This probes temporal structure at timescale τ within a single signal.
    """
    N = len(signal) - 3 * tau
    
    if N < min_samples:  # Need minimum samples for reliable estimation
        return {name: 0.0 for name in ATOM_NAMES}
    
    if N < 500:
        warnings.warn(f"Sample size {N} may be insufficient for reliable PhiID estimation")
    
    # Construct 4D Takens embedding with PERFECTLY REGULAR spacing
    X_4d = np.zeros((4, N))
    X_4d[0] = signal[0:N]                          # p1 = x(t)
    X_4d[1] = signal[tau:N+tau]                    # p2 = x(t+τ)
    X_4d[2] = signal[2*tau:N+2*tau]                # t1 = x(t+2τ)
    X_4d[3] = signal[3*tau:N+3*tau]                # t2 = x(t+3τ)
    
    return compute_phiid_4vec(X_4d, kind=kind)


def compute_summary_measures(atoms: dict) -> dict:
    """Compute high-level summary measures from PhiID atoms."""
    summary = {}
    
    for measure_name, atom_list in INFORMATION_DYNAMICS.items():
        total = sum(atoms.get(a, 0) for a in atom_list)
        summary[measure_name] = total
    
    # Add specific measures
    summary['Transfer_asymmetry'] = atoms.get('xty', 0) - atoms.get('ytx', 0)
    summary['Synergy_Redundancy_ratio'] = (
        summary['Synergy_total'] / (summary['Redundancy_total'] + 1e-10)
    )
    
    # Phi (integrated info with subtraction of redundant storage)
    summary['Phi'] = summary['Integrated_info'] - atoms.get('rtr', 0)
    
    return summary


# =============================================================================
# LAG SWEEP ANALYSES
# =============================================================================

def analyze_lag_sweep(
    X: np.ndarray,
    Y: np.ndarray,
    lag_range_ms: List[float],
    fs: float = 1000.0,
    model_name: str = 'model',
    kind: str = 'gaussian'
) -> pd.DataFrame:
    """
    Comprehensive lag sweep computing PhiID at each lag.
    
    Returns DataFrame with all atoms and summary measures for each lag.
    """
    results = []
    
    for lag_ms in lag_range_ms:
        tau = int(lag_ms * fs / 1000)
        if tau < 1:
            continue
            
        print(f"    τ = {lag_ms} ms ({tau} samples)...")
        
        atoms = compute_bivariate_phiid_at_lag(X, Y, tau, kind=kind)
        summary = compute_summary_measures(atoms)
        
        row = {
            'model': model_name,
            'tau_ms': lag_ms,
            'tau_samples': tau,
            **atoms,
            **{f'summary_{k}': v for k, v in summary.items()}
        }
        results.append(row)
    
    return pd.DataFrame(results)


def analyze_takens_lag_sweep(
    signal: np.ndarray,
    lag_range_ms: List[float],
    fs: float = 1000.0,
    signal_name: str = 'signal',
    kind: str = 'gaussian'
) -> pd.DataFrame:
    """
    Lag sweep using Takens embedding for single signal analysis.
    """
    results = []
    
    for lag_ms in lag_range_ms:
        tau = int(lag_ms * fs / 1000)
        if tau < 1:
            continue
            
        print(f"    τ = {lag_ms} ms ({tau} samples)...")
        
        atoms = compute_takens_phiid_at_lag(signal, tau, kind=kind)
        summary = compute_summary_measures(atoms)
        
        row = {
            'signal': signal_name,
            'tau_ms': lag_ms,
            'tau_samples': tau,
            **atoms,
            **{f'summary_{k}': v for k, v in summary.items()}
        }
        results.append(row)
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_all_atoms_vs_lag(df: pd.DataFrame, title: str, save_path: Path):
    """Plot all 16 PhiID atoms as a function of lag."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    
    for i, atom in enumerate(ATOM_NAMES):
        ax = axes[i // 4, i % 4]
        if atom in df.columns:
            ax.plot(df['tau_ms'], df[atom], 'o-', linewidth=2, markersize=4)
        ax.set_title(atom, fontsize=12, fontweight='bold')
        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('bits')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'{title}\nAll 16 PhiID Atoms vs Lag', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_measures_vs_lag(df: pd.DataFrame, title: str, save_path: Path):
    """Plot high-level summary measures vs lag."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    measures = [
        ('summary_Storage', 'summary_Copy', 'Storage & Copy'),
        ('summary_Transfer_X_to_Y', 'summary_Transfer_Y_to_X', 'Transfer'),
        ('summary_Erasure', None, 'Erasure'),
        ('summary_Downward_causation', 'summary_Upward_causation', 'Causation'),
        ('summary_Integrated_info', 'summary_Phi', 'Integration'),
        ('summary_Synergy_total', 'summary_Redundancy_total', 'Synergy vs Redundancy'),
    ]
    
    for idx, (m1, m2, label) in enumerate(measures):
        ax = axes[idx // 3, idx % 3]
        
        if m1 in df.columns:
            ax.plot(df['tau_ms'], df[m1], 'o-', linewidth=2, markersize=4, 
                   label=m1.replace('summary_', ''))
        if m2 and m2 in df.columns:
            ax.plot(df['tau_ms'], df[m2], 's-', linewidth=2, markersize=4,
                   label=m2.replace('summary_', ''))
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('bits')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=9)
    
    plt.suptitle(f'{title}\nHigh-Level Information Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_atom_heatmap(df: pd.DataFrame, title: str, save_path: Path):
    """Heatmap of all atoms across lags."""
    atom_data = df[ATOM_NAMES].values.T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    sns.heatmap(atom_data, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=[f"{int(t)}" for t in df['tau_ms']],
                yticklabels=ATOM_NAMES,
                annot=True, fmt='.3f', annot_kws={'size': 8})
    
    ax.set_xlabel('τ (ms)', fontsize=12)
    ax.set_ylabel('PhiID Atom', fontsize=12)
    ax.set_title(f'{title}\nPhiID Atoms Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_parameter_sweep_summary(
    df: pd.DataFrame, 
    param_col: str,
    title: str, 
    save_path: Path
):
    """Plot summary measures across parameter values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    summary_cols = [
        'summary_Storage', 'summary_Copy', 'summary_Transfer',
        'summary_Integrated_info', 'summary_Downward_causation', 'summary_Upward_causation'
    ]
    
    # Get unique lags and parameters
    lags = df['tau_ms'].unique()
    params = df[param_col].unique()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(params)))
    
    for idx, col in enumerate(summary_cols):
        ax = axes[idx // 3, idx % 3]
        
        for i, param in enumerate(params):
            df_param = df[df[param_col] == param].sort_values('tau_ms')
            if col in df_param.columns:
                ax.plot(df_param['tau_ms'], df_param[col], 'o-', 
                       color=colors[i], linewidth=2, markersize=4,
                       label=f'{param_col}={param:.2f}')
        
        ax.set_title(col.replace('summary_', ''), fontsize=11, fontweight='bold')
        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('bits')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MODEL-SPECIFIC ANALYSES
# =============================================================================

def analyze_hierarchical_model(
    tau_fast_ms: float = 5.0,
    tau_slow_ms: float = 50.0,
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    Comprehensive PhiID analysis of hierarchical timescales model.
    """
    print("\n" + "="*70)
    print("HIERARCHICAL TIMESCALES MODEL: PhiID Exploration")
    print("="*70)
    print(f"τ_fast = {tau_fast_ms} ms, τ_slow = {tau_slow_ms} ms")
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    
    # Simulate
    x_fast, x_slow, params = simulate_hierarchical_timescales(
        n_samples=n_samples, fs=fs,
        tau_fast_ms=tau_fast_ms, tau_slow_ms=tau_slow_ms,
        w_cross_up=0.4, w_cross_down=0.3,
        seed=42
    )
    
    # Cross-population PhiID
    print("\nCross-population PhiID (fast ↔ slow):")
    df_cross = analyze_lag_sweep(x_fast, x_slow, lag_range_ms, fs, 
                                  model_name='hierarchical_cross')
    
    # Within-population Takens PhiID
    print("\nWithin-fast Takens PhiID:")
    df_fast = analyze_takens_lag_sweep(x_fast, lag_range_ms, fs, 
                                        signal_name='hierarchical_fast')
    
    print("\nWithin-slow Takens PhiID:")
    df_slow = analyze_takens_lag_sweep(x_slow, lag_range_ms, fs,
                                        signal_name='hierarchical_slow')
    
    # Combine
    df_cross['analysis_type'] = 'cross'
    df_fast['analysis_type'] = 'within_fast'
    df_slow['analysis_type'] = 'within_slow'
    
    df_all = pd.concat([df_cross, df_fast, df_slow], ignore_index=True)
    
    if save_dir:
        # Save data
        df_all.to_csv(save_dir / 'hierarchical_phiid_full.csv', index=False)
        
        # Plots
        plot_all_atoms_vs_lag(df_cross, 'Hierarchical: Fast↔Slow',
                             save_dir / 'hierarchical_cross_atoms.png')
        plot_summary_measures_vs_lag(df_cross, 'Hierarchical: Fast↔Slow',
                                     save_dir / 'hierarchical_cross_summary.png')
        plot_atom_heatmap(df_cross, 'Hierarchical: Fast↔Slow',
                         save_dir / 'hierarchical_cross_heatmap.png')
        
        plot_all_atoms_vs_lag(df_fast, 'Hierarchical: Within-Fast (Takens)',
                             save_dir / 'hierarchical_fast_atoms.png')
        plot_all_atoms_vs_lag(df_slow, 'Hierarchical: Within-Slow (Takens)',
                             save_dir / 'hierarchical_slow_atoms.png')
    
    return df_all


def analyze_ei_model(
    wEE: float = 2.0,
    wEI: float = 1.5,
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    Comprehensive PhiID analysis of E-I population model.
    """
    print("\n" + "="*70)
    print("E-I POPULATION MODEL: PhiID Exploration")
    print("="*70)
    print(f"wEE = {wEE}, wEI = {wEI}")
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    
    # Simulate
    E, I, params = simulate_ei_population(
        n_samples=n_samples, fs=fs,
        wEE=wEE, wEI=wEI, wIE=1.5, wII=0.3,
        tau_E=8.0, tau_I=16.0,
        noise_std=0.1, seed=42
    )
    
    # Cross-population PhiID
    print("\nE↔I PhiID:")
    df = analyze_lag_sweep(E, I, lag_range_ms, fs, model_name='ei')
    
    if save_dir:
        df.to_csv(save_dir / 'ei_phiid_full.csv', index=False)
        plot_all_atoms_vs_lag(df, 'E-I Model', save_dir / 'ei_atoms.png')
        plot_summary_measures_vs_lag(df, 'E-I Model', save_dir / 'ei_summary.png')
        plot_atom_heatmap(df, 'E-I Model', save_dir / 'ei_heatmap.png')
    
    return df


def analyze_ei_coupling_sweep(
    coupling_strengths: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    PhiID analysis across E-I coupling strengths.
    """
    print("\n" + "="*70)
    print("E-I COUPLING SWEEP: PhiID vs Coupling Strength")
    print("="*70)
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 20, 30, 50, 75, 100, 150]
    
    all_results = []
    
    for coupling in coupling_strengths:
        print(f"\nCoupling = {coupling}")
        
        E, I, params = simulate_ei_oscillatory(
            n_samples=n_samples, fs=fs,
            target_freq_hz=10.0,
            coupling_strength=coupling,
            noise_std=0.05, seed=42
        )
        
        df = analyze_lag_sweep(E, I, lag_range_ms, fs, model_name=f'ei_coupling_{coupling}')
        df['coupling'] = coupling
        all_results.append(df)
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    if save_dir:
        df_all.to_csv(save_dir / 'ei_coupling_sweep_full.csv', index=False)
        plot_parameter_sweep_summary(df_all, 'coupling', 
                                     'E-I Coupling Sweep: PhiID Dynamics',
                                     save_dir / 'ei_coupling_sweep_summary.png')
    
    return df_all


def analyze_single_population_takens(
    delay_ms: float = 20.0,
    gains: List[float] = [0.5, 1.0, 2.0, 4.0],
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    PhiID using Takens embedding for single population with different gains.
    """
    print("\n" + "="*70)
    print("SINGLE POPULATION TAKENS: PhiID vs Nonlinearity")
    print("="*70)
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    all_results = []
    
    for gain in gains:
        print(f"\nGain = {gain}, Delay = {delay_ms} ms")
        
        x, params = simulate_single_population_delayed(
            n_samples=n_samples, fs=fs,
            delay_ms=delay_ms, weight=0.85,
            noise_std=0.1, activation='tanh',
            gain=gain, seed=42
        )
        
        df = analyze_takens_lag_sweep(x, lag_range_ms, fs, 
                                      signal_name=f'single_gain_{gain}')
        df['gain'] = gain
        df['delay_ms'] = delay_ms
        all_results.append(df)
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    if save_dir:
        df_all.to_csv(save_dir / 'single_population_takens_full.csv', index=False)
        plot_parameter_sweep_summary(df_all, 'gain',
                                     'Single Population: PhiID vs Gain (Takens)',
                                     save_dir / 'single_population_gain_sweep.png')
    
    return df_all


def analyze_xor_model(
    tau1_ms: float = 10.0,
    tau2_ms: float = 50.0,
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    PhiID analysis of XOR model using Takens embedding.
    """
    print("\n" + "="*70)
    print("XOR TIMESCALES MODEL: PhiID Exploration")
    print("="*70)
    print(f"τ₁ = {tau1_ms} ms, τ₂ = {tau2_ms} ms")
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 15, 20, 30, 40, 50, 60, 75, 100]
    
    # Simulate
    x, params = simulate_xor_timescales(
        n_samples=n_samples, fs=fs,
        tau1_ms=tau1_ms, tau2_ms=tau2_ms,
        mix_prob=0.7, noise_prob=0.1,
        seed=42
    )
    
    # Use Takens embedding
    print("\nXOR Takens PhiID:")
    df = analyze_takens_lag_sweep(x, lag_range_ms, fs, signal_name='xor')
    df['tau1_ms'] = tau1_ms
    df['tau2_ms'] = tau2_ms
    
    if save_dir:
        df.to_csv(save_dir / 'xor_phiid_full.csv', index=False)
        plot_all_atoms_vs_lag(df, f'XOR Model (τ₁={tau1_ms}ms, τ₂={tau2_ms}ms)',
                             save_dir / 'xor_atoms.png')
        plot_summary_measures_vs_lag(df, f'XOR Model (τ₁={tau1_ms}ms, τ₂={tau2_ms}ms)',
                                     save_dir / 'xor_summary.png')
    
    return df


def analyze_timescale_ratio_sweep(
    tau_fast_ms: float = 5.0,
    ratios: List[float] = [2, 5, 10, 20],
    lag_range_ms: List[float] = None,
    n_samples: int = 100000,
    fs: float = 1000.0,
    save_dir: Path = None
) -> pd.DataFrame:
    """
    PhiID analysis across different timescale ratios.
    """
    print("\n" + "="*70)
    print("TIMESCALE RATIO SWEEP: PhiID Dynamics")
    print("="*70)
    
    if lag_range_ms is None:
        lag_range_ms = [5, 10, 20, 30, 50, 75, 100, 150]
    
    all_results = []
    
    for ratio in ratios:
        tau_slow_ms = tau_fast_ms * ratio
        print(f"\nRatio = {ratio}x (τ_slow = {tau_slow_ms} ms)")
        
        x_fast, x_slow, params = simulate_hierarchical_timescales(
            n_samples=n_samples, fs=fs,
            tau_fast_ms=tau_fast_ms, tau_slow_ms=tau_slow_ms,
            seed=42
        )
        
        df = analyze_lag_sweep(x_fast, x_slow, lag_range_ms, fs,
                               model_name=f'ratio_{ratio}')
        df['ratio'] = ratio
        df['tau_slow_ms'] = tau_slow_ms
        all_results.append(df)
    
    df_all = pd.concat(all_results, ignore_index=True)
    
    if save_dir:
        df_all.to_csv(save_dir / 'timescale_ratio_sweep_full.csv', index=False)
        plot_parameter_sweep_summary(df_all, 'ratio',
                                     'Timescale Ratio Sweep: PhiID Dynamics',
                                     save_dir / 'timescale_ratio_sweep_summary.png')
    
    return df_all


# =============================================================================
# COMPREHENSIVE COMPARISON PLOT
# =============================================================================

def plot_comprehensive_comparison(
    results: Dict[str, pd.DataFrame],
    save_dir: Path
):
    """Create comprehensive comparison across all models."""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    measures = ['summary_Storage', 'summary_Transfer', 'summary_Integrated_info',
                'summary_Downward_causation', 'summary_Upward_causation', 'summary_Phi']
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for idx, measure in enumerate(measures):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        for (name, df), color in zip(results.items(), colors):
            if measure in df.columns:
                df_plot = df.groupby('tau_ms')[measure].mean().reset_index()
                ax.plot(df_plot['tau_ms'], df_plot[measure], 'o-',
                       color=color, linewidth=2, markersize=4, label=name)
        
        ax.set_title(measure.replace('summary_', ''), fontsize=11, fontweight='bold')
        ax.set_xlabel('τ (ms)')
        ax.set_ylabel('bits')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    
    # Add atom comparison subplot
    ax_atoms = fig.add_subplot(gs[2:, :])
    
    # Collect mean atoms at τ=20ms for each model
    atom_data = []
    model_names = []
    
    for name, df in results.items():
        df_20 = df[df['tau_ms'] == 20]
        if len(df_20) > 0:
            atoms = [df_20[a].mean() for a in ATOM_NAMES]
            atom_data.append(atoms)
            model_names.append(name)
    
    if atom_data:
        atom_data = np.array(atom_data)
        
        im = ax_atoms.imshow(atom_data, cmap='RdBu_r', aspect='auto')
        ax_atoms.set_xticks(range(len(ATOM_NAMES)))
        ax_atoms.set_xticklabels(ATOM_NAMES, rotation=45, ha='right')
        ax_atoms.set_yticks(range(len(model_names)))
        ax_atoms.set_yticklabels(model_names)
        ax_atoms.set_title('All PhiID Atoms at τ=20ms', fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax_atoms, label='bits')
    
    plt.suptitle('Comprehensive PhiID Comparison Across Neural Mass Models', 
                 fontsize=16, fontweight='bold')
    plt.savefig(save_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_dir / 'comprehensive_comparison.png'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive PhiID Exploration')
    parser.add_argument('--bins', type=int, default=8, help='Discretization bins (for discrete mode)')
    args = parser.parse_args()
    
    print("="*70)
    print("COMPREHENSIVE TEMPORAL PhiID EXPLORATION")
    print("Neural Mass Models with Known Dynamics")
    print("="*70)
    
    save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define lag range - comprehensive sweep
    lag_range = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200]
    n_samples = 100000  # More samples for stable estimates
    
    all_results = {}
    
    # 1. Hierarchical Timescales
    df_hier = analyze_hierarchical_model(
        tau_fast_ms=5.0, tau_slow_ms=50.0,
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['Hierarchical'] = df_hier[df_hier['analysis_type'] == 'cross']
    
    # 2. E-I Model
    df_ei = analyze_ei_model(
        wEE=2.0, wEI=1.5,
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['E-I'] = df_ei
    
    # 3. E-I Coupling Sweep
    df_ei_coupling = analyze_ei_coupling_sweep(
        coupling_strengths=[0.1, 0.3, 0.5, 0.7, 0.9],
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['E-I_coupling_0.5'] = df_ei_coupling[df_ei_coupling['coupling'] == 0.5]
    
    # 4. Single Population Takens with Gain Sweep
    df_single = analyze_single_population_takens(
        delay_ms=20.0,
        gains=[0.5, 1.0, 2.0, 4.0],
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['Single_gain2'] = df_single[df_single['gain'] == 2.0]
    
    # 5. XOR Model
    df_xor = analyze_xor_model(
        tau1_ms=10.0, tau2_ms=50.0,
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['XOR'] = df_xor
    
    # 6. Timescale Ratio Sweep
    df_ratio = analyze_timescale_ratio_sweep(
        tau_fast_ms=5.0,
        ratios=[2, 5, 10, 20],
        lag_range_ms=lag_range,
        n_samples=n_samples,
        save_dir=save_dir
    )
    all_results['Ratio_10x'] = df_ratio[df_ratio['ratio'] == 10]
    
    # Comprehensive comparison
    plot_comprehensive_comparison(all_results, save_dir)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {save_dir}")
    print("\nFiles generated:")
    for f in sorted(save_dir.glob('*.csv')):
        print(f"  - {f.name}")
    print("\nPlots generated:")
    for f in sorted(save_dir.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
