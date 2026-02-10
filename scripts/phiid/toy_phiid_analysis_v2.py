"""
Toy PhiID Analysis - Version 2 (Improved)
==========================================

Fixes issues from v1:
1. AND/OR processes collapse to fixed points - replaced with proper versions
2. XOR has period-3 cycles causing zeros at certain lags - use longer periods
3. Temporal overlap when extra_lag == tau causes spurious correlations
4. Need to ensure extra_lag > tau to avoid overlap

PhiID Setup for Temporal Analysis:
- src = signal[:-extra_lag]  (X process: time t to t+N-extra_lag)
- tgt = signal[extra_lag:]   (Y process: time t+extra_lag to t+N)
- tau = embedding delay

Creates 4 vectors (after embedding with tau):
- X_past   = signal[t-tau]        (from src)
- X_future = signal[t]            (from src)  
- Y_past   = signal[t+extra_lag-tau]  (from tgt)
- Y_future = signal[t+extra_lag]  (from tgt)

CRITICAL: To avoid overlap, we need extra_lag >= tau
Otherwise X_future and Y_past can be the same timepoint!

Usage:
    python toy_phiid_analysis_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "toy_systems_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATOM_NAMES = list(PhiID_atoms_abbr)

INFORMATION_DYNAMICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward_causation": ["sty", "stx", "str"],
    "Upward_causation": ["xts", "yts", "rts"],
}


# =============================================================================
# IMPROVED TOY PROCESS GENERATORS
# =============================================================================

def generate_iid_binary(n_samples=10000, seed=42):
    """IID binary - no temporal structure."""
    np.random.seed(seed)
    return np.random.randint(0, 2, n_samples).astype(float)


def generate_iid_gaussian(n_samples=10000, seed=42):
    """IID Gaussian - no temporal structure."""
    np.random.seed(seed)
    return np.random.randn(n_samples)


def generate_copy_process(n_samples=10000, seed=42):
    """
    COPY: x[t] = x[t-1]
    Start with random binary, then just copy.
    Creates blocks of constant values.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.randint(0, 2)
    for t in range(1, n_samples):
        x[t] = x[t-1]
    return x


def generate_noisy_copy(n_samples=10000, noise_level=0.1, seed=42):
    """
    Noisy COPY (random walk): x[t] = x[t-1] + noise
    """
    np.random.seed(seed)
    x = np.cumsum(np.random.randn(n_samples) * noise_level)
    return x


def generate_xor_binary(n_samples=10000, seed=42):
    """
    Binary XOR: x[t] = x[t-1] XOR x[t-2]
    
    Note: This creates a period-6 sequence (Fibonacci-like mod 2)
    The pattern is: 0,1,1,0,1,1,0,1,1,...
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        x[t] = x[t-1] ^ x[t-2]
    return x.astype(float)


def generate_xor_noisy(n_samples=10000, noise_prob=0.1, seed=42):
    """
    Noisy XOR: x[t] = x[t-1] XOR x[t-2] with probability 1-noise_prob
              = random bit with probability noise_prob
    
    Adding noise prevents perfect periodicity and makes it more realistic.
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


def generate_and_refreshed(n_samples=10000, refresh_prob=0.05, seed=42):
    """
    AND with random refreshes to prevent collapse.
    
    Occasionally injects random bits to restart the dynamics.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        if np.random.rand() < refresh_prob:
            x[t] = np.random.randint(0, 2)
        else:
            x[t] = x[t-1] & x[t-2]
    return x.astype(float)


def generate_or_refreshed(n_samples=10000, refresh_prob=0.05, seed=42):
    """
    OR with random refreshes to prevent collapse.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        if np.random.rand() < refresh_prob:
            x[t] = np.random.randint(0, 2)
        else:
            x[t] = x[t-1] | x[t-2]
    return x.astype(float)


def generate_ar1(n_samples=10000, phi=0.9, seed=42):
    """AR(1): x[t] = phi * x[t-1] + noise"""
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * np.sqrt(1 - phi**2)
    x[0] = np.random.randn()
    for t in range(1, n_samples):
        x[t] = phi * x[t-1] + noise[t]
    return x


def generate_ar2(n_samples=10000, phi1=0.5, phi2=0.3, seed=42):
    """AR(2): x[t] = phi1*x[t-1] + phi2*x[t-2] + noise"""
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise_var = max(0.1, 1 - phi1**2 - phi2**2)
    noise = np.random.randn(n_samples) * np.sqrt(noise_var)
    x[0] = np.random.randn()
    x[1] = np.random.randn()
    for t in range(2, n_samples):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + noise[t]
    return x


def generate_oscillation(n_samples=10000, period=20, seed=42):
    """Noisy sinusoidal oscillation."""
    np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.sin(2 * np.pi * t / period) + 0.2 * np.random.randn(n_samples)
    return x


def generate_markov_chain(n_samples=10000, p_stay=0.9, seed=42):
    """
    Binary Markov chain: P(x[t]=x[t-1]) = p_stay
    
    This is a proper binary process with memory but no fixed point.
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


# =============================================================================
# PhiID ANALYSIS
# =============================================================================

def compute_temporal_phiid(signal, extra_lag, tau=1, kind='gaussian', 
                            redundancy='MMI', min_extra_lag_multiple=2):
    """
    Compute PhiID for temporal analysis.
    
    IMPORTANT: To avoid X_future/Y_past overlap, we require:
    extra_lag >= tau * min_extra_lag_multiple
    
    If extra_lag is too small relative to tau, results are unreliable.
    """
    n = len(signal)
    
    # Ensure enough separation
    effective_extra_lag = max(extra_lag, tau * min_extra_lag_multiple)
    if effective_extra_lag != extra_lag:
        # Silently adjust to prevent overlap issues
        extra_lag = effective_extra_lag
    
    if n <= extra_lag + tau + 100:
        return None
    
    # Create pseudo-bivariate system
    src = signal[:-extra_lag].copy()
    tgt = signal[extra_lag:].copy()
    
    try:
        atoms_res, calc_res = calc_PhiID(src, tgt, tau, kind=kind, redundancy=redundancy)
    except Exception as e:
        print(f"    PhiID error: {e}")
        return None
    
    # Average over time
    atoms = {}
    for name in ATOM_NAMES:
        if name in atoms_res:
            val = np.nanmean(atoms_res[name])
            atoms[name] = float(val) if np.isfinite(val) else 0.0
        else:
            atoms[name] = 0.0
    
    return atoms


def analyze_process(signal, name, extra_lags, tau=1, kind='gaussian'):
    """Analyze a process across multiple extra_lag values."""
    results = []
    
    for extra_lag in extra_lags:
        print(f"    Processing extra_lag = {extra_lag}...")
        atoms = compute_temporal_phiid(signal, extra_lag, tau, kind)
        if atoms is None:
            continue
        atoms['extra_lag'] = extra_lag
        atoms['process'] = name
        atoms['tau'] = tau
        results.append(atoms)
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_atoms_comparison(all_results, save_path=None):
    """4x4 grid of all 16 atoms."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    processes = all_results['process'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(processes)))
    
    for idx, atom in enumerate(ATOM_NAMES):
        ax = axes[idx]
        for i, proc in enumerate(processes):
            df_proc = all_results[all_results['process'] == proc]
            ax.plot(df_proc['extra_lag'], df_proc[atom], 'o-', 
                   color=colors[i], label=proc, linewidth=2, markersize=4)
        
        ax.set_xlabel('Extra Lag')
        ax.set_ylabel('bits')
        ax.set_title(f'{atom}', fontweight='bold')
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    plt.suptitle('PhiID Atoms Across Toy Processes (v2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_dynamics_comparison(all_results, save_path=None):
    """Information dynamics summary."""
    dynamics_data = []
    
    for _, row in all_results.iterrows():
        entry = {'process': row['process'], 'extra_lag': row['extra_lag']}
        for metric, atoms in INFORMATION_DYNAMICS.items():
            entry[metric] = sum(row.get(a, 0) for a in atoms)
        dynamics_data.append(entry)
    
    df_dynamics = pd.DataFrame(dynamics_data)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    processes = df_dynamics['process'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(processes)))
    
    for idx, metric in enumerate(INFORMATION_DYNAMICS.keys()):
        ax = axes[idx]
        for i, proc in enumerate(processes):
            df_proc = df_dynamics[df_dynamics['process'] == proc]
            ax.plot(df_proc['extra_lag'], df_proc[metric], 'o-',
                   color=colors[i], label=proc, linewidth=2, markersize=5)
        
        ax.set_xlabel('Extra Lag')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Information Dynamics Across Toy Processes (v2)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_dynamics


def plot_interpretation_guide():
    """Create interpretation guide figure."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    text = """
    PhiID INTERPRETATION FOR TEMPORAL ANALYSIS (v2)
    ================================================
    
    SETUP (FIXED):
    - src = signal[:-extra_lag]   -> X process (earlier portion)
    - tgt = signal[extra_lag:]    -> Y process (later portion)
    - tau = embedding delay (past/future within each process)
    - CRITICAL: extra_lag >= 2*tau to avoid timepoint overlap!
    
    Creates 4 vectors:
    - X_past:   signal at time t-tau
    - X_future: signal at time t
    - Y_past:   signal at time t+extra_lag-tau
    - Y_future: signal at time t+extra_lag
    
    KEY ATOMS (16 total, organized in 4x4):
    
             | Redundancy  | X-unique   | Y-unique   | Synergy    |
             |     (r)     |    (x)     |    (y)     |    (s)     |
    ---------+-------------+------------+------------+------------+
    Red (r)  |     rtr     |    xtr     |    ytr     |    str     |
    X-uniq   |     rtx     |    xtx     |    ytx     |    stx     |
    Y-uniq   |     rty     |    xty     |    yty     |    sty     |
    Syn (s)  |     rts     |    xts     |    yts     |    sts     |
    
    EXPECTED PATTERNS:
    
    - IID NOISE:        All atoms near 0 (no temporal structure)
    
    - COPY (x[t]=x[t-1]): High xtx, yty (self-storage)
                          High rtr (redundant transfer)
                          Low synergy (simple linear)
    
    - XOR (x[t]=x[t-1]^x[t-2]): High str, sts (synergistic)
                                 Pattern depends on lag alignment
    
    - AR(1) (x[t]=phi*x[t-1]+e): Similar to COPY but scaled by phi
                                  Higher phi -> more rtr, xtx, yty
    
    - MARKOV CHAIN: Memory creates rtr, xtx, yty
                    Amount depends on transition probability
    
    - OSCILLATION: Periodic patterns in atoms
                   Phase relationships across lags
    
    INFORMATION DYNAMICS GROUPINGS:
    - Storage:   rtr + xtx + yty + sts  (info preserved over time)
    - Transfer:  xty + ytx              (info flow between processes)
    - Copy:      xtx + yty              (within-process continuity)
    - Erasure:   rtx + rty              (lost information)
    - Downward:  sty + stx + str        (synergy -> parts)
    - Upward:    xts + yts + rts        (parts -> synergy)
    """
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TOY PhiID ANALYSIS v2 (Improved)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Parameters
    N_SAMPLES = 10000
    TAU = 1
    # Start extra_lag at 2*tau to avoid overlap
    EXTRA_LAGS = [2, 3, 5, 8, 10, 15, 20, 30, 50]
    
    print(f"\nParameters:")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  tau: {TAU}")
    print(f"  Extra lags: {EXTRA_LAGS}")
    print(f"  Min extra_lag = 2*tau = {2*TAU} (to avoid overlap)")
    
    # Generate processes
    print("\nGenerating toy processes...")
    
    processes = {
        # Binary/discrete
        'IID_Binary': generate_iid_binary(N_SAMPLES),
        'Markov_0.9': generate_markov_chain(N_SAMPLES, p_stay=0.9),
        'XOR_Noisy': generate_xor_noisy(N_SAMPLES, noise_prob=0.1),
        'AND_Refresh': generate_and_refreshed(N_SAMPLES, refresh_prob=0.1),
        
        # Gaussian/continuous
        'IID_Gauss': generate_iid_gaussian(N_SAMPLES),
        'AR1_0.5': generate_ar1(N_SAMPLES, phi=0.5),
        'AR1_0.9': generate_ar1(N_SAMPLES, phi=0.9),
        'AR2': generate_ar2(N_SAMPLES),
        'NoisyCopy': generate_noisy_copy(N_SAMPLES),
        'Oscillation': generate_oscillation(N_SAMPLES, period=20),
    }
    
    # Determine signal type
    binary_processes = ['IID_Binary', 'Markov_0.9', 'XOR_Noisy', 'AND_Refresh']
    
    all_results = []
    
    for name, signal in processes.items():
        print(f"\n{'='*50}")
        print(f"Analyzing: {name}")
        print("=" * 50)
        
        # Choose kind based on signal type
        kind = 'discrete' if name in binary_processes else 'gaussian'
        print(f"  Using kind='{kind}'")
        
        df = analyze_process(signal, name, EXTRA_LAGS, tau=TAU, kind=kind)
        if len(df) > 0:
            all_results.append(df)
            print(f"  Completed {len(df)} lag points")
    
    # Combine
    all_results = pd.concat(all_results, ignore_index=True)
    all_results.to_csv(RESULTS_DIR / "all_toy_results_v2.csv", index=False)
    
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    
    # Plots
    print("\n1. Atoms comparison...")
    plot_atoms_comparison(all_results, RESULTS_DIR / "atoms_all_v2.png")
    
    # Binary only
    binary_results = all_results[all_results['process'].isin(binary_processes)]
    if len(binary_results) > 0:
        print("2. Binary processes atoms...")
        plot_atoms_comparison(binary_results, RESULTS_DIR / "atoms_binary_v2.png")
    
    # Gaussian only
    gauss_processes = [p for p in processes.keys() if p not in binary_processes]
    gauss_results = all_results[all_results['process'].isin(gauss_processes)]
    if len(gauss_results) > 0:
        print("3. Gaussian processes atoms...")
        plot_atoms_comparison(gauss_results, RESULTS_DIR / "atoms_gaussian_v2.png")
    
    print("4. Information dynamics...")
    plot_dynamics_comparison(all_results, RESULTS_DIR / "dynamics_v2.png")
    
    print("5. Interpretation guide...")
    fig = plot_interpretation_guide()
    fig.savefig(RESULTS_DIR / "interpretation_guide_v2.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY: Mean atoms at extra_lag = 5")
    print("=" * 70)
    
    lag5 = all_results[all_results['extra_lag'] == 5]
    if len(lag5) > 0:
        summary = lag5.groupby('process')[ATOM_NAMES].mean()
        print(summary.round(3).to_string())
    
    print(f"\n\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
