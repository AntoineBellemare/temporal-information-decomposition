"""
Temporal PID Analysis Script
=============================

This script investigates the temporal structure of information using 
Partial Information Decomposition (PID) on time-delay embeddings.

Following Pedro Mediano's recommended progression:
1. Discrete logic gates with known ground truth (COPY vs XOR)
2. Simple AR/VAR processes
3. Lag sweeps to map temporal structure

Results are saved to the 'results/' directory.
Figures are saved to the 'figures/' directory.

Usage:
    python temporal_pid_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import json
from datetime import datetime

import dit
from dit.pid import PID_MMI, PID_BROJA, PID_CCS
from dit import Distribution
from dit.pid.distributions import bivariates

# Set display options
dit.ditParams['print.exact'] = dit.ditParams['repr.print'] = True

# Create output directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "toy_systems"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_random_binary(n_samples=1000, seed=42):
    """Generate IID random binary sequence."""
    np.random.seed(seed)
    return np.random.randint(0, 2, n_samples)


def generate_copy_process(n_samples=1000, seed=42):
    """
    Generate a process with high redundancy across lags.
    Uses 95% copy / 5% random flip to avoid degenerate constant sequences.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(2)
    for t in range(1, n_samples):
        if np.random.rand() < 0.05:
            x[t] = np.random.randint(2)
        else:
            x[t] = x[t-1]
    return x


def generate_xor_process(n_samples=1000, seed=42):
    """Generate XOR process: x_t = x_{t-1} XOR x_{t-2}"""
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(2)
    x[1] = np.random.randint(2)
    for t in range(2, n_samples):
        x[t] = x[t-1] ^ x[t-2]
    return x


def generate_noisy_copy_process(n_samples=1000, noise_prob=0.1, seed=42):
    """Generate noisy copy: x_t = x_{t-1} with probability (1-noise_prob)"""
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(2)
    for t in range(1, n_samples):
        if np.random.rand() < noise_prob:
            x[t] = 1 - x[t-1]
        else:
            x[t] = x[t-1]
    return x


def generate_ar2_process(n_samples=1000, a1=0.3, a2=0.5, noise_std=0.5, seed=42):
    """Generate AR(2) process: x_t = a1*x_{t-1} + a2*x_{t-2} + noise"""
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.randn()
    x[1] = np.random.randn()
    for t in range(2, n_samples):
        x[t] = a1 * x[t-1] + a2 * x[t-2] + noise_std * np.random.randn()
    return x


def generate_mixed_timescale(n_samples=10000, seed=42):
    """
    Generate a process with multiple timescales:
    x_t = 0.4*x_{t-1} + 0.3*x_{t-3} + noise
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    for i in range(4):
        x[i] = np.random.randn()
    for t in range(4, n_samples):
        x[t] = 0.4 * x[t-1] + 0.3 * x[t-3] + 0.4 * np.random.randn()
    return x


def generate_hierarchical_timescales(n_samples=20000, seed=42):
    """
    Generate a process with SEPARATED redundancy and synergy:
    
    - MEMORY at short timescales: x_t depends on x_{t-1} (creates redundancy at lags 1,2,3)
    - XOR at longer timescales: x_t also depends on x_{t-5} XOR x_{t-6} (creates synergy at 5,6)
    
    This should produce:
    - Redundancy peak at (1,2) or (1,3)
    - Synergy peak at (5,6)
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    
    # Initialize
    for i in range(7):
        x[i] = np.random.randint(2)
    
    for t in range(7, n_samples):
        # 60% of the time: copy from lag-1 (creates redundancy in short lags)
        # 30% of the time: XOR of lag-5 and lag-6 (creates synergy in long lags)  
        # 10% of the time: random (noise)
        r = np.random.rand()
        if r < 0.6:
            x[t] = x[t-1]  # COPY → redundancy at nearby lags
        elif r < 0.9:
            x[t] = x[t-5] ^ x[t-6]  # XOR → synergy at lags 5,6
        else:
            x[t] = np.random.randint(2)  # noise
    
    return x


def discretize_timeseries(x, n_bins=4, method='quantile'):
    """Discretize continuous time series into symbols."""
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
        bins[0] -= 1e-10
        bins[-1] += 1e-10
    else:
        bins = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    return np.digitize(x, bins[1:-1])


def build_embedding_distribution(x, lags=[1, 2]):
    """
    Build a dit Distribution from time series with specified lags.
    Sources: x[t - lag] for each lag
    Target: x[t]
    """
    max_lag = max(lags)
    n = len(x)
    
    outcomes = []
    for t in range(max_lag, n):
        outcome = tuple(x[t - lag] for lag in lags) + (x[t],)
        outcomes.append(''.join(map(str, outcome)))
    
    counts = Counter(outcomes)
    total = sum(counts.values())
    
    outcomes_list = list(counts.keys())
    probs = [counts[o] / total for o in outcomes_list]
    
    return Distribution(outcomes_list, probs)


def compute_pid_summary(dist, pid_class=PID_MMI):
    """Extract key PID values from a distribution."""
    pid = pid_class(dist)
    
    summary = {'redundancy': 0.0, 'unique_0': 0.0, 'unique_1': 0.0, 'synergy': 0.0}
    
    # Use get_pi() method which works with both local and pip-installed dit
    for node in pid._lattice:
        try:
            val = float(pid.get_pi(node))
        except:
            val = 0.0
        
        # Identify node type by structure
        if len(node) == 2 and all(len(n) == 1 for n in node):
            summary['redundancy'] = val
        elif len(node) == 1 and len(node[0]) == 2:
            summary['synergy'] = val
        elif node == ((0,),):
            summary['unique_0'] = val
        elif node == ((1,),):
            summary['unique_1'] = val
    
    return summary


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_ideal_distributions():
    """Analyze DIT's built-in ideal distributions as ground truth."""
    print("\n" + "="*70)
    print("IDEAL DISTRIBUTIONS (Ground Truth)")
    print("="*70)
    
    results = []
    
    # Ideal redundancy
    ideal_rdn = bivariates['redundant']
    pid_rdn = compute_pid_summary(ideal_rdn)
    pid_rdn['distribution'] = 'ideal_redundancy'
    pid_rdn['description'] = 'X1 = X2 = Y'
    results.append(pid_rdn)
    print(f"\nIdeal Redundancy: {pid_rdn}")
    
    # Ideal synergy (XOR)
    ideal_syn = bivariates['synergy']
    pid_syn = compute_pid_summary(ideal_syn)
    pid_syn['distribution'] = 'ideal_synergy'
    pid_syn['description'] = 'Y = X1 XOR X2'
    results.append(pid_syn)
    print(f"Ideal Synergy (XOR): {pid_syn}")
    
    # Unique information
    ideal_unq = bivariates['unique 1']
    pid_unq = compute_pid_summary(ideal_unq)
    pid_unq['distribution'] = 'ideal_unique'
    pid_unq['description'] = 'Y = X1 (X2 independent)'
    results.append(pid_unq)
    print(f"Ideal Unique: {pid_unq}")
    
    return pd.DataFrame(results)


def analyze_logic_gates(n_samples=10000):
    """Analyze logic gate processes."""
    print("\n" + "="*70)
    print("LOGIC GATE PROCESSES")
    print("="*70)
    
    results = []
    
    # COPY process (with noise)
    x_copy = generate_copy_process(n_samples=n_samples)
    dist_copy = build_embedding_distribution(x_copy, lags=[1, 2])
    pid_copy = compute_pid_summary(dist_copy)
    pid_copy['process'] = 'copy_noisy'
    pid_copy['description'] = 'x_t = x_{t-1} (95% copy)'
    results.append(pid_copy)
    print(f"\nCOPY (noisy): {pid_copy}")
    
    # XOR process
    x_xor = generate_xor_process(n_samples=n_samples)
    dist_xor = build_embedding_distribution(x_xor, lags=[1, 2])
    pid_xor = compute_pid_summary(dist_xor)
    pid_xor['process'] = 'xor'
    pid_xor['description'] = 'x_t = x_{t-1} XOR x_{t-2}'
    results.append(pid_xor)
    print(f"XOR: {pid_xor}")
    
    # Noisy COPY with varying noise
    for noise in [0.05, 0.1, 0.2, 0.3]:
        x_noisy = generate_noisy_copy_process(n_samples=n_samples, noise_prob=noise)
        dist_noisy = build_embedding_distribution(x_noisy, lags=[1, 2])
        pid_noisy = compute_pid_summary(dist_noisy)
        pid_noisy['process'] = f'copy_noise_{noise}'
        pid_noisy['noise_prob'] = noise
        pid_noisy['description'] = f'x_t = x_{{t-1}} with {noise*100}% noise'
        results.append(pid_noisy)
        print(f"Noisy COPY ({noise*100}%): {pid_noisy}")
    
    return pd.DataFrame(results)


def analyze_ar_processes(n_samples=10000):
    """Analyze AR(2) processes with different coefficient ratios."""
    print("\n" + "="*70)
    print("AR(2) PROCESSES")
    print("="*70)
    
    results = []
    
    # Different (a1, a2) combinations
    coefficients = [
        (0.7, 0.1, "lag1_dominates"),
        (0.5, 0.3, "lag1_slightly"),
        (0.3, 0.5, "lag2_slightly"),
        (0.1, 0.7, "lag2_dominates"),
        (0.4, 0.4, "equal"),
    ]
    
    for a1, a2, label in coefficients:
        x = generate_ar2_process(n_samples=n_samples, a1=a1, a2=a2)
        x_discrete = discretize_timeseries(x, n_bins=4)
        dist = build_embedding_distribution(x_discrete, lags=[1, 2])
        pid = compute_pid_summary(dist)
        pid['process'] = f'ar2_{label}'
        pid['a1'] = a1
        pid['a2'] = a2
        pid['description'] = f'AR(2): a1={a1}, a2={a2}'
        results.append(pid)
        print(f"AR(2) a1={a1}, a2={a2}: Unq0={pid['unique_0']:.4f}, Unq1={pid['unique_1']:.4f}")
    
    return pd.DataFrame(results)


def analyze_lag_sweep(n_samples=10000, max_lag=6):
    """Sweep over all lag pairs for mixed-timescale process."""
    print("\n" + "="*70)
    print("LAG SWEEP: Mixed-timescale AR process")
    print("="*70)
    
    x = generate_mixed_timescale(n_samples=n_samples)
    x_discrete = discretize_timeseries(x, n_bins=4)
    
    results = []
    
    for lag1 in range(1, max_lag):
        for lag2 in range(lag1 + 1, max_lag + 1):
            dist = build_embedding_distribution(x_discrete, lags=[lag1, lag2])
            pid = compute_pid_summary(dist)
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            results.append(pid)
    
    df = pd.DataFrame(results)
    print(df[['lag1', 'lag2', 'redundancy', 'unique_0', 'unique_1', 'synergy']].to_string())
    
    return df


def analyze_hierarchical_timescales(n_samples=20000, max_lag=7):
    """Analyze hierarchical timescales process with separated redundancy and synergy."""
    print("\n" + "="*70)
    print("HIERARCHICAL TIMESCALES: Memory (lags 1-2) + XOR (lags 5-6)")
    print("="*70)
    
    x = generate_hierarchical_timescales(n_samples=n_samples)
    
    results = []
    
    for lag1 in range(1, max_lag):
        for lag2 in range(lag1 + 1, max_lag + 1):
            dist = build_embedding_distribution(x, lags=[lag1, lag2])
            pid = compute_pid_summary(dist)
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            results.append(pid)
    
    df = pd.DataFrame(results)
    
    # Find peaks
    red_peak = df.loc[df['redundancy'].idxmax()]
    syn_peak = df.loc[df['synergy'].idxmax()]
    
    print(f"\nREDUNDANCY PEAK: lags ({int(red_peak['lag1'])}, {int(red_peak['lag2'])}) = {red_peak['redundancy']:.4f}")
    print(f"SYNERGY PEAK:    lags ({int(syn_peak['lag1'])}, {int(syn_peak['lag2'])}) = {syn_peak['synergy']:.4f}")
    
    if (red_peak['lag1'], red_peak['lag2']) != (syn_peak['lag1'], syn_peak['lag2']):
        print("\n✓ SUCCESS: Redundancy and Synergy peak at DIFFERENT lag pairs!")
        print(f"  → Memory/persistence at lags {int(red_peak['lag1'])}-{int(red_peak['lag2'])}")
        print(f"  → Nonlinear interaction at lags {int(syn_peak['lag1'])}-{int(syn_peak['lag2'])}")
    
    # Add peak info to dataframe metadata
    df.attrs['red_peak'] = (int(red_peak['lag1']), int(red_peak['lag2']), red_peak['redundancy'])
    df.attrs['syn_peak'] = (int(syn_peak['lag1']), int(syn_peak['lag2']), syn_peak['synergy'])
    
    print(df[['lag1', 'lag2', 'redundancy', 'unique_0', 'unique_1', 'synergy']].to_string())
    
    return df


def analyze_discretization_sensitivity(n_samples=10000):
    """Test sensitivity to number of bins."""
    print("\n" + "="*70)
    print("DISCRETIZATION SENSITIVITY")
    print("="*70)
    
    x = generate_ar2_process(n_samples=n_samples, a1=0.5, a2=0.3)
    
    results = []
    
    for n_bins in [2, 3, 4, 5, 6, 8, 10]:
        x_discrete = discretize_timeseries(x, n_bins=n_bins)
        dist = build_embedding_distribution(x_discrete, lags=[1, 2])
        pid = compute_pid_summary(dist)
        pid['n_bins'] = n_bins
        results.append(pid)
        print(f"Bins={n_bins}: Red={pid['redundancy']:.4f}, Syn={pid['synergy']:.4f}")
    
    return pd.DataFrame(results)


def compare_pid_measures(n_samples=10000):
    """Compare different PID measures on the same distributions."""
    print("\n" + "="*70)
    print("COMPARING PID MEASURES")
    print("="*70)
    
    results = []
    
    # Test distributions
    test_cases = [
        ('ideal_synergy', bivariates['synergy']),
        ('ideal_redundancy', bivariates['redundant']),
    ]
    
    # Add empirical distributions
    x_xor = generate_xor_process(n_samples=n_samples)
    dist_xor = build_embedding_distribution(x_xor, lags=[1, 2])
    test_cases.append(('empirical_xor', dist_xor))
    
    x_copy = generate_copy_process(n_samples=n_samples)
    dist_copy = build_embedding_distribution(x_copy, lags=[1, 2])
    test_cases.append(('empirical_copy', dist_copy))
    
    measures = [
        ('MMI', PID_MMI),
        ('BROJA', PID_BROJA),
        ('CCS', PID_CCS),
    ]
    
    for dist_name, dist in test_cases:
        for measure_name, measure_class in measures:
            try:
                pid = compute_pid_summary(dist, pid_class=measure_class)
                pid['distribution'] = dist_name
                pid['measure'] = measure_name
                results.append(pid)
            except Exception as e:
                print(f"Warning: {measure_name} failed on {dist_name}: {e}")
    
    df = pd.DataFrame(results)
    print(df.pivot_table(
        index='distribution', 
        columns='measure', 
        values=['redundancy', 'synergy']
    ).round(4).to_string())
    
    return df


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_pid_bars(df, title, filename, figsize=(10, 6)):
    """Create bar plot of PID components."""
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['redundancy'], width, label='Redundancy', color='#2ecc71')
    ax.bar(x - 0.5*width, df['unique_0'], width, label='Unique (lag1)', color='#3498db')
    ax.bar(x + 0.5*width, df['unique_1'], width, label='Unique (lag2)', color='#9b59b6')
    ax.bar(x + 1.5*width, df['synergy'], width, label='Synergy', color='#e74c3c')
    
    ax.set_xlabel('Process')
    ax.set_ylabel('Information (bits)')
    ax.set_title(title)
    ax.set_xticks(x)
    
    if 'process' in df.columns:
        ax.set_xticklabels(df['process'], rotation=45, ha='right')
    elif 'distribution' in df.columns:
        ax.set_xticklabels(df['distribution'], rotation=45, ha='right')
    
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


def plot_lag_sweep_heatmaps(df, filename):
    """Create heatmaps of PID components across lag pairs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    components = ['redundancy', 'unique_0', 'unique_1', 'synergy']
    titles = ['Redundancy', 'Unique (Lag 1)', 'Unique (Lag 2)', 'Synergy']
    cmaps = ['Greens', 'Blues', 'Purples', 'Reds']
    
    max_lag = df['lag2'].max()
    
    for ax, comp, title, cmap in zip(axes.flat, components, titles, cmaps):
        # Create matrix
        matrix = np.zeros((max_lag, max_lag))
        matrix[:] = np.nan
        
        for _, row in df.iterrows():
            l1, l2 = int(row['lag1']), int(row['lag2'])
            matrix[l1-1, l2-1] = row[comp]
            matrix[l2-1, l1-1] = row[comp]  # Symmetric
        
        sns.heatmap(matrix, ax=ax, cmap=cmap, annot=True, fmt='.3f',
                    xticklabels=range(1, max_lag+1),
                    yticklabels=range(1, max_lag+1),
                    cbar_kws={'label': 'bits'})
        ax.set_title(title)
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1')
    
    plt.suptitle('PID Components Across Lag Pairs\n(Mixed-timescale AR: x_t = 0.4*x_{t-1} + 0.3*x_{t-3} + noise)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


def plot_hierarchical_heatmaps(df, filename):
    """Create heatmaps for hierarchical timescales showing separated redundancy/synergy peaks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    max_lag = df['lag2'].max()
    lags = list(range(1, max_lag + 1))
    n_lags = len(lags)
    
    # Initialize matrices with NaN (only upper triangle is filled)
    red_matrix = np.full((n_lags, n_lags), np.nan)
    syn_matrix = np.full((n_lags, n_lags), np.nan)
    uniq0_matrix = np.full((n_lags, n_lags), np.nan)
    uniq1_matrix = np.full((n_lags, n_lags), np.nan)
    
    # Fill matrices from results
    for _, row in df.iterrows():
        i = int(row['lag1']) - 1
        j = int(row['lag2']) - 1
        red_matrix[i, j] = row['redundancy']
        syn_matrix[i, j] = row['synergy']
        uniq0_matrix[i, j] = row['unique_0']
        uniq1_matrix[i, j] = row['unique_1']
    
    # Get peak info from dataframe attrs or compute it
    if hasattr(df, 'attrs') and 'red_peak' in df.attrs:
        red_peak = df.attrs['red_peak']
        syn_peak = df.attrs['syn_peak']
    else:
        red_idx = df['redundancy'].idxmax()
        syn_idx = df['synergy'].idxmax()
        red_peak = (int(df.loc[red_idx, 'lag1']), int(df.loc[red_idx, 'lag2']), df.loc[red_idx, 'redundancy'])
        syn_peak = (int(df.loc[syn_idx, 'lag1']), int(df.loc[syn_idx, 'lag2']), df.loc[syn_idx, 'synergy'])
    
    # Helper function for heatmap with peak marker
    def plot_heatmap(ax, matrix, title, cmap, peak_coords=None):
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, annot=True, fmt='.3f', cmap=cmap, 
                    mask=mask, xticklabels=lags, yticklabels=lags,
                    cbar_kws={'label': 'bits'})
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1')
        ax.set_title(title)
        
        # Mark peak with a star
        if peak_coords is not None:
            ax.plot(peak_coords[1] - 0.5, peak_coords[0] - 0.5, 'w*', markersize=20, 
                    markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot each component
    plot_heatmap(axes[0, 0], red_matrix, 
                 f'REDUNDANCY\nPeak at lags ({red_peak[0]}, {red_peak[1]})', 
                 'Reds', peak_coords=(red_peak[0], red_peak[1]))
    
    plot_heatmap(axes[0, 1], syn_matrix, 
                 f'SYNERGY\nPeak at lags ({syn_peak[0]}, {syn_peak[1]})', 
                 'Blues', peak_coords=(syn_peak[0], syn_peak[1]))
    
    plot_heatmap(axes[1, 0], uniq0_matrix, 'UNIQUE (Lag 1)', 'Greens')
    plot_heatmap(axes[1, 1], uniq1_matrix, 'UNIQUE (Lag 2)', 'Purples')
    
    plt.suptitle('Hierarchical Timescales: Memory (lags 1-2) + XOR (lags 5-6)\n'
                 'Temporal PID across lag pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


def plot_ar_coefficients(df, filename):
    """Plot PID vs AR coefficient ratio."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate coefficient ratio
    df = df.copy()
    df['a1_ratio'] = df['a1'] / (df['a1'] + df['a2'])
    df = df.sort_values('a1_ratio')
    
    ax.plot(df['a1_ratio'], df['redundancy'], 'o-', label='Redundancy', color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df['a1_ratio'], df['unique_0'], 's-', label='Unique (lag1)', color='#3498db', linewidth=2, markersize=8)
    ax.plot(df['a1_ratio'], df['unique_1'], '^-', label='Unique (lag2)', color='#9b59b6', linewidth=2, markersize=8)
    ax.plot(df['a1_ratio'], df['synergy'], 'd-', label='Synergy', color='#e74c3c', linewidth=2, markersize=8)
    
    ax.set_xlabel('a₁ / (a₁ + a₂)  [← lag2 dominates | lag1 dominates →]')
    ax.set_ylabel('Information (bits)')
    ax.set_title('PID Components vs AR(2) Coefficient Ratio')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


def plot_discretization_sensitivity(df, filename):
    """Plot PID sensitivity to number of bins."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df['n_bins'], df['redundancy'], 'o-', label='Redundancy', color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df['n_bins'], df['unique_0'], 's-', label='Unique (lag1)', color='#3498db', linewidth=2, markersize=8)
    ax.plot(df['n_bins'], df['unique_1'], '^-', label='Unique (lag2)', color='#9b59b6', linewidth=2, markersize=8)
    ax.plot(df['n_bins'], df['synergy'], 'd-', label='Synergy', color='#e74c3c', linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Bins')
    ax.set_ylabel('Information (bits)')
    ax.set_title('PID Sensitivity to Discretization\n(AR(2) with a1=0.5, a2=0.3)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


def plot_measure_comparison(df, filename):
    """Compare PID measures on different distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Synergy comparison
    pivot_syn = df.pivot(index='distribution', columns='measure', values='synergy')
    pivot_syn.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#c0392b', '#922b21'])
    axes[0].set_title('Synergy by Measure')
    axes[0].set_ylabel('Synergy (bits)')
    axes[0].set_xlabel('')
    axes[0].legend(title='Measure')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Redundancy comparison
    pivot_red = df.pivot(index='distribution', columns='measure', values='redundancy')
    pivot_red.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#27ae60', '#1e8449'])
    axes[1].set_title('Redundancy by Measure')
    axes[1].set_ylabel('Redundancy (bits)')
    axes[1].set_xlabel('')
    axes[1].legend(title='Measure')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comparison of PID Measures', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all analyses and save results."""
    print("="*70)
    print("TEMPORAL PID ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    all_results = {}
    
    # 1. Ideal distributions (ground truth)
    df_ideal = analyze_ideal_distributions()
    df_ideal.to_csv(RESULTS_DIR / "ideal_distributions.csv", index=False)
    all_results['ideal'] = df_ideal.to_dict('records')
    plot_pid_bars(df_ideal, 'Ideal Distributions (Ground Truth)', 'ideal_distributions.png')
    
    # 2. Logic gate processes
    df_gates = analyze_logic_gates()
    df_gates.to_csv(RESULTS_DIR / "logic_gates.csv", index=False)
    all_results['logic_gates'] = df_gates.to_dict('records')
    plot_pid_bars(df_gates, 'Logic Gate Processes', 'logic_gates.png')
    
    # 3. AR(2) processes
    df_ar = analyze_ar_processes()
    df_ar.to_csv(RESULTS_DIR / "ar_processes.csv", index=False)
    all_results['ar_processes'] = df_ar.to_dict('records')
    plot_ar_coefficients(df_ar, 'ar_coefficients.png')
    
    # 4. Lag sweep
    df_lags = analyze_lag_sweep()
    df_lags.to_csv(RESULTS_DIR / "lag_sweep.csv", index=False)
    all_results['lag_sweep'] = df_lags.to_dict('records')
    plot_lag_sweep_heatmaps(df_lags, 'lag_sweep_heatmaps.png')
    
    # 5. Discretization sensitivity
    df_bins = analyze_discretization_sensitivity()
    df_bins.to_csv(RESULTS_DIR / "discretization_sensitivity.csv", index=False)
    all_results['discretization'] = df_bins.to_dict('records')
    plot_discretization_sensitivity(df_bins, 'discretization_sensitivity.png')
    
    # 6. Hierarchical timescales (separated redundancy/synergy peaks)
    df_hier = analyze_hierarchical_timescales()
    df_hier.to_csv(RESULTS_DIR / "hierarchical_timescales.csv", index=False)
    all_results['hierarchical'] = df_hier.to_dict('records')
    plot_hierarchical_heatmaps(df_hier, 'hierarchical_timescales.png')
    
    # 7. Compare PID measures
    df_measures = compare_pid_measures()
    df_measures.to_csv(RESULTS_DIR / "measure_comparison.csv", index=False)
    all_results['measures'] = df_measures.to_dict('records')
    plot_measure_comparison(df_measures, 'measure_comparison.png')
    
    # Save all results as JSON
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {RESULTS_DIR}")
    print(f"\nFiles created:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
