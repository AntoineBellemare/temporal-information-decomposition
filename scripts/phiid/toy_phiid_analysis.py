"""
Toy PhiID Analysis
==================

Validate PhiID on toy processes with known temporal structure.

PhiID Setup for Temporal Analysis:
- src = signal[:-extra_lag]  (earlier portion, "X process")
- tgt = signal[extra_lag:]   (later portion, "Y process") 
- tau = embedding delay (creates past/future within each process)

This creates 4 vectors:
- X_past = src[:-tau]     = signal at time t-tau
- X_future = src[tau:]    = signal at time t
- Y_past = tgt[:-tau]     = signal at time t+extra_lag-tau
- Y_future = tgt[tau:]    = signal at time t+extra_lag

The 16 PhiID atoms decompose information flow between the two processes.

Toy Processes:
1. COPY: x[t] = x[t-1] - perfect memory
2. XOR: x[t] = x[t-1] XOR x[t-2] - synergistic temporal structure
3. AR(1): x[t] = a*x[t-1] + noise - linear predictability
4. IID noise: no temporal structure (baseline)

Expected Results:
- COPY: High storage (xtx, yty), high redundancy (rtr)
- XOR: Synergistic patterns, information requires both past points
- AR(1): Redundancy dominates, storage proportional to coefficient
- IID: All atoms near zero

Usage:
    python toy_phiid_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "toy_systems"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PhiID atom names
ATOM_NAMES = list(PhiID_atoms_abbr)

# Grouped measures
INFORMATION_DYNAMICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward_causation": ["sty", "stx", "str"],
    "Upward_causation": ["xts", "yts", "rts"],
}


# =============================================================================
# TOY PROCESS GENERATORS
# =============================================================================

def generate_iid_binary(n_samples=5000, seed=42):
    """Generate IID random binary sequence (no temporal structure)."""
    np.random.seed(seed)
    return np.random.randint(0, 2, n_samples).astype(float)


def generate_iid_gaussian(n_samples=5000, seed=42):
    """Generate IID Gaussian noise (no temporal structure)."""
    np.random.seed(seed)
    return np.random.randn(n_samples)


def generate_copy_process(n_samples=5000, seed=42):
    """
    Generate COPY process: x[t] = x[t-1]
    
    Perfect memory - the signal is just the initial value repeated.
    For PhiID: Should show maximum storage and redundancy.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.randn()
    for t in range(1, n_samples):
        x[t] = x[t-1]
    return x


def generate_noisy_copy(n_samples=5000, noise_level=0.1, seed=42):
    """
    Generate noisy COPY: x[t] = x[t-1] + noise
    
    This is actually a random walk, showing accumulating memory.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    x[0] = np.random.randn()
    noise = np.random.randn(n_samples) * noise_level
    for t in range(1, n_samples):
        x[t] = x[t-1] + noise[t]
    return x


def generate_xor_process(n_samples=5000, seed=42):
    """
    Generate XOR process: x[t] = x[t-1] XOR x[t-2]
    
    Synergistic: knowing just x[t-1] OR x[t-2] gives no info,
    but knowing BOTH perfectly predicts x[t].
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        x[t] = x[t-1] ^ x[t-2]
    return x.astype(float)


def generate_and_process(n_samples=5000, seed=42):
    """
    Generate AND process: x[t] = x[t-1] AND x[t-2]
    
    Redundant: if either input is 0, output is 0.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        x[t] = x[t-1] & x[t-2]
    return x.astype(float)


def generate_or_process(n_samples=5000, seed=42):
    """
    Generate OR process: x[t] = x[t-1] OR x[t-2]
    
    Redundant: if either input is 1, output is 1.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples, dtype=int)
    x[0] = np.random.randint(0, 2)
    x[1] = np.random.randint(0, 2)
    for t in range(2, n_samples):
        x[t] = x[t-1] | x[t-2]
    return x.astype(float)


def generate_ar1(n_samples=5000, phi=0.9, seed=42):
    """
    Generate AR(1) process: x[t] = phi * x[t-1] + noise
    
    Linear predictability. Higher phi = more memory/redundancy.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise = np.random.randn(n_samples) * np.sqrt(1 - phi**2)
    x[0] = np.random.randn()
    for t in range(1, n_samples):
        x[t] = phi * x[t-1] + noise[t]
    return x


def generate_ar2(n_samples=5000, phi1=0.5, phi2=0.3, seed=42):
    """
    Generate AR(2) process: x[t] = phi1*x[t-1] + phi2*x[t-2] + noise
    
    Two-lag memory structure.
    """
    np.random.seed(seed)
    x = np.zeros(n_samples)
    noise_var = 1 - phi1**2 - phi2**2 - 2*phi1**2*phi2/(1-phi2)
    noise_var = max(0.1, noise_var)
    noise = np.random.randn(n_samples) * np.sqrt(noise_var)
    x[0] = np.random.randn()
    x[1] = np.random.randn()
    for t in range(2, n_samples):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + noise[t]
    return x


def generate_oscillation(n_samples=5000, period=10, seed=42):
    """
    Generate deterministic oscillation with noise.
    
    Periodic structure should show up in PhiID at matching lags.
    """
    np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.sin(2 * np.pi * t / period) + 0.1 * np.random.randn(n_samples)
    return x


# =============================================================================
# PhiID ANALYSIS FUNCTIONS
# =============================================================================

def compute_temporal_phiid(signal, extra_lag, tau=1, kind='gaussian', redundancy='MMI'):
    """
    Compute PhiID for temporal analysis of a single signal.
    
    Creates pseudo-bivariate system:
    - src = signal[:-extra_lag]  (X process: earlier)
    - tgt = signal[extra_lag:]   (Y process: later, shifted by extra_lag)
    
    Then PhiID analyzes the 4-way relationship using tau.
    """
    n = len(signal)
    
    if n <= extra_lag + tau + 10:
        return None
    
    # Create pseudo-bivariate system
    src = signal[:-extra_lag].copy()
    tgt = signal[extra_lag:].copy()
    
    try:
        atoms_res, calc_res = calc_PhiID(src, tgt, tau, kind=kind, redundancy=redundancy)
    except Exception as e:
        print(f"    PhiID error: {e}")
        return None
    
    # Average over time to get scalar values
    atoms = {}
    for name in ATOM_NAMES:
        if name in atoms_res:
            atoms[name] = float(np.nanmean(atoms_res[name]))
        else:
            atoms[name] = np.nan
    
    return atoms


def analyze_process(signal, name, extra_lags, tau=1, kind='gaussian'):
    """Analyze a process across multiple extra_lag values."""
    results = []
    
    for extra_lag in extra_lags:
        atoms = compute_temporal_phiid(signal, extra_lag, tau, kind)
        if atoms is None:
            continue
        atoms['extra_lag'] = extra_lag
        atoms['process'] = name
        results.append(atoms)
    
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_atoms_comparison(all_results, save_path=None):
    """Compare all 16 atoms across processes."""
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
    
    plt.suptitle('PhiID Atoms Across Toy Processes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_dynamics_comparison(all_results, save_path=None):
    """Compare information dynamics measures across processes."""
    # Compute dynamics for each row
    dynamics_data = []
    
    for _, row in all_results.iterrows():
        entry = {
            'process': row['process'],
            'extra_lag': row['extra_lag']
        }
        for metric, atoms in INFORMATION_DYNAMICS.items():
            entry[metric] = sum(row.get(a, 0) for a in atoms if a in row)
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
    
    plt.suptitle('Information Dynamics Across Toy Processes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_dynamics


def plot_summary_heatmap(all_results, extra_lag=1, save_path=None):
    """Heatmap of atoms for each process at a fixed lag."""
    df_lag = all_results[all_results['extra_lag'] == extra_lag]
    
    if len(df_lag) == 0:
        print(f"No data for extra_lag={extra_lag}")
        return
    
    # Create matrix: processes x atoms
    processes = df_lag['process'].unique()
    matrix = np.zeros((len(processes), len(ATOM_NAMES)))
    
    for i, proc in enumerate(processes):
        row = df_lag[df_lag['process'] == proc].iloc[0]
        for j, atom in enumerate(ATOM_NAMES):
            matrix[i, j] = row.get(atom, 0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(matrix, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=ATOM_NAMES, yticklabels=processes,
                annot=True, fmt='.3f', cbar_kws={'label': 'bits'})
    
    ax.set_xlabel('PhiID Atom')
    ax.set_ylabel('Process')
    ax.set_title(f'PhiID Atoms at Extra Lag = {extra_lag}', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_key_atoms_bar(all_results, extra_lag=1, save_path=None):
    """Bar chart of key atoms for each process."""
    df_lag = all_results[all_results['extra_lag'] == extra_lag]
    
    if len(df_lag) == 0:
        return
    
    key_atoms = ['rtr', 'xtx', 'yty', 'sts', 'xty', 'ytx']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    processes = df_lag['process'].unique()
    x = np.arange(len(processes))
    width = 0.12
    
    colors = ['green', 'blue', 'cyan', 'red', 'orange', 'purple']
    
    for i, atom in enumerate(key_atoms):
        values = [df_lag[df_lag['process'] == p][atom].values[0] for p in processes]
        ax.bar(x + i * width, values, width, label=atom, color=colors[i], alpha=0.8)
    
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(processes, rotation=45, ha='right')
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'Key PhiID Atoms by Process (Extra Lag = {extra_lag})', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_binary_vs_gaussian(binary_results, gaussian_results, save_path=None):
    """Compare discrete (binary) vs Gaussian PhiID."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel A: Storage comparison
    ax = axes[0, 0]
    for df, label, style in [(binary_results, 'Binary', 'o-'), 
                              (gaussian_results, 'Gaussian', 's--')]:
        for proc in df['process'].unique():
            df_proc = df[df['process'] == proc]
            storage = df_proc['rtr'] + df_proc['xtx'] + df_proc['yty'] + df_proc['sts']
            ax.plot(df_proc['extra_lag'], storage, style, 
                   label=f'{proc} ({label})', alpha=0.7)
    ax.set_xlabel('Extra Lag')
    ax.set_ylabel('Total Storage (bits)')
    ax.set_title('A) Storage: Binary vs Gaussian')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel B: Transfer comparison
    ax = axes[0, 1]
    for df, label, style in [(binary_results, 'Binary', 'o-'), 
                              (gaussian_results, 'Gaussian', 's--')]:
        for proc in df['process'].unique():
            df_proc = df[df['process'] == proc]
            transfer = df_proc['xty'] + df_proc['ytx']
            ax.plot(df_proc['extra_lag'], transfer, style,
                   label=f'{proc} ({label})', alpha=0.7)
    ax.set_xlabel('Extra Lag')
    ax.set_ylabel('Total Transfer (bits)')
    ax.set_title('B) Transfer: Binary vs Gaussian')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel C: rtr (redundant transfer)
    ax = axes[1, 0]
    for df, label, style in [(binary_results, 'Binary', 'o-'), 
                              (gaussian_results, 'Gaussian', 's--')]:
        for proc in df['process'].unique():
            df_proc = df[df['process'] == proc]
            ax.plot(df_proc['extra_lag'], df_proc['rtr'], style,
                   label=f'{proc} ({label})', alpha=0.7)
    ax.set_xlabel('Extra Lag')
    ax.set_ylabel('rtr (bits)')
    ax.set_title('C) Redundant Transfer (rtr)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel D: sts (synergistic storage)
    ax = axes[1, 1]
    for df, label, style in [(binary_results, 'Binary', 'o-'), 
                              (gaussian_results, 'Gaussian', 's--')]:
        for proc in df['process'].unique():
            df_proc = df[df['process'] == proc]
            ax.plot(df_proc['extra_lag'], df_proc['sts'], style,
                   label=f'{proc} ({label})', alpha=0.7)
    ax.set_xlabel('Extra Lag')
    ax.set_ylabel('sts (bits)')
    ax.set_title('D) Synergistic Storage (sts)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.suptitle('PhiID: Binary (Discrete) vs Gaussian Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_interpretation_figure(save_path=None):
    """Create a figure explaining PhiID interpretation."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    text = """
    PhiID INTERPRETATION FOR TEMPORAL ANALYSIS
    ==========================================
    
    SETUP:
    • src = signal[:-extra_lag]   →  X process (earlier portion)
    • tgt = signal[extra_lag:]    →  Y process (later portion)
    • tau = embedding delay (past/future within each process)
    
    Creates 4 vectors:
    • X_past:   signal at time t-tau
    • X_future: signal at time t
    • Y_past:   signal at time t+extra_lag-tau  
    • Y_future: signal at time t+extra_lag
    
    
    KEY ATOMS (16 total, organized in 4x4):
    ┌─────────────────────────────────────────────────────────────────┐
    │ Source →      │ Redundancy    │ X-unique     │ Y-unique    │ Synergy      │
    │ Target ↓      │    (r)        │    (x)       │    (y)      │    (s)       │
    ├───────────────┼───────────────┼──────────────┼─────────────┼──────────────┤
    │ Redundancy (r)│ rtr           │ xtr          │ ytr         │ str          │
    │ X-unique (x)  │ rtx           │ xtx          │ ytx         │ stx          │
    │ Y-unique (y)  │ rty           │ xty          │ yty         │ sty          │
    │ Synergy (s)   │ rts           │ xts          │ yts         │ sts          │
    └───────────────┴───────────────┴──────────────┴─────────────┴──────────────┘
    
    
    EXPECTED PATTERNS FOR TOY PROCESSES:
    
    • IID NOISE:       All atoms ≈ 0 (no temporal structure)
    
    • COPY (x[t]=x[t-1]): High xtx, yty (storage within each process)
                          High rtr (redundant transfer - past predicts future)
                          Low synergy (simple linear dependence)
    
    • XOR (x[t]=x[t-1]⊕x[t-2]): Synergistic patterns
                                 Information requires BOTH time points
                                 Should show in str, sts atoms
    
    • AR(1) (x[t]=φx[t-1]+ε): Similar to COPY but scaled by φ
                               Higher φ → more rtr, xtx, yty
    
    • OSCILLATION:     Periodic structure
                       High atoms at lags matching period
                       Storage and transfer vary with extra_lag
    
    
    INFORMATION DYNAMICS GROUPINGS:
    • Storage:    rtr + xtx + yty + sts  (info preserved over time)
    • Transfer:   xty + ytx              (info flow between processes)
    • Copy:       xtx + yty              (within-process continuity)
    • Erasure:    rtx + rty              (lost information)
    • Causation:  Downward (sty, stx, str) vs Upward (xts, yts, rts)
    """
    
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("TOY PhiID ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Parameters
    N_SAMPLES = 10000
    TAU = 1  # PhiID embedding delay
    EXTRA_LAGS = [1, 2, 3, 5, 10, 15, 20, 30, 50]  # Lags to analyze
    
    print(f"\nParameters:")
    print(f"  N samples: {N_SAMPLES}")
    print(f"  Tau: {TAU}")
    print(f"  Extra lags: {EXTRA_LAGS}")
    
    # =========================================================================
    # PART 1: Binary/Discrete Processes
    # =========================================================================
    print("\n" + "="*70)
    print("PART 1: DISCRETE (BINARY) PROCESSES")
    print("="*70)
    
    binary_processes = {
        'IID': generate_iid_binary(N_SAMPLES),
        'XOR': generate_xor_process(N_SAMPLES),
        'AND': generate_and_process(N_SAMPLES),
        'OR': generate_or_process(N_SAMPLES),
    }
    
    binary_results = []
    for name, signal in binary_processes.items():
        print(f"\nAnalyzing {name}...")
        df = analyze_process(signal, name, EXTRA_LAGS, tau=TAU, kind='discrete')
        if len(df) > 0:
            binary_results.append(df)
            print(f"  ✓ {len(df)} lag points analyzed")
    
    df_binary = pd.concat(binary_results, ignore_index=True) if binary_results else pd.DataFrame()
    
    # =========================================================================
    # PART 2: Gaussian/Continuous Processes
    # =========================================================================
    print("\n" + "="*70)
    print("PART 2: GAUSSIAN (CONTINUOUS) PROCESSES")
    print("="*70)
    
    gaussian_processes = {
        'IID_Gauss': generate_iid_gaussian(N_SAMPLES),
        'AR1_0.5': generate_ar1(N_SAMPLES, phi=0.5),
        'AR1_0.9': generate_ar1(N_SAMPLES, phi=0.9),
        'AR2': generate_ar2(N_SAMPLES, phi1=0.5, phi2=0.3),
        'Noisy_Copy': generate_noisy_copy(N_SAMPLES, noise_level=0.1),
        'Oscillation': generate_oscillation(N_SAMPLES, period=10),
    }
    
    gaussian_results = []
    for name, signal in gaussian_processes.items():
        print(f"\nAnalyzing {name}...")
        df = analyze_process(signal, name, EXTRA_LAGS, tau=TAU, kind='gaussian')
        if len(df) > 0:
            gaussian_results.append(df)
            print(f"  ✓ {len(df)} lag points analyzed")
    
    df_gaussian = pd.concat(gaussian_results, ignore_index=True) if gaussian_results else pd.DataFrame()
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    if len(df_binary) > 0:
        df_binary.to_csv(RESULTS_DIR / "binary_processes.csv", index=False)
        print(f"  Saved: binary_processes.csv")
    
    if len(df_gaussian) > 0:
        df_gaussian.to_csv(RESULTS_DIR / "gaussian_processes.csv", index=False)
        print(f"  Saved: gaussian_processes.csv")
    
    # Combine for overall comparison
    df_all = pd.concat([df_binary, df_gaussian], ignore_index=True)
    df_all.to_csv(RESULTS_DIR / "all_toy_results.csv", index=False)
    
    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    # 1. Interpretation guide
    print("\n1. Creating interpretation guide...")
    create_interpretation_figure(RESULTS_DIR / "phiid_interpretation.png")
    
    # 2. Binary process comparison
    if len(df_binary) > 0:
        print("\n2. Binary process analysis...")
        plot_atoms_comparison(df_binary, RESULTS_DIR / "binary_atoms.png")
        plot_summary_heatmap(df_binary, extra_lag=1, 
                            save_path=RESULTS_DIR / "binary_heatmap_lag1.png")
        plot_key_atoms_bar(df_binary, extra_lag=1,
                          save_path=RESULTS_DIR / "binary_key_atoms.png")
    
    # 3. Gaussian process comparison  
    if len(df_gaussian) > 0:
        print("\n3. Gaussian process analysis...")
        plot_atoms_comparison(df_gaussian, RESULTS_DIR / "gaussian_atoms.png")
        plot_dynamics_comparison(df_gaussian, RESULTS_DIR / "gaussian_dynamics.png")
        plot_summary_heatmap(df_gaussian, extra_lag=1,
                            save_path=RESULTS_DIR / "gaussian_heatmap_lag1.png")
        plot_key_atoms_bar(df_gaussian, extra_lag=1,
                          save_path=RESULTS_DIR / "gaussian_key_atoms.png")
    
    # 4. Binary vs Gaussian comparison (for overlapping process types)
    if len(df_binary) > 0 and len(df_gaussian) > 0:
        print("\n4. Binary vs Gaussian comparison...")
        # Compare IID processes
        df_binary_iid = df_binary[df_binary['process'] == 'IID'].copy()
        df_binary_iid['process'] = 'IID_Binary'
        df_gauss_iid = df_gaussian[df_gaussian['process'] == 'IID_Gauss'].copy()
        df_comparison = pd.concat([df_binary_iid, df_gauss_iid])
        if len(df_comparison) > 0:
            plot_atoms_comparison(df_comparison, 
                                 RESULTS_DIR / "iid_comparison.png")
    
    # 5. All processes overview
    print("\n5. All processes overview...")
    plot_atoms_comparison(df_all, RESULTS_DIR / "all_processes_atoms.png")
    plot_dynamics_comparison(df_all, RESULTS_DIR / "all_processes_dynamics.png")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Print key findings at lag=1
    print("\nKey PhiID atoms at extra_lag=1:")
    print("-" * 60)
    
    df_lag1 = df_all[df_all['extra_lag'] == 1]
    if len(df_lag1) > 0:
        summary_atoms = ['rtr', 'xtx', 'yty', 'sts', 'xty', 'ytx']
        print(f"{'Process':<15} " + " ".join(f"{a:>8}" for a in summary_atoms))
        print("-" * 60)
        for proc in df_lag1['process'].unique():
            row = df_lag1[df_lag1['process'] == proc].iloc[0]
            values = [f"{row.get(a, 0):8.4f}" for a in summary_atoms]
            print(f"{proc:<15} " + " ".join(values))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
