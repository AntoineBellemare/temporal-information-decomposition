"""
EEG Temporal PhiID Analysis (Takens Embedding)
===============================================

Use PhiID (Integrated Information Decomposition) for temporal structure analysis
of EEG signals across frequency bands.

KEY IMPROVEMENT (v2 - Takens):
- v1: Used irregular two-lag embedding (tau + extra_lag) with overlap artifacts
- v2: Uses TRUE Takens delay embedding with single τ parameter

Takens Embedding:
  Creates 4 vectors with PERFECTLY REGULAR spacing:
    p1 = x(t)           "X past"
    p2 = x(t + τ)       "Y past"
    t1 = x(t + 2τ)      "X future"
    t2 = x(t + 3τ)      "Y future"
  
  Timeline: t → t+τ → t+2τ → t+3τ (uniform spacing)

Frequency Bands:
- Delta: 1-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Gamma: 30-50 Hz

Usage:
    python eeg_phiid_temporal.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as sig

# Import PhiID - use direct internal functions for Takens embedding
from phyid.calculate import (
    calc_PhiID,
    _get_entropy_four_vec,
    _get_coinfo_four_vec,
    _get_redundancy_four_vec,
    _get_double_redundancy_four_vec,
    _get_atoms_four_vec
)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/phiid/ to project root
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "eeg"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PhiID atom names (16 atoms)
ATOM_NAMES = [
    "rtr", "rtx", "rty", "rts",  # Redundancy → (redundancy, X, Y, synergy)
    "xtr", "xtx", "xty", "xts",  # X-unique → (redundancy, X, Y, synergy)
    "ytr", "ytx", "yty", "yts",  # Y-unique → (redundancy, X, Y, synergy)
    "str", "stx", "sty", "sts",  # Synergy → (redundancy, X, Y, synergy)
]

# Grouped measures (from your goofi code)
INFORMATION_DYNAMICS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward_causation": ["sty", "stx", "str"],
    "Upward_causation": ["xts", "yts", "rts"],
}

IIT_METRICS = {
    "Info_storage": ["xtx", "yty", "rtr", "sts"],
    "Transfer_entropy": ["xty", "xtr", "str", "sty"],
    "Causal_density": ["xtr", "ytr", "sty", "str", "str", "xty", "ytx", "stx"],
    "Integrated_info": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],  # minus rtr
}

# Frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

BAND_COLORS = {
    'delta': '#1f77b4',
    'theta': '#2ca02c',
    'alpha': '#ff7f0e',
    'beta': '#d62728',
    'gamma': '#9467bd'
}


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.99))
    high = max(low + 0.01, min(highcut / nyq, 0.99))
    
    b, a = sig.butter(order, [low, high], btype='band')
    try:
        return sig.filtfilt(b, a, data)
    except:
        return sig.lfilter(b, a, data)


def estimate_decorrelation_tau(signal, fs, threshold=0.5, max_tau_ms=500):
    """
    Estimate the minimum τ where autocorrelation drops below threshold.
    
    This identifies the timescale where the signal becomes "sufficiently different"
    from itself to avoid trivial redundancy inflation.
    
    Parameters
    ----------
    signal : array
        1D time series
    fs : float
        Sampling frequency in Hz
    threshold : float
        ACF threshold (default 0.5 = half-power point)
        Lower = more conservative (larger τ required)
    max_tau_ms : float
        Maximum τ to consider in milliseconds
    
    Returns
    -------
    min_tau : int
        Minimum τ in samples where ACF < threshold
    acf_at_min : float
        ACF value at that τ
    first_zero : int or None
        First zero-crossing of ACF (if exists)
    """
    max_tau_samples = int(max_tau_ms * fs / 1000)
    max_tau_samples = min(max_tau_samples, len(signal) // 4)
    
    # Compute normalized autocorrelation
    signal_centered = signal - np.mean(signal)
    acf = np.correlate(signal_centered, signal_centered, mode='full')
    acf = acf[len(acf)//2:]  # Keep positive lags only
    acf = acf / acf[0]  # Normalize so ACF(0) = 1
    
    # Find first point where ACF < threshold
    min_tau = 1
    acf_at_min = acf[1] if len(acf) > 1 else 1.0
    
    for i in range(1, min(max_tau_samples, len(acf))):
        if acf[i] < threshold:
            min_tau = i
            acf_at_min = acf[i]
            break
    else:
        # Never dropped below threshold - use max
        min_tau = max_tau_samples
        acf_at_min = acf[min(max_tau_samples, len(acf)-1)]
    
    # Also find first zero crossing or local minimum
    first_zero = None
    for i in range(1, min(max_tau_samples, len(acf))):
        if acf[i] <= 0:
            first_zero = i
            break
        # Check for local minimum
        if i > 1 and acf[i] > acf[i-1] and acf[i-1] < acf[i-2]:
            first_zero = i - 1  # Use the minimum
            break
    
    return min_tau, acf_at_min, first_zero


def filter_tau_values_by_acf(tau_values, min_tau, strategy='threshold'):
    """
    Filter τ values to only include those beyond the decorrelation threshold.
    
    Parameters
    ----------
    tau_values : list
        Original list of τ values in samples
    min_tau : int
        Minimum τ from ACF analysis
    strategy : str
        'threshold': Only use τ >= min_tau
        'relaxed': Use τ >= min_tau/2 (more inclusive)
        'all': Keep all τ but mark which are "trivial"
    
    Returns
    -------
    valid_taus : list
        Filtered τ values
    """
    if strategy == 'threshold':
        return [t for t in tau_values if t >= min_tau]
    elif strategy == 'relaxed':
        return [t for t in tau_values if t >= min_tau // 2]
    else:  # 'all'
        return tau_values


def get_band_decorrelation_info(signal, fs, band_name):
    """
    Get decorrelation info for a bandpass-filtered signal.
    
    Different frequency bands have very different autocorrelation structures:
    - Delta (1-4 Hz): Very slow, ACF stays high for 100s of ms
    - Gamma (30-50 Hz): Fast, ACF drops quickly (~10-20ms)
    
    Returns min_tau and recommended τ range for the band.
    """
    # Estimate decorrelation at 0.5 threshold
    min_tau_05, acf_05, first_zero = estimate_decorrelation_tau(signal, fs, threshold=0.5)
    
    # Also estimate at 0.3 for more conservative threshold
    min_tau_03, acf_03, _ = estimate_decorrelation_tau(signal, fs, threshold=0.3)
    
    # Use first zero if available, otherwise 0.3 threshold
    recommended_min = first_zero if first_zero else min_tau_03
    
    return {
        'band': band_name,
        'min_tau_acf05': min_tau_05,
        'min_tau_acf03': min_tau_03,
        'first_zero': first_zero,
        'recommended_min_tau': recommended_min,
        'recommended_min_ms': recommended_min / fs * 1000 if recommended_min else None,
    }


def create_lag_schedule(fs, timescales_ms):
    """Create lag values from timescales in ms."""
    lags = [max(1, int(t * fs / 1000)) for t in timescales_ms]
    return sorted(list(set(lags)))


# =============================================================================
# PhiID TEMPORAL ANALYSIS
# =============================================================================

def compute_temporal_phiid(signal, tau_embed, kind='gaussian', redundancy='MMI'):
    """
    Compute PhiID using TRUE Takens delay embedding.
    
    Creates 4 vectors with PERFECTLY REGULAR spacing:
        p1 = x(t)           "X past"
        p2 = x(t + τ)       "Y past"  
        t1 = x(t + 2τ)      "X future"
        t2 = x(t + 3τ)      "Y future"
    
    Timeline: t → t+τ → t+2τ → t+3τ (uniform spacing!)
    
    Parameters
    ----------
    signal : array
        1D time series (continuous values work best with kind='gaussian')
    tau_embed : int
        Embedding delay in samples - the ONLY temporal parameter!
        This directly controls the timescale being probed.
    kind : str
        'gaussian' for continuous, 'discrete' for binarized
    redundancy : str
        'MMI' or 'CCS' for redundancy measure
        
    Returns
    -------
    atoms : dict
        Dictionary with mean value for each of 16 atoms
    dynamics : dict
        Summary metrics for information dynamics
    iit : dict
        Summary metrics for IIT concepts
    """
    N = len(signal) - 3 * tau_embed
    
    if N < 50:  # Need minimum samples for reliable estimation
        return None, None, None
    
    # Construct 4D Takens embedding with PERFECTLY REGULAR spacing
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
        # PhiID computation pipeline (direct, bypassing calc_PhiID)
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
        
        # Compute summary metrics
        dynamics = {}
        for metric_name, atom_list in INFORMATION_DYNAMICS.items():
            dynamics[metric_name] = sum(atoms.get(a, 0) for a in atom_list)
        
        iit = {}
        for metric_name, atom_list in IIT_METRICS.items():
            iit[metric_name] = sum(atoms.get(a, 0) for a in atom_list)
            if metric_name == "Integrated_info":
                iit[metric_name] -= atoms.get("rtr", 0)  # Subtract redundancy
        
        return atoms, dynamics, iit
        
    except Exception as e:
        print(f"    PhiID error: {e}")
        return None, None, None


def analyze_channel_temporal_phiid(signal, tau_values, kind='gaussian', redundancy='MMI'):
    """
    Analyze a channel across multiple tau (embedding delay) values.
    
    Parameters
    ----------
    signal : array
        1D time series
    tau_values : list of int
        List of embedding delays to probe (in samples)
        Each τ probes dynamics at ~fs/(4τ) Hz
    kind : str
        'gaussian' or 'discrete'
    redundancy : str
        'MMI' or 'CCS'
    
    Returns DataFrames for atoms, dynamics, and IIT metrics.
    """
    all_atoms = []
    all_dynamics = []
    all_iit = []
    
    for tau_embed in tau_values:
        atoms, dynamics, iit = compute_temporal_phiid(signal, tau_embed, kind, redundancy)
        
        if atoms is None:
            continue
        
        atoms['tau_embed'] = tau_embed
        dynamics['tau_embed'] = tau_embed
        iit['tau_embed'] = tau_embed
        
        all_atoms.append(atoms)
        all_dynamics.append(dynamics)
        all_iit.append(iit)
    
    if not all_atoms:
        return None, None, None
    
    return pd.DataFrame(all_atoms), pd.DataFrame(all_dynamics), pd.DataFrame(all_iit)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_phiid_atoms_heatmap(df_atoms, channel, fs, save_path=None):
    """Heatmap of all 16 atoms vs τ embedding delay."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pivot to matrix
    atom_cols = [c for c in df_atoms.columns if c != 'tau_embed']
    matrix = df_atoms[atom_cols].values.T
    
    time_labels = [f'{l/fs*1000:.0f}' for l in df_atoms['tau_embed']]
    
    sns.heatmap(matrix, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=time_labels, yticklabels=atom_cols,
                cbar_kws={'label': 'bits'})
    
    ax.set_xlabel('τ Embedding Delay (ms)')
    ax.set_ylabel('PhiID Atom')
    ax.set_title(f'PhiID Atoms vs Takens τ: {channel}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_dynamics_summary(df_dynamics, channel, fs, save_path=None):
    """Plot information dynamics metrics vs τ."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    time_ms = df_dynamics['tau_embed'] / fs * 1000
    metrics = [c for c in df_dynamics.columns if c != 'tau_embed']
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(time_ms, df_dynamics[metric], 'o-', color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel('τ Embedding Delay (ms)')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' '))
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    plt.suptitle(f'Information Dynamics (Takens): {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_iit_summary(df_iit, channel, fs, save_path=None):
    """Plot IIT metrics vs τ."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    time_ms = df_iit['tau_embed'] / fs * 1000
    metrics = [c for c in df_iit.columns if c != 'tau_embed']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(time_ms, df_iit[metric], 'o-', color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel('τ Embedding Delay (ms)')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' '))
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    plt.suptitle(f'IIT Metrics (Takens): {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_band_comparison_phiid(all_results, metric, fs, save_path=None):
    """Compare a specific metric across bands."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for band_name, results in all_results.items():
        if results is None:
            continue
        
        # Average across channels
        dfs = [r['dynamics'] for r in results.values() if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        
        combined = pd.concat(dfs)
        grouped = combined.groupby('tau_embed')[metric].agg(['mean', 'std']).reset_index()
        
        time_ms = grouped['tau_embed'] / fs * 1000
        
        ax.errorbar(time_ms, grouped['mean'], yerr=grouped['std'],
                   fmt='o-', color=BAND_COLORS[band_name], label=band_name.upper(),
                   linewidth=2, capsize=3)
    
    ax.set_xlabel('τ Embedding Delay (ms)')
    ax.set_ylabel(f'{metric} (bits)')
    ax.set_title(f'{metric.replace("_", " ")} Across Frequency Bands (Takens)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_grand_summary_phiid(all_results, fs, save_path=None):
    """Grand summary of PhiID analysis."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Panel A: Storage across bands
    ax_storage = fig.add_subplot(gs[0, 0])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['dynamics'] for r in all_results[band_name].values() 
               if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('tau_embed')['Storage'].mean().reset_index()
        time_ms = grouped['tau_embed'] / fs * 1000
        ax_storage.plot(time_ms, grouped['Storage'], 'o-', color=BAND_COLORS[band_name],
                       label=band_name.upper(), linewidth=2)
    ax_storage.set_xlabel('τ (ms)')
    ax_storage.set_ylabel('Storage (bits)')
    ax_storage.set_title('A) Information Storage')
    ax_storage.legend(fontsize=8)
    ax_storage.grid(alpha=0.3)
    ax_storage.set_xscale('log')
    
    # Panel B: Transfer across bands
    ax_transfer = fig.add_subplot(gs[0, 1])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['dynamics'] for r in all_results[band_name].values() 
               if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('tau_embed')['Transfer'].mean().reset_index()
        time_ms = grouped['tau_embed'] / fs * 1000
        ax_transfer.plot(time_ms, grouped['Transfer'], 'o-', color=BAND_COLORS[band_name],
                        label=band_name.upper(), linewidth=2)
    ax_transfer.set_xlabel('τ (ms)')
    ax_transfer.set_ylabel('Transfer (bits)')
    ax_transfer.set_title('B) Information Transfer')
    ax_transfer.legend(fontsize=8)
    ax_transfer.grid(alpha=0.3)
    ax_transfer.set_xscale('log')
    
    # Panel C: Integrated Info
    ax_phi = fig.add_subplot(gs[0, 2])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['iit'] for r in all_results[band_name].values() 
               if r is not None and r['iit'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('tau_embed')['Integrated_info'].mean().reset_index()
        time_ms = grouped['tau_embed'] / fs * 1000
        ax_phi.plot(time_ms, grouped['Integrated_info'], 'o-', color=BAND_COLORS[band_name],
                   label=band_name.upper(), linewidth=2)
    ax_phi.set_xlabel('τ (ms)')
    ax_phi.set_ylabel('Φ (bits)')
    ax_phi.set_title('C) Integrated Information')
    ax_phi.legend(fontsize=8)
    ax_phi.grid(alpha=0.3)
    ax_phi.set_xscale('log')
    
    # Panel D: Downward vs Upward causation for alpha
    ax_causation = fig.add_subplot(gs[1, 0])
    if 'alpha' in all_results and all_results['alpha'] is not None:
        dfs = [r['dynamics'] for r in all_results['alpha'].values() 
               if r is not None and r['dynamics'] is not None]
        if dfs:
            combined = pd.concat(dfs)
            grouped = combined.groupby('tau_embed').agg({
                'Downward_causation': 'mean',
                'Upward_causation': 'mean'
            }).reset_index()
            time_ms = grouped['tau_embed'] / fs * 1000
            ax_causation.plot(time_ms, grouped['Downward_causation'], 'o-', 
                             color='purple', label='Downward', linewidth=2)
            ax_causation.plot(time_ms, grouped['Upward_causation'], 's-', 
                             color='orange', label='Upward', linewidth=2)
    ax_causation.set_xlabel('τ (ms)')
    ax_causation.set_ylabel('Causation (bits)')
    ax_causation.set_title('D) Causation Direction (Alpha)')
    ax_causation.legend()
    ax_causation.grid(alpha=0.3)
    ax_causation.set_xscale('log')
    
    # Panel E: Bars by band
    ax_bars = fig.add_subplot(gs[1, 1])
    
    bar_data = []
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs_dyn = [r['dynamics'] for r in all_results[band_name].values() 
                   if r is not None and r['dynamics'] is not None]
        if not dfs_dyn:
            continue
        combined = pd.concat(dfs_dyn)
        bar_data.append({
            'band': band_name,
            'Storage': combined['Storage'].mean(),
            'Transfer': combined['Transfer'].mean(),
            'Copy': combined['Copy'].mean()
        })
    
    if bar_data:
        df_bars = pd.DataFrame(bar_data)
        x = np.arange(len(df_bars))
        width = 0.25
        
        ax_bars.bar(x - width, df_bars['Storage'], width, label='Storage', color='blue', alpha=0.7)
        ax_bars.bar(x, df_bars['Transfer'], width, label='Transfer', color='green', alpha=0.7)
        ax_bars.bar(x + width, df_bars['Copy'], width, label='Copy', color='red', alpha=0.7)
        ax_bars.set_xticks(x)
        ax_bars.set_xticklabels([b.upper() for b in df_bars['band']])
        ax_bars.set_ylabel('Mean (bits)')
        ax_bars.set_title('E) Dynamics by Band')
        ax_bars.legend()
        ax_bars.grid(axis='y', alpha=0.3)
    
    # Panel F: Interpretation
    ax_interp = fig.add_subplot(gs[1, 2])
    ax_interp.axis('off')
    
    interpretation = """
    PhiID TEMPORAL ANALYSIS
    =======================
    
    WHAT WE'RE MEASURING:
    • Temporal PhiID: How info flows between
      a signal and its time-shifted self
    
    KEY METRICS:
    • Storage: Info preserved over time
    • Transfer: Info flowing forward in time  
    • Copy: Redundant temporal structure
    • Causation: Direction of info flow
    
    INTERPRETATION:
    • High Storage: Persistent dynamics
    • High Transfer: Active processing
    • High Φ: Integrated temporal structure
    
    BAND SIGNATURES:
    • Delta: High storage (slow persistence)
    • Alpha: Periodic transfer patterns
    • Gamma: Rapid transfer, low storage
    """
    
    ax_interp.text(0.05, 0.95, interpretation, transform=ax_interp.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('EEG Temporal PhiID Analysis', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EEG TEMPORAL PhiID ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    eeg_file = DATA_DIR / "test_eeg_dsi.csv"
    print(f"\nLoading: {eeg_file}")
    df = pd.read_csv(eeg_file)
    
    # Get EEG channels
    channels = [c for c in df.columns if c.startswith('eeg-')]
    print(f"Found {len(channels)} EEG channels")
    print(f"Total samples: {len(df)}")
    
    # Estimate sampling rate
    try:
        ts = pd.to_datetime(df['timestamp'])
        dt = (ts.iloc[1] - ts.iloc[0]).total_seconds()
        fs = 1 / dt
    except:
        fs = 300
    print(f"Sampling rate: {fs:.1f} Hz")
    
    # Parameters
    SUBSAMPLE = 50000  # Smaller for PhiID (more expensive)
    KIND = 'gaussian'  # Use Gaussian for continuous EEG
    REDUNDANCY = 'MMI'
    ACF_FILTER_STRATEGY = 'threshold'  # 'threshold', 'relaxed', or 'all'
    
    # Takens embedding delays (τ values) - uniform spacing: t, t+τ, t+2τ, t+3τ
    # Extended to probe SLOW dynamics (attention, cognitive states, etc.)
    # With 50000 samples at 300Hz = 166.7 seconds, max τ could be ~40 seconds (12000 samples)
    # But practical limit: need enough windows for statistics
    TIMESCALES_MS = [
        # Fast dynamics (gamma/beta)
        10, 20, 30, 50, 75, 100,
        # Medium dynamics (alpha/theta)
        150, 200, 300, 500, 750, 1000,
        # Slow dynamics (delta, attention, cognitive states)
        1500, 2000, 3000, 5000,
        # Very slow dynamics (if signal is long enough)
        # 7500, 10000
    ]
    ALL_TAU_VALUES = create_lag_schedule(fs, TIMESCALES_MS)
    
    # Verify signal length is sufficient for largest τ
    # Takens needs 3τ samples, plus some buffer
    max_tau = max(ALL_TAU_VALUES)
    min_samples_needed = 4 * max_tau  # 3τ for embedding + 1τ buffer
    if SUBSAMPLE < min_samples_needed:
        print(f"  Warning: SUBSAMPLE ({SUBSAMPLE}) may be too small for max τ={max_tau}")
        print(f"           Need at least {min_samples_needed} samples")
        # Reduce τ range
        safe_max_tau = SUBSAMPLE // 4
        ALL_TAU_VALUES = [t for t in ALL_TAU_VALUES if t <= safe_max_tau]
        print(f"           Limiting τ to max {safe_max_tau} samples ({safe_max_tau/fs*1000:.0f}ms)")
    
    # Select key channels for analysis
    KEY_CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    channels_to_analyze = [c for c in KEY_CHANNELS if c in channels]
    
    print(f"\nParameters:")
    print(f"  Samples: {SUBSAMPLE} ({SUBSAMPLE/fs:.1f}s)")
    print(f"  Kind: {KIND}")
    print(f"  Redundancy: {REDUNDANCY}")
    print(f"  ACF Filter Strategy: {ACF_FILTER_STRATEGY}")
    print(f"  Candidate τ values ({len(ALL_TAU_VALUES)}): {ALL_TAU_VALUES}")
    print(f"  Timescales: {TIMESCALES_MS} ms")
    print(f"  Max embedding span: {3*max(ALL_TAU_VALUES)/fs*1000:.0f}ms (3×τ_max)")
    print(f"  Channels: {channels_to_analyze}")
    
    # Storage
    all_results = {}  # {band: {channel: {atoms, dynamics, iit}}}
    band_acf_info = {}  # Store ACF info for each band
    
    # Process each band
    for band_name, (fmin, fmax) in BANDS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {band_name.upper()} ({fmin}-{fmax} Hz)")
        print("="*60)
        
        all_results[band_name] = {}
        
        # Get ACF info from first valid channel to determine band-specific τ values
        first_valid_signal = None
        for ch in channels_to_analyze:
            signal = df[ch].values[:SUBSAMPLE]
            signal = signal[~np.isnan(signal)]
            if len(signal) > 1000:
                filtered = bandpass_filter(signal, fmin, fmax, fs)
                first_valid_signal = filtered
                break
        
        if first_valid_signal is not None:
            acf_info = get_band_decorrelation_info(first_valid_signal, fs, band_name)
            band_acf_info[band_name] = acf_info
            
            # Filter τ values based on ACF
            min_tau = acf_info['recommended_min_tau'] or 1
            band_tau_values = filter_tau_values_by_acf(ALL_TAU_VALUES, min_tau, ACF_FILTER_STRATEGY)
            
            print(f"  ACF Analysis:")
            print(f"    Min τ (ACF<0.5): {acf_info['min_tau_acf05']} samples ({acf_info['min_tau_acf05']/fs*1000:.1f}ms)")
            print(f"    Min τ (ACF<0.3): {acf_info['min_tau_acf03']} samples ({acf_info['min_tau_acf03']/fs*1000:.1f}ms)")
            if acf_info['first_zero']:
                print(f"    First zero/min:  {acf_info['first_zero']} samples ({acf_info['first_zero']/fs*1000:.1f}ms)")
            print(f"    Using τ values:  {band_tau_values} (filtered from {len(ALL_TAU_VALUES)} → {len(band_tau_values)})")
        else:
            band_tau_values = ALL_TAU_VALUES
            print(f"  Warning: Could not estimate ACF, using all τ values")
        
        if not band_tau_values:
            print(f"  Warning: No valid τ values for this band, using minimum set")
            band_tau_values = ALL_TAU_VALUES[-3:]  # Use largest 3 τ values
        
        for ch in channels_to_analyze:
            print(f"  Channel: {ch}", end=" ")
            
            # Get signal
            signal = df[ch].values[:SUBSAMPLE]
            signal = signal[~np.isnan(signal)]
            
            try:
                # Bandpass filter
                filtered = bandpass_filter(signal, fmin, fmax, fs)
                
                # Analyze PhiID with Takens embedding (using band-specific τ values!)
                df_atoms, df_dynamics, df_iit = analyze_channel_temporal_phiid(
                    filtered, band_tau_values, kind=KIND, redundancy=REDUNDANCY
                )
                
                if df_atoms is None:
                    print("✗ failed")
                    all_results[band_name][ch] = None
                    continue
                
                all_results[band_name][ch] = {
                    'atoms': df_atoms,
                    'dynamics': df_dynamics,
                    'iit': df_iit,
                    'tau_values_used': band_tau_values,
                    'acf_info': acf_info if first_valid_signal is not None else None
                }
                
                print(f"✓ ({len(df_atoms)} τ values)")
                
            except Exception as e:
                print(f"✗ {e}")
                all_results[band_name][ch] = None
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)
    
    # Combine all atoms
    all_atoms_dfs = []
    all_dynamics_dfs = []
    all_iit_dfs = []
    
    for band_name, band_results in all_results.items():
        for ch, results in band_results.items():
            if results is None:
                continue
            
            atoms = results['atoms'].copy()
            atoms['channel'] = ch
            atoms['band'] = band_name
            all_atoms_dfs.append(atoms)
            
            dyn = results['dynamics'].copy()
            dyn['channel'] = ch
            dyn['band'] = band_name
            all_dynamics_dfs.append(dyn)
            
            iit = results['iit'].copy()
            iit['channel'] = ch
            iit['band'] = band_name
            all_iit_dfs.append(iit)
    
    if all_atoms_dfs:
        pd.concat(all_atoms_dfs).to_csv(RESULTS_DIR / "phiid_atoms.csv", index=False)
        pd.concat(all_dynamics_dfs).to_csv(RESULTS_DIR / "phiid_dynamics.csv", index=False)
        pd.concat(all_iit_dfs).to_csv(RESULTS_DIR / "phiid_iit.csv", index=False)
    
    # Save ACF info summary
    if band_acf_info:
        acf_summary = pd.DataFrame([
            {
                'band': band,
                'min_tau_acf05_samples': info['min_tau_acf05'],
                'min_tau_acf05_ms': info['min_tau_acf05'] / fs * 1000,
                'min_tau_acf03_samples': info['min_tau_acf03'],
                'min_tau_acf03_ms': info['min_tau_acf03'] / fs * 1000,
                'first_zero_samples': info['first_zero'],
                'first_zero_ms': info['first_zero'] / fs * 1000 if info['first_zero'] else None,
                'recommended_min_tau_samples': info['recommended_min_tau'],
                'recommended_min_tau_ms': info['recommended_min_ms'],
            }
            for band, info in band_acf_info.items()
        ])
        acf_summary.to_csv(RESULTS_DIR / "acf_thresholds_by_band.csv", index=False)
        
        print(f"\n  ACF Thresholds by Band:")
        print(acf_summary.to_string(index=False))
    
    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print("="*70)
    
    # 1. Grand summary
    print("\n1. Grand summary...")
    plot_grand_summary_phiid(all_results, fs, save_path=RESULTS_DIR / "grand_summary.png")
    
    # 2. Band comparisons for key metrics
    print("2. Band comparisons...")
    for metric in ['Storage', 'Transfer', 'Downward_causation']:
        plot_band_comparison_phiid(all_results, metric, fs,
                                   save_path=RESULTS_DIR / f"band_comparison_{metric.lower()}.png")
    
    # 3. Individual channel plots for alpha
    print("3. Alpha band channel details...")
    if 'alpha' in all_results:
        for ch, results in all_results['alpha'].items():
            if results is None:
                continue
            ch_name = ch.replace('eeg-', '')
            
            plot_phiid_atoms_heatmap(results['atoms'], ch, fs,
                                     save_path=RESULTS_DIR / f"atoms_{ch_name}_alpha.png")
            plot_dynamics_summary(results['dynamics'], ch, fs,
                                  save_path=RESULTS_DIR / f"dynamics_{ch_name}_alpha.png")
            plot_iit_summary(results['iit'], ch, fs,
                             save_path=RESULTS_DIR / f"iit_{ch_name}_alpha.png")
    
    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"\nFiles created:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
