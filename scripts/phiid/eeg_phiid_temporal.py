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
# Frequency bands - including BROADBAND for LRTC analysis
BANDS = {
    'broadband': (0.5, 45),   # Full EEG range - preserves LRTC/1/f structure!
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

BAND_COLORS = {
    'broadband': '#000000',  # Black for broadband
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


def highpass_filter(data, cutoff, fs, order=4):
    """Apply Butterworth highpass filter (for broadband with LRTC)."""
    nyq = 0.5 * fs
    high = max(0.001, min(cutoff / nyq, 0.99))
    
    b, a = sig.butter(order, high, btype='high')
    try:
        return sig.filtfilt(b, a, data)
    except:
        return sig.lfilter(b, a, data)


def compute_amplitude_envelope(data, lowcut, highcut, fs, smooth_ms=100):
    """
    Compute amplitude envelope of a bandpass-filtered signal.
    
    The amplitude envelope preserves LRTC even when the narrowband
    oscillation itself decorrelates quickly!
    
    Parameters
    ----------
    data : array
        Raw signal
    lowcut, highcut : float
        Band limits in Hz
    fs : float
        Sampling frequency
    smooth_ms : float
        Smoothing window in ms for envelope
    
    Returns
    -------
    envelope : array
        Amplitude envelope (always positive, slow fluctuations)
    """
    # Bandpass filter
    filtered = bandpass_filter(data, lowcut, highcut, fs)
    
    # Hilbert transform for analytic signal
    analytic = sig.hilbert(filtered)
    envelope = np.abs(analytic)
    
    # Smooth the envelope (optional, reduces high-freq noise)
    if smooth_ms > 0:
        smooth_samples = max(1, int(smooth_ms * fs / 1000))
        kernel = np.ones(smooth_samples) / smooth_samples
        envelope = np.convolve(envelope, kernel, mode='same')
    
    return envelope


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
    """Compare a specific metric across bands with dual y-axis for broadband."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()  # Secondary y-axis for broadband
    
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
        
        # Use secondary axis for broadband (different scale)
        if band_name == 'broadband':
            ax2.errorbar(time_ms, grouped['mean'], yerr=grouped['std'],
                        fmt='s--', color=BAND_COLORS[band_name], label=f'{band_name.upper()} (right axis)',
                        linewidth=2, capsize=3, markersize=8)
        else:
            ax.errorbar(time_ms, grouped['mean'], yerr=grouped['std'],
                       fmt='o-', color=BAND_COLORS[band_name], label=band_name.upper(),
                       linewidth=2, capsize=3)
    
    ax.set_xlabel('τ Embedding Delay (ms)')
    ax.set_ylabel(f'{metric} (bits) - Narrowband', color='black')
    ax2.set_ylabel(f'{metric} (bits) - BROADBAND', color=BAND_COLORS['broadband'])
    ax2.tick_params(axis='y', labelcolor=BAND_COLORS['broadband'])
    ax.set_title(f'{metric.replace("_", " ")} Across Frequency Bands (Takens)')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_band_all_channels(band_results, band_name, fs, save_path=None):
    """
    Consolidated plot for one band showing all channels and metrics.
    One figure per band with all channels as subplots.
    """
    channels = [ch for ch, r in band_results.items() if r is not None]
    if not channels:
        return
    
    n_channels = len(channels)
    metrics = ['Storage', 'Transfer', 'Copy', 'Erasure', 'Downward_causation', 'Upward_causation']
    
    fig, axes = plt.subplots(len(metrics), n_channels, figsize=(4*n_channels, 3*len(metrics)))
    if n_channels == 1:
        axes = axes.reshape(-1, 1)
    
    channel_colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
    
    for col, ch in enumerate(channels):
        results = band_results[ch]
        if results is None:
            continue
        
        df_dyn = results['dynamics']
        time_ms = df_dyn['tau_embed'] / fs * 1000
        ch_short = ch.replace('eeg-', '')
        
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            ax.plot(time_ms, df_dyn[metric], 'o-', color=channel_colors[col], 
                   linewidth=2, markersize=4)
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            
            if row == 0:
                ax.set_title(ch_short, fontsize=12, fontweight='bold')
            if col == 0:
                ax.set_ylabel(metric.replace('_', '\n'), fontsize=9)
            if row == len(metrics) - 1:
                ax.set_xlabel('τ (ms)')
    
    plt.suptitle(f'{band_name.upper()} Band - Information Dynamics', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_all_bands_summary(all_results, fs, save_path=None):
    """
    Master summary: rows=bands, cols=key metrics, averaged across channels.
    """
    band_order = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    metrics = ['Storage', 'Transfer', 'Integrated_info']
    
    fig, axes = plt.subplots(len(band_order), len(metrics), figsize=(5*len(metrics), 3*len(band_order)))
    
    for row, band_name in enumerate(band_order):
        if band_name not in all_results or all_results[band_name] is None:
            for col in range(len(metrics)):
                axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue
        
        band_results = all_results[band_name]
        channels = [ch for ch, r in band_results.items() if r is not None]
        
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            # Get data source (dynamics or iit)
            if metric == 'Integrated_info':
                dfs = [band_results[ch]['iit'] for ch in channels if band_results[ch] is not None]
            else:
                dfs = [band_results[ch]['dynamics'] for ch in channels if band_results[ch] is not None]
            
            if not dfs:
                continue
            
            # Plot each channel
            for i, ch in enumerate(channels):
                if band_results[ch] is None:
                    continue
                if metric == 'Integrated_info':
                    df = band_results[ch]['iit']
                else:
                    df = band_results[ch]['dynamics']
                
                time_ms = df['tau_embed'] / fs * 1000
                ch_short = ch.replace('eeg-', '')
                ax.plot(time_ms, df[metric], 'o-', alpha=0.7, markersize=3,
                       label=ch_short if row == 0 else None)
            
            ax.set_xscale('log')
            ax.grid(alpha=0.3)
            
            # Labels
            if row == 0:
                ax.set_title(metric.replace('_', ' '), fontsize=12, fontweight='bold')
                ax.legend(fontsize=7, loc='upper right')
            if col == 0:
                ax.set_ylabel(f'{band_name.upper()}\n(bits)', fontsize=10)
            if row == len(band_order) - 1:
                ax.set_xlabel('τ (ms)')
    
    plt.suptitle('PhiID Temporal Analysis: All Bands × Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_atoms_heatmap_all_channels(band_results, band_name, fs, save_path=None):
    """
    Heatmap of atoms for all channels in one band, side by side.
    """
    channels = [ch for ch, r in band_results.items() if r is not None]
    if not channels:
        return
    
    n_channels = len(channels)
    fig, axes = plt.subplots(1, n_channels, figsize=(5*n_channels, 8))
    if n_channels == 1:
        axes = [axes]
    
    atom_cols = ['rtr', 'rtx', 'rty', 'rts', 'xtr', 'xtx', 'xty', 'xts',
                 'ytr', 'ytx', 'yty', 'yts', 'str', 'stx', 'sty', 'sts']
    
    # Find global min/max for consistent colorbar
    all_vals = []
    for ch in channels:
        if band_results[ch] is not None:
            df = band_results[ch]['atoms']
            all_vals.extend(df[atom_cols].values.flatten())
    if all_vals:
        vmin, vmax = np.percentile(all_vals, [5, 95])
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
    else:
        vmin, vmax = -1, 1
    
    for i, ch in enumerate(channels):
        ax = axes[i]
        if band_results[ch] is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue
        
        df = band_results[ch]['atoms']
        matrix = df[atom_cols].values.T
        time_labels = [f'{int(t/fs*1000)}' for t in df['tau_embed']]
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(time_labels)))
        ax.set_xticklabels(time_labels, rotation=45, fontsize=8)
        ax.set_yticks(range(len(atom_cols)))
        ax.set_yticklabels(atom_cols, fontsize=8)
        ax.set_xlabel('τ (ms)')
        ax.set_title(ch.replace('eeg-', ''), fontsize=12, fontweight='bold')
        
        if i == 0:
            ax.set_ylabel('PhiID Atom')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label='bits')
    
    plt.suptitle(f'{band_name.upper()} - PhiID Atoms (All Channels)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def plot_grand_summary_phiid(all_results, fs, save_path=None):
    """Grand summary of PhiID analysis with proper broadband visibility."""
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Include broadband first to show LRTC comparison
    band_order = ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']
    narrowband_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Helper to get grouped data for a metric
    def get_band_data(band_name, metric, source='dynamics'):
        if band_name not in all_results or all_results[band_name] is None:
            return None, None
        dfs = [r[source] for r in all_results[band_name].values() 
               if r is not None and r[source] is not None]
        if not dfs:
            return None, None
        combined = pd.concat(dfs)
        grouped = combined.groupby('tau_embed')[metric].mean().reset_index()
        time_ms = grouped['tau_embed'] / fs * 1000
        return time_ms, grouped[metric]
    
    # Panel A: Storage (narrowband only, to show structure)
    ax_storage = fig.add_subplot(gs[0, 0])
    for band_name in narrowband_order:
        time_ms, values = get_band_data(band_name, 'Storage')
        if time_ms is not None:
            ax_storage.plot(time_ms, values, 'o-', color=BAND_COLORS[band_name],
                           label=band_name.upper(), linewidth=2)
    ax_storage.set_xlabel('τ (ms)')
    ax_storage.set_ylabel('Storage (bits)')
    ax_storage.set_title('A) Storage - Narrowband')
    ax_storage.legend(fontsize=8)
    ax_storage.grid(alpha=0.3)
    ax_storage.set_xscale('log')
    
    # Panel B: Broadband zoomed (own scale!)
    ax_broadband = fig.add_subplot(gs[0, 1])
    bb_metrics = ['Storage', 'Transfer', 'Copy']
    bb_colors = ['#1f77b4', '#2ca02c', '#d62728']
    for metric, color in zip(bb_metrics, bb_colors):
        time_ms, values = get_band_data('broadband', metric)
        if time_ms is not None:
            ax_broadband.plot(time_ms, values, 'o-', color=color, label=metric, linewidth=2, markersize=6)
    ax_broadband.set_xlabel('τ (ms)')
    ax_broadband.set_ylabel('bits')
    ax_broadband.set_title('B) BROADBAND (zoomed scale) - LRTC Structure')
    ax_broadband.legend(fontsize=8)
    ax_broadband.grid(alpha=0.3)
    ax_broadband.set_xscale('log')
    # Note: y-axis auto-scales to show broadband's small but meaningful values!
    
    # Panel C: Normalized Storage (all bands, each normalized by its max)
    ax_normalized = fig.add_subplot(gs[0, 2])
    for band_name in band_order:
        time_ms, values = get_band_data(band_name, 'Storage')
        if time_ms is not None and len(values) > 0:
            max_val = values.abs().max()
            if max_val > 0:
                normalized = values / max_val
                ax_normalized.plot(time_ms, normalized, 'o-', color=BAND_COLORS[band_name],
                                  label=band_name.upper(), linewidth=2)
    ax_normalized.set_xlabel('τ (ms)')
    ax_normalized.set_ylabel('Normalized Storage (max=1)')
    ax_normalized.set_title('C) Storage Normalized - All Bands Comparable')
    ax_normalized.legend(fontsize=8)
    ax_normalized.grid(alpha=0.3)
    ax_normalized.set_xscale('log')
    ax_normalized.set_ylim(-0.1, 1.1)
    
    # Panel D: Transfer across bands (narrowband)
    ax_transfer = fig.add_subplot(gs[1, 0])
    for band_name in narrowband_order:
        time_ms, values = get_band_data(band_name, 'Transfer')
        if time_ms is not None:
            ax_transfer.plot(time_ms, values, 'o-', color=BAND_COLORS[band_name],
                            label=band_name.upper(), linewidth=2)
    ax_transfer.set_xlabel('τ (ms)')
    ax_transfer.set_ylabel('Transfer (bits)')
    ax_transfer.set_title('D) Transfer - Narrowband')
    ax_transfer.legend(fontsize=8)
    ax_transfer.grid(alpha=0.3)
    ax_transfer.set_xscale('log')
    
    # Panel E: Integrated Info
    ax_phi = fig.add_subplot(gs[1, 1])
    for band_name in narrowband_order:
        time_ms, values = get_band_data(band_name, 'Integrated_info', source='iit')
        if time_ms is not None:
            ax_phi.plot(time_ms, values, 'o-', color=BAND_COLORS[band_name],
                       label=band_name.upper(), linewidth=2)
    ax_phi.set_xlabel('τ (ms)')
    ax_phi.set_ylabel('Φ (bits)')
    ax_phi.set_title('E) Integrated Information')
    ax_phi.legend(fontsize=8)
    ax_phi.grid(alpha=0.3)
    ax_phi.set_xscale('log')
    
    # Panel F: Downward vs Upward causation for alpha
    ax_causation = fig.add_subplot(gs[1, 2])
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
    
    # Panel G: Bars by band
    ax_bars = fig.add_subplot(gs[2, 0])
    
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
    
    # Panel H: Broadband vs Alpha at large τ
    ax_compare = fig.add_subplot(gs[2, 1])
    for band_name in ['broadband', 'alpha']:
        time_ms, storage = get_band_data(band_name, 'Storage')
        if time_ms is not None:
            # Focus on large τ where LRTC matters
            mask = time_ms >= 100  # Only τ >= 100ms
            ax_compare.plot(time_ms[mask], storage[mask], 'o-', 
                           color=BAND_COLORS[band_name],
                           label=f'{band_name.upper()} Storage', linewidth=2, markersize=6)
    ax_compare.set_xlabel('τ (ms)')
    ax_compare.set_ylabel('Storage (bits)')
    ax_compare.set_title('H) Large-τ Storage: Broadband vs Alpha')
    ax_compare.legend(fontsize=8)
    ax_compare.grid(alpha=0.3)
    ax_compare.set_xscale('log')
    
    # Panel I: Interpretation
    ax_interp = fig.add_subplot(gs[2, 2])
    ax_interp.axis('off')
    
    interpretation = """
    PhiID TEMPORAL ANALYSIS
    =======================
    
    WHY BROADBAND LOOKS SMALL:
    • Narrowband: 0.1-1.8 bits
    • Broadband: 0.001-0.01 bits
    → 100x difference in scale!
    
    IMPORTANT:
    • Panel B shows broadband on its OWN scale
    • Panel C normalizes all bands (max=1)
    
    BROADBAND STRUCTURE (LRTC):
    • Non-zero at large τ (seconds)
    • Reflects 1/f long-range correlations
    • Narrowband decorrelates quickly
    
    INTERPRETATION:
    • High Storage: Persistent dynamics
    • High Transfer: Active processing
    • High Φ: Integrated temporal structure
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
    # Extended to probe SLOW dynamics (attention, cognitive states, LRTC, etc.)
    # With 50000 samples at 300Hz = 166.7 seconds, max τ could be ~40 seconds (12000 samples)
    # But practical limit: need enough samples for statistics (at least 10x embedding span)
    TIMESCALES_MS = [
        # Fast dynamics (gamma/beta)
        10, 20, 30, 50, 75, 100,
        # Medium dynamics (alpha/theta)
        150, 200, 300, 500, 750, 1000,
        # Slow dynamics (delta, attention, cognitive states)
        1500, 2000, 3000, 5000,
        # Very slow dynamics / LRTC (requires long recordings!)
        7500, 10000, 15000, 20000,  # 7.5s, 10s, 15s, 20s
    ]
    ALL_TAU_VALUES = create_lag_schedule(fs, TIMESCALES_MS)
    
    # Verify signal length is sufficient for largest τ
    # Takens needs 3τ samples, plus buffer for statistical reliability
    max_tau = max(ALL_TAU_VALUES)
    min_samples_needed = 10 * max_tau  # Need ~10x embedding span for reliable estimates
    if SUBSAMPLE < min_samples_needed:
        print(f"  Note: SUBSAMPLE ({SUBSAMPLE}) limits max τ for reliable statistics")
        # Reduce τ range to what's statistically reliable
        safe_max_tau = SUBSAMPLE // 10
        ALL_TAU_VALUES = [t for t in ALL_TAU_VALUES if t <= safe_max_tau]
        print(f"        Limiting τ to max {safe_max_tau} samples ({safe_max_tau/fs*1000:.0f}ms = {safe_max_tau/fs:.1f}s)")
    
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
                if band_name == 'broadband':
                    filtered = highpass_filter(signal, fmin, fs)
                else:
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
                # Apply appropriate filter based on band type
                if band_name == 'broadband':
                    # For broadband: just highpass to remove DC drift, preserves LRTC!
                    filtered = highpass_filter(signal, fmin, fs)
                else:
                    # Standard bandpass for narrowband oscillations
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
    
    # 3. Master summary: all bands × key metrics (one big figure!)
    print("3. Master summary (all bands × metrics)...")
    plot_all_bands_summary(all_results, fs, save_path=RESULTS_DIR / "master_summary.png")
    
    # 4. Per-band consolidated plots (all channels in one figure per band)
    print("4. Per-band consolidated plots...")
    for band_name in ['broadband', 'delta', 'theta', 'alpha', 'beta', 'gamma']:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        print(f"   {band_name.upper()}...")
        
        # Dynamics: all channels, all metrics in one figure
        plot_band_all_channels(all_results[band_name], band_name, fs,
                              save_path=RESULTS_DIR / f"dynamics_{band_name}_all_channels.png")
        
        # Atoms heatmap: all channels side by side
        plot_atoms_heatmap_all_channels(all_results[band_name], band_name, fs,
                                        save_path=RESULTS_DIR / f"atoms_{band_name}_all_channels.png")
    
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
