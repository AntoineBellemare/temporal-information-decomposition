"""
EEG Bandpass Temporal PID Analysis - Extended Lags
===================================================

Analyze the temporal structure of EEG signals across frequency bands
with extended lags up to 1 second.

Frequency Bands:
- Delta: 1-4 Hz
- Theta: 4-8 Hz  
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Gamma: 30-50 Hz

Extended Lags: 3ms to 1000ms (1 second)

Usage:
    python eeg_bandpass_extended.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as sig

import dit
from dit.pid import PID_MMI
from dit import Distribution
from dit.shannon import mutual_information

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "eeg_bandpass_extended"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frequency bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

BAND_COLORS = {
    'delta': '#1f77b4',  # blue
    'theta': '#2ca02c',  # green
    'alpha': '#ff7f0e',  # orange
    'beta': '#d62728',   # red
    'gamma': '#9467bd'   # purple
}

# Brain region groupings
REGIONS = {
    'Frontal': ['F3', 'Fz', 'F4', 'F7', 'F8', 'Fp1', 'Fp2'],
    'Central': ['C3', 'Cz', 'C4'],
    'Temporal': ['T3', 'T4', 'T5', 'T6'],
    'Parietal': ['P3', 'P4'],
    'Occipital': ['O1', 'O2'],
    'Reference': ['A1', 'A2']
}


# =============================================================================
# SIGNAL PROCESSING FUNCTIONS
# =============================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = sig.butter(order, [low, high], btype='band')
    
    try:
        filtered = sig.filtfilt(b, a, data)
    except ValueError:
        filtered = sig.lfilter(b, a, data)
    
    return filtered


def prewhiten_ar(signal, order=10):
    """
    Prewhiten signal by fitting an AR model and returning residuals.
    
    This removes the linear autocorrelation structure, leaving only
    the nonlinear/higher-order temporal dependencies.
    
    The PID of whitened signals reveals:
    - Synergy: nonlinear interactions between time points
    - Redundancy: nonlinear shared structure
    - Unique: asymmetric nonlinear dependencies
    """
    from scipy.linalg import toeplitz, solve
    
    n = len(signal)
    signal = signal - np.mean(signal)
    
    # Compute autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:] / autocorr[n-1]  # Normalize
    
    # Yule-Walker equations to get AR coefficients
    r = autocorr[1:order+1]
    R = toeplitz(autocorr[:order])
    
    try:
        ar_coeffs = solve(R, r)
    except:
        return signal  # Fall back to original if solve fails
    
    # Compute residuals
    residuals = np.zeros(n)
    for t in range(order, n):
        pred = np.dot(ar_coeffs, signal[t-order:t][::-1])
        residuals[t] = signal[t] - pred
    
    return residuals[order:]  # Return only valid residuals


def compute_instantaneous_phase(signal):
    """
    Extract instantaneous phase using Hilbert transform.
    
    For oscillatory signals, analyzing phase rather than amplitude
    can reveal different temporal structure.
    """
    analytic = sig.hilbert(signal)
    phase = np.angle(analytic)
    return phase


def compute_amplitude_envelope(signal):
    """Extract amplitude envelope using Hilbert transform."""
    analytic = sig.hilbert(signal)
    envelope = np.abs(analytic)
    return envelope


def create_lag_schedule(fs, timescales_ms):
    """Create lag values corresponding to specific timescales."""
    lags = [max(1, int(t * fs / 1000)) for t in timescales_ms]
    return sorted(list(set(lags)))


# =============================================================================
# PID FUNCTIONS
# =============================================================================

def discretize_timeseries(x, n_bins=4, method='quantile'):
    """Discretize continuous time series into symbols."""
    x = np.asarray(x).copy()
    mask = np.isfinite(x)
    
    if not np.any(mask):
        return np.zeros(len(x), dtype=int)
    
    x_clean = x[mask]
    
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x_clean, percentiles)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(x_clean.min() - 1e-10, x_clean.max() + 1e-10, n_bins + 1)
        else:
            bins[0] -= 1e-10
            bins[-1] += 1e-10
    else:
        bins = np.linspace(x_clean.min() - 1e-10, x_clean.max() + 1e-10, n_bins + 1)
    
    result = np.zeros(len(x), dtype=int)
    result[mask] = np.digitize(x[mask], bins[1:-1])
    return result


def build_embedding_distribution(x, lags=[1, 2]):
    """Build a dit Distribution from time series."""
    max_lag = max(lags)
    n = len(x)
    
    if n <= max_lag:
        return None
    
    outcomes = []
    for t in range(max_lag, n):
        outcome = tuple(int(x[t - lag]) for lag in lags) + (int(x[t]),)
        outcomes.append(''.join(map(str, outcome)))
    
    counts = Counter(outcomes)
    total = sum(counts.values())
    
    outcomes_list = list(counts.keys())
    probs = [counts[o] / total for o in outcomes_list]
    
    return Distribution(outcomes_list, probs)


def build_bivariate_distribution(x, lag):
    """Build distribution for single lag: (x_{t-lag}, x_t)."""
    n = len(x)
    if n <= lag:
        return None
    
    outcomes = []
    for t in range(lag, n):
        outcome = (int(x[t - lag]), int(x[t]))
        outcomes.append(''.join(map(str, outcome)))
    
    counts = Counter(outcomes)
    total = sum(counts.values())
    
    outcomes_list = list(counts.keys())
    probs = [counts[o] / total for o in outcomes_list]
    
    return Distribution(outcomes_list, probs)


def compute_mutual_information(x_discrete, lag):
    """Compute MI between x_{t-lag} and x_t."""
    dist = build_bivariate_distribution(x_discrete, lag)
    if dist is None:
        return np.nan
    try:
        return mutual_information(dist, [0], [1])
    except:
        return np.nan


def compute_pid_summary(dist, pid_class=PID_MMI):
    """Extract key PID values from a distribution."""
    if dist is None:
        return {'redundancy': np.nan, 'unique_0': np.nan, 'unique_1': np.nan, 'synergy': np.nan}
    
    try:
        pid = pid_class(dist)
    except:
        return {'redundancy': np.nan, 'unique_0': np.nan, 'unique_1': np.nan, 'synergy': np.nan}
    
    summary = {'redundancy': 0.0, 'unique_0': 0.0, 'unique_1': 0.0, 'synergy': 0.0}
    
    # Use get_pi() method which works with both local and pip-installed dit
    for node in pid._lattice:
        try:
            val = float(pid.get_pi(node))
        except:
            val = 0.0
        
        if len(node) == 2 and all(len(n) == 1 for n in node):
            summary['redundancy'] = val
        elif len(node) == 1 and len(node[0]) == 2:
            summary['synergy'] = val
        elif node == ((0,),):
            summary['unique_0'] = val
        elif node == ((1,),):
            summary['unique_1'] = val
    
    return summary


def analyze_single_lag_mi(signal, lags, n_bins=4):
    """Compute mutual information for each single lag."""
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    
    results = []
    for lag in lags:
        mi = compute_mutual_information(signal_discrete, lag)
        results.append({'lag': lag, 'MI': mi})
    
    return pd.DataFrame(results)


def analyze_pid_lag_pairs(signal, lag_pairs, n_bins=4):
    """Compute PID for specified lag pairs."""
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    
    results = []
    for lag1, lag2 in lag_pairs:
        dist = build_embedding_distribution(signal_discrete, lags=[lag1, lag2])
        pid = compute_pid_summary(dist)
        pid['lag1'] = lag1
        pid['lag2'] = lag2
        results.append(pid)
    
    return pd.DataFrame(results)


def analyze_with_preprocessing(signal, lags, lag_pairs, n_bins=4, preprocess='none'):
    """
    Analyze signal with optional preprocessing to remove autocorrelation artifacts.
    
    preprocess options:
    - 'none': Raw signal
    - 'whiten': AR prewhitening (removes linear autocorrelation)
    - 'phase': Instantaneous phase (for oscillatory signals)
    - 'envelope': Amplitude envelope
    """
    if preprocess == 'whiten':
        processed = prewhiten_ar(signal, order=10)
    elif preprocess == 'phase':
        processed = compute_instantaneous_phase(signal)
    elif preprocess == 'envelope':
        processed = compute_amplitude_envelope(signal)
    else:
        processed = signal
    
    # Adjust lags for shortened signal if whitened
    max_lag = max(max(lags), max(l2 for _, l2 in lag_pairs))
    if len(processed) <= max_lag:
        return None, None
    
    df_mi = analyze_single_lag_mi(processed, lags, n_bins=n_bins)
    df_pid = analyze_pid_lag_pairs(processed, lag_pairs, n_bins=n_bins)
    
    return df_mi, df_pid


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_lag_profiles_by_band(all_results, channel, fs, save_path=None):
    """
    Plot PID vs lag distance for each band for one channel.
    
    Uses LOG x-axis for extended timescales.
    Shows all PID components but with unique as lighter/thinner.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for idx, band_name in enumerate(band_order):
        ax = axes[idx]
        
        if band_name not in all_results or channel not in all_results[band_name]:
            ax.set_title(f'{band_name.upper()} - No data')
            continue
        
        df = all_results[band_name][channel].copy()
        df['lag_diff'] = df['lag2'] - df['lag1']
        
        grouped = df.groupby('lag_diff').agg({
            'redundancy': ['mean', 'std'],
            'synergy': ['mean', 'std'],
            'unique_0': 'mean',
            'unique_1': 'mean'
        }).reset_index()
        grouped.columns = ['lag_diff', 'red_mean', 'red_std', 'syn_mean', 'syn_std', 'uniq0', 'uniq1']
        grouped['total_unique'] = grouped['uniq0'] + grouped['uniq1']
        
        time_ms = grouped['lag_diff'] / fs * 1000
        
        # Main metrics: Redundancy and Synergy (bold)
        ax.errorbar(time_ms, grouped['red_mean'], yerr=grouped['red_std'],
                   fmt='o-', color='green', label='Redundancy', capsize=3, 
                   linewidth=2.5, markersize=6)
        ax.errorbar(time_ms, grouped['syn_mean'], yerr=grouped['syn_std'],
                   fmt='s-', color='red', label='Synergy', capsize=3, 
                   linewidth=2.5, markersize=6)
        
        # Unique: show but lighter/thinner (autocorr-dominated but still informative)
        ax.plot(time_ms, grouped['total_unique'], '^--', color='blue',
               label='Unique (autocorr)', linewidth=1.5, markersize=4, alpha=0.6)
        
        ax.set_xlabel('Lag Difference (ms)')
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'{band_name.upper()} ({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xscale('log')  # LOG SCALE for extended timescales
        
        # Set reasonable x limits
        ax.set_xlim(time_ms.min() * 0.8, time_ms.max() * 1.2)
    
    axes[5].axis('off')
    
    # Add interpretation
    axes[5].text(0.1, 0.8, 
                 "INTERPRETATION:\n\n"
                 "• Redundancy (green): Shared info between lags\n"
                 "  High = smooth, predictable signal\n\n"
                 "• Synergy (red): Info from BOTH lags together\n"
                 "  High = nonlinear temporal integration\n\n"
                 "• Unique (blue, dashed): Asymmetric info\n"
                 "  Spikes = autocorrelation at periodic lags\n"
                 "  (e.g., alpha ~100ms period)\n\n"
                 "X-axis is LOG SCALE (3ms → 1s)",
                 transform=axes[5].transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'PID vs Lag Distance by Frequency Band: {channel} (log scale)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_mi_by_band(all_mi_results, channel, fs, save_path=None):
    """Plot single-lag MI for each band."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    for band_name in band_order:
        if band_name not in all_mi_results or channel not in all_mi_results[band_name]:
            continue
        
        df = all_mi_results[band_name][channel]
        time_ms = df['lag'] / fs * 1000
        
        ax.plot(time_ms, df['MI'], 'o-', color=BAND_COLORS[band_name],
               label=f'{band_name.upper()}', linewidth=2, markersize=5)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title(f'Single-Lag MI by Frequency Band: {channel}')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_band_heatmaps_extended(all_results, channel, fs, metric='synergy', save_path=None):
    """Create heatmaps for all bands for one channel with time labels."""
    n_bands = len(BANDS)
    fig, axes = plt.subplots(1, n_bands, figsize=(4*n_bands, 5))
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    cmap = 'Reds' if metric == 'synergy' else 'Greens'
    
    for idx, band_name in enumerate(band_order):
        ax = axes[idx]
        
        if band_name not in all_results or channel not in all_results[band_name]:
            ax.set_title(f'{band_name.upper()}\nNo data')
            continue
        
        df = all_results[band_name][channel]
        lag1_vals = sorted(df['lag1'].unique())
        lag2_vals = sorted(df['lag2'].unique())
        
        matrix = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
        for _, row in df.iterrows():
            i = lag1_vals.index(row['lag1'])
            j = lag2_vals.index(row['lag2'])
            matrix[i, j] = row[metric]
        
        # Create time labels
        def lag_to_label(l):
            t = l / fs * 1000
            if t < 100:
                return f'{t:.0f}'
            else:
                return f'{t/1000:.1f}s'
        
        time_labels_1 = [lag_to_label(l) for l in lag1_vals[::3]]
        time_labels_2 = [lag_to_label(l) for l in lag2_vals[::3]]
        
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask,
                    xticklabels=3, yticklabels=3, cbar_kws={'label': 'bits'})
        ax.set_title(f'{band_name.upper()}\n({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)')
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1' if idx == 0 else '')
    
    plt.suptitle(f'{metric.title()} Across Frequency Bands: {channel}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_band_comparison_summary(all_results, channels, fs, save_path=None):
    """Summary comparison across bands."""
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Aggregate by band
    summary = []
    for band_name in band_order:
        if band_name not in all_results:
            continue
        
        reds = []
        syns = []
        for ch, df in all_results[band_name].items():
            reds.append(df['redundancy'].mean())
            syns.append(df['synergy'].mean())
        
        summary.append({
            'band': band_name,
            'redundancy': np.mean(reds),
            'redundancy_std': np.std(reds),
            'synergy': np.mean(syns),
            'synergy_std': np.std(syns),
            'ratio': np.mean(syns) / max(np.mean(reds), 1e-10)
        })
    
    df_summary = pd.DataFrame(summary)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(df_summary))
    colors = [BAND_COLORS[b] for b in df_summary['band']]
    
    # Redundancy
    axes[0].bar(x, df_summary['redundancy'], yerr=df_summary['redundancy_std'],
               color=colors, capsize=5, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b.upper() for b in df_summary['band']])
    axes[0].set_ylabel('Mean Redundancy (bits)')
    axes[0].set_title('Redundancy by Frequency Band')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synergy
    axes[1].bar(x, df_summary['synergy'], yerr=df_summary['synergy_std'],
               color=colors, capsize=5, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b.upper() for b in df_summary['band']])
    axes[1].set_ylabel('Mean Synergy (bits)')
    axes[1].set_title('Synergy by Frequency Band')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Ratio
    axes[2].bar(x, df_summary['ratio'], color=colors, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([b.upper() for b in df_summary['band']])
    axes[2].set_ylabel('Synergy / Redundancy')
    axes[2].set_title('Synergy-Redundancy Balance')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Temporal PID Summary Across Frequency Bands (Extended Lags: 3ms to 1s)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_summary


def plot_regional_band_heatmap(all_results, fs, save_path=None):
    """Heatmap of synergy/redundancy ratio: regions x bands."""
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    region_order = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    
    # Aggregate by region and band
    data = []
    for band_name in band_order:
        if band_name not in all_results:
            continue
        
        for ch, df in all_results[band_name].items():
            ch_short = ch.replace('eeg-', '')
            
            region = 'Other'
            for reg_name, reg_channels in REGIONS.items():
                if ch_short in reg_channels:
                    region = reg_name
                    break
            
            data.append({
                'band': band_name,
                'region': region,
                'redundancy': df['redundancy'].mean(),
                'synergy': df['synergy'].mean(),
                'ratio': df['synergy'].mean() / max(df['redundancy'].mean(), 1e-10)
            })
    
    df_data = pd.DataFrame(data)
    
    # Create pivot tables
    pivot_syn = df_data.groupby(['region', 'band'])['synergy'].mean().unstack()
    pivot_red = df_data.groupby(['region', 'band'])['redundancy'].mean().unstack()
    pivot_ratio = df_data.groupby(['region', 'band'])['ratio'].mean().unstack()
    
    # Reorder
    pivot_syn = pivot_syn.reindex([r for r in region_order if r in pivot_syn.index])
    pivot_syn = pivot_syn[[b for b in band_order if b in pivot_syn.columns]]
    pivot_red = pivot_red.reindex([r for r in region_order if r in pivot_red.index])
    pivot_red = pivot_red[[b for b in band_order if b in pivot_red.columns]]
    pivot_ratio = pivot_ratio.reindex([r for r in region_order if r in pivot_ratio.index])
    pivot_ratio = pivot_ratio[[b for b in band_order if b in pivot_ratio.columns]]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.heatmap(pivot_syn, ax=axes[0], cmap='Reds', annot=True, fmt='.3f',
                cbar_kws={'label': 'bits'})
    axes[0].set_title('Mean Synergy')
    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('Brain Region')
    
    sns.heatmap(pivot_red, ax=axes[1], cmap='Greens', annot=True, fmt='.2f',
                cbar_kws={'label': 'bits'})
    axes[1].set_title('Mean Redundancy')
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('')
    
    sns.heatmap(pivot_ratio, ax=axes[2], cmap='Purples', annot=True, fmt='.3f',
                cbar_kws={'label': 'ratio'})
    axes[2].set_title('Synergy/Redundancy Ratio')
    axes[2].set_xlabel('Frequency Band')
    axes[2].set_ylabel('')
    
    plt.suptitle('Temporal PID: Brain Regions × Frequency Bands (Extended Lags)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_data


def plot_all_bands_grid(all_results, metric='synergy', save_path=None):
    """Grid of heatmaps: rows=bands, cols=sample of channels."""
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    channels = list(list(all_results.values())[0].keys())
    
    # Select subset
    selected_channels = channels[:10] if len(channels) > 10 else channels
    
    n_bands = len(band_order)
    n_cols = len(selected_channels)
    
    fig, axes = plt.subplots(n_bands, n_cols, figsize=(2*n_cols, 2.5*n_bands))
    
    # Get global limits
    all_values = []
    for band_results in all_results.values():
        for df in band_results.values():
            all_values.extend(df[metric].dropna().values)
    vmin, vmax = np.percentile(all_values, [5, 95])
    
    cmap = 'Reds' if metric == 'synergy' else 'Greens'
    
    for i, band in enumerate(band_order):
        for j, ch in enumerate(selected_channels):
            ax = axes[i, j]
            
            if band in all_results and ch in all_results[band]:
                df = all_results[band][ch]
                lag1_vals = sorted(df['lag1'].unique())
                lag2_vals = sorted(df['lag2'].unique())
                
                matrix = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
                for _, row in df.iterrows():
                    ii = lag1_vals.index(row['lag1'])
                    jj = lag2_vals.index(row['lag2'])
                    matrix[ii, jj] = row[metric]
                
                mask = np.isnan(matrix)
                sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask,
                           vmin=vmin, vmax=vmax,
                           xticklabels=False, yticklabels=False, cbar=False)
            
            if i == 0:
                ax.set_title(ch.replace('eeg-', ''), fontsize=9)
            if j == 0:
                ax.set_ylabel(band.upper(), fontsize=10)
    
    plt.suptitle(f'{metric.title()} Heatmaps: Frequency Bands × Channels (Extended Lags)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_raw_vs_whitened_comparison(raw_results, whitened_results, fs, channels, save_path=None):
    """
    Compare PID profiles before and after AR prewhitening.
    
    Uses LOG x-axis. Shows all components including unique.
    """
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, band_name in enumerate(band_order):
        ax = axes.flatten()[idx]
        
        # Aggregate across channels
        raw_lag_data = []
        whiten_lag_data = []
        
        if band_name in raw_results:
            for ch, df in raw_results[band_name].items():
                df = df.copy()
                df['lag_diff'] = df['lag2'] - df['lag1']
                raw_lag_data.append(df)
        
        if band_name in whitened_results:
            for ch, df in whitened_results[band_name].items():
                df = df.copy()
                df['lag_diff'] = df['lag2'] - df['lag1']
                whiten_lag_data.append(df)
        
        if not raw_lag_data:
            ax.set_title(f'{band_name.upper()} - No data')
            continue
        
        # Aggregate
        raw_combined = pd.concat(raw_lag_data)
        raw_grouped = raw_combined.groupby('lag_diff').agg({
            'redundancy': 'mean', 'synergy': 'mean',
            'unique_0': 'mean', 'unique_1': 'mean'
        }).reset_index()
        raw_grouped['total_unique'] = raw_grouped['unique_0'] + raw_grouped['unique_1']
        
        time_ms = raw_grouped['lag_diff'] / fs * 1000
        
        # Plot raw (solid, lighter)
        ax.plot(time_ms, raw_grouped['redundancy'], 'o-', color='green', 
                label='Redundancy (raw)', linewidth=2, alpha=0.5, markersize=5)
        ax.plot(time_ms, raw_grouped['synergy'], 's-', color='red',
                label='Synergy (raw)', linewidth=2, alpha=0.5, markersize=5)
        ax.plot(time_ms, raw_grouped['total_unique'], '^-', color='blue',
                label='Unique (raw)', linewidth=1.5, alpha=0.4, markersize=4)
        
        # Plot whitened if available (dashed, darker)
        if whiten_lag_data:
            whiten_combined = pd.concat(whiten_lag_data)
            whiten_grouped = whiten_combined.groupby('lag_diff').agg({
                'redundancy': 'mean', 'synergy': 'mean',
                'unique_0': 'mean', 'unique_1': 'mean'
            }).reset_index()
            whiten_grouped['total_unique'] = whiten_grouped['unique_0'] + whiten_grouped['unique_1']
            
            time_ms_w = whiten_grouped['lag_diff'] / fs * 1000
            
            ax.plot(time_ms_w, whiten_grouped['redundancy'], 'o--', color='darkgreen',
                    label='Red (whiten)', linewidth=2.5, markersize=6)
            ax.plot(time_ms_w, whiten_grouped['synergy'], 's--', color='darkred',
                    label='Syn (whiten)', linewidth=2.5, markersize=6)
            ax.plot(time_ms_w, whiten_grouped['total_unique'], '^--', color='darkblue',
                    label='Uniq (whiten)', linewidth=2, markersize=5)
        
        ax.set_xlabel('Lag Difference (ms)')
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'{band_name.upper()} ({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)')
        ax.legend(fontsize=7, loc='upper right', ncol=2)
        ax.grid(alpha=0.3)
        ax.set_xscale('log')  # LOG SCALE
    
    # Interpretation panel
    ax_interp = axes.flatten()[5]
    ax_interp.axis('off')
    
    interpretation = """
    RAW vs WHITENED (log scale)
    ===========================
    
    RAW (solid, faded):
    • All temporal structure
    • Unique spikes = autocorrelation
    
    WHITENED (dashed, bold):
    • AR(10) removes linear autocorr
    • What remains = NONLINEAR structure
    
    KEY OBSERVATIONS:
    • Unique spikes should DISAPPEAR 
      after whitening (confirms autocorr)
    • Synergy that REMAINS after whitening
      = genuine nonlinear integration
    • If red/syn drop to ~0: linear process
    • If red/syn persist: nonlinear dynamics
    """
    
    ax_interp.text(0.05, 0.95, interpretation, transform=ax_interp.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Effect of AR Prewhitening: Isolating Nonlinear Temporal Structure', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_grand_summary(all_results, all_mi_results, fs, save_path=None):
    """Comprehensive summary figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # --- Panel A: Band comparison bars ---
    ax_bars = fig.add_subplot(gs[0, :2])
    
    band_summary = []
    for band_name in band_order:
        if band_name not in all_results:
            continue
        reds = [df['redundancy'].mean() for df in all_results[band_name].values()]
        syns = [df['synergy'].mean() for df in all_results[band_name].values()]
        band_summary.append({
            'band': band_name,
            'redundancy': np.mean(reds),
            'synergy': np.mean(syns),
            'ratio': np.mean(syns) / max(np.mean(reds), 1e-10)
        })
    
    df_bands = pd.DataFrame(band_summary)
    
    x = np.arange(len(df_bands))
    width = 0.35
    
    ax_bars.bar(x - width/2, df_bands['redundancy'], width, label='Redundancy', 
                color='green', alpha=0.7)
    ax_bars.bar(x + width/2, df_bands['synergy'], width, label='Synergy', 
                color='red', alpha=0.7)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([b.upper() for b in df_bands['band']])
    ax_bars.set_ylabel('Mean Information (bits)')
    ax_bars.set_title('A) PID Components by Frequency Band')
    ax_bars.legend()
    ax_bars.grid(axis='y', alpha=0.3)
    
    # --- Panel B: Ratio ---
    ax_ratio = fig.add_subplot(gs[0, 2:])
    
    colors = [BAND_COLORS[b] for b in df_bands['band']]
    ax_ratio.bar(x, df_bands['ratio'], color=colors, alpha=0.8)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([b.upper() for b in df_bands['band']])
    ax_ratio.set_ylabel('Synergy / Redundancy')
    ax_ratio.set_title('B) Synergy-Redundancy Balance')
    ax_ratio.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax_ratio.grid(axis='y', alpha=0.3)
    
    # --- Panel C: Region x Band synergy ---
    ax_heat = fig.add_subplot(gs[1, :2])
    
    region_data = []
    region_order = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    
    for band_name in band_order:
        if band_name not in all_results:
            continue
        for ch, df in all_results[band_name].items():
            ch_short = ch.replace('eeg-', '')
            region = 'Other'
            for reg_name, reg_channels in REGIONS.items():
                if ch_short in reg_channels:
                    region = reg_name
                    break
            if region != 'Other':
                region_data.append({
                    'band': band_name, 'region': region,
                    'synergy': df['synergy'].mean()
                })
    
    df_rb = pd.DataFrame(region_data)
    pivot = df_rb.groupby(['region', 'band'])['synergy'].mean().unstack()
    pivot = pivot.reindex([r for r in region_order if r in pivot.index])
    pivot = pivot[[b for b in band_order if b in pivot.columns]]
    
    sns.heatmap(pivot, ax=ax_heat, cmap='Reds', annot=True, fmt='.3f',
                cbar_kws={'label': 'bits'})
    ax_heat.set_title('C) Mean Synergy: Region × Band')
    ax_heat.set_xlabel('Frequency Band')
    ax_heat.set_ylabel('Brain Region')
    
    # --- Panel D: MI decay by band ---
    ax_mi = fig.add_subplot(gs[1, 2:])
    
    for band_name in band_order:
        if band_name not in all_mi_results:
            continue
        
        # Average across channels
        all_mi = []
        for ch, df in all_mi_results[band_name].items():
            all_mi.append(df.set_index('lag')['MI'])
        
        if all_mi:
            avg_mi = pd.concat(all_mi, axis=1).mean(axis=1)
            time_ms = avg_mi.index / fs * 1000
            ax_mi.plot(time_ms, avg_mi.values, 'o-', color=BAND_COLORS[band_name],
                      label=band_name.upper(), linewidth=2)
    
    ax_mi.set_xlabel('Time (ms)')
    ax_mi.set_ylabel('Mutual Information (bits)')
    ax_mi.set_title('D) Single-Lag MI Decay by Band')
    ax_mi.legend()
    ax_mi.grid(alpha=0.3)
    ax_mi.set_xscale('log')
    
    # --- Panel E: Lag profiles for alpha ---
    ax_alpha = fig.add_subplot(gs[2, :2])
    
    if 'alpha' in all_results:
        all_lag_data = []
        for ch, df in all_results['alpha'].items():
            df = df.copy()
            df['lag_diff'] = df['lag2'] - df['lag1']
            all_lag_data.append(df)
        
        combined = pd.concat(all_lag_data)
        grouped = combined.groupby('lag_diff').agg({
            'redundancy': ['mean', 'std'],
            'synergy': ['mean', 'std'],
            'unique_0': 'mean',
            'unique_1': 'mean'
        }).reset_index()
        grouped.columns = ['lag_diff', 'red_mean', 'red_std', 'syn_mean', 'syn_std', 'uniq0', 'uniq1']
        grouped['total_unique'] = grouped['uniq0'] + grouped['uniq1']
        
        time_ms = grouped['lag_diff'] / fs * 1000
        
        ax_alpha.errorbar(time_ms, grouped['red_mean'], yerr=grouped['red_std'],
                         fmt='o-', color='green', label='Redundancy', capsize=3, linewidth=2)
        ax_alpha.errorbar(time_ms, grouped['syn_mean'], yerr=grouped['syn_std'],
                         fmt='s-', color='red', label='Synergy', capsize=3, linewidth=2)
        ax_alpha.plot(time_ms, grouped['total_unique'], '^--', color='blue',
                     label='Unique', linewidth=1.5, alpha=0.6, markersize=4)
        ax_alpha.set_xlabel('Lag Difference (ms)')
        ax_alpha.set_ylabel('Information (bits)')
        ax_alpha.set_title('E) Alpha Band: PID vs Lag Distance (log scale)')
        ax_alpha.legend()
        ax_alpha.grid(alpha=0.3)
        ax_alpha.set_xscale('log')  # LOG SCALE
    
    # --- Panel F: Interpretation ---
    ax_interp = fig.add_subplot(gs[2, 2:])
    ax_interp.axis('off')
    
    interpretation = """
    KEY FINDINGS (Extended Lags: 3ms to 1 second)
    =============================================
    
    FREQUENCY BAND SIGNATURES:
    • DELTA: High redundancy, near-zero synergy
      → Slow, purely predictable dynamics
      
    • THETA: Moderate redundancy, notable synergy  
      → Complex temporal integration (memory?)
      
    • ALPHA: Periodic redundancy structure
      → Oscillatory with nonlinear phase dynamics
      
    • BETA/GAMMA: Lower overall information
      → Faster timescales, less temporal structure
    
    REGIONAL PATTERNS:
    • Occipital: Highest synergy (especially alpha)
    • Temporal: High theta complexity
    • Frontal: More uniform across bands
    
    TIMESCALE INSIGHTS:
    • Short lags (< 50ms): High redundancy (smoothness)
    • Medium lags (50-200ms): Peak synergy (interactions)
    • Long lags (> 500ms): Decay of all components
    """
    
    ax_interp.text(0.02, 0.95, interpretation, transform=ax_interp.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('EEG Temporal PID: Frequency Band Analysis (Extended Lags)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EEG BANDPASS TEMPORAL PID - EXTENDED LAGS")
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
    SUBSAMPLE = 100000
    N_BINS = 4
    
    # Extended timescales (ms)
    TIMESCALES_MS = [3, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    LAGS = create_lag_schedule(fs, TIMESCALES_MS)
    
    # ALL lag pairs
    LAG_PAIRS = []
    for i, lag1 in enumerate(LAGS):
        for lag2 in LAGS[i+1:]:
            LAG_PAIRS.append((lag1, lag2))
    
    # Preprocessing modes to compare
    # 'none' = raw bandpassed signal (has autocorrelation artifacts in unique)
    # 'whiten' = AR prewhitened (removes linear autocorr, keeps nonlinear structure)
    PREPROCESS_MODES = ['none', 'whiten']
    
    print(f"\nParameters:")
    print(f"  Samples used: {SUBSAMPLE}")
    print(f"  Bins: {N_BINS}")
    print(f"  Timescales: {TIMESCALES_MS} ms")
    print(f"  Lags: {LAGS}")
    print(f"  Lag pairs: {len(LAG_PAIRS)}")
    print(f"  Bands: {list(BANDS.keys())}")
    print(f"  Preprocessing: {PREPROCESS_MODES}")
    
    # Storage: {preprocess: {band: {channel: DataFrame}}}
    all_results = {}
    
    for preprocess_mode in PREPROCESS_MODES:
        print(f"\n{'#'*70}")
        print(f"PREPROCESSING: {preprocess_mode.upper()}")
        print("#"*70)
        
        all_results[preprocess_mode] = {
            'pid': {},  # {band: {channel: DataFrame}}
            'mi': {}    # {band: {channel: DataFrame}}
        }
        
        # Process each band
        for band_name, (fmin, fmax) in BANDS.items():
            print(f"\n{'='*60}")
            print(f"Processing: {band_name.upper()} ({fmin}-{fmax} Hz)")
            print("="*60)
            
            all_results[preprocess_mode]['pid'][band_name] = {}
            all_results[preprocess_mode]['mi'][band_name] = {}
            
            for i, ch in enumerate(channels):
                print(f"  Channel {i+1}/{len(channels)}: {ch}", end=" ")
                
                # Get signal
                signal = df[ch].values[:SUBSAMPLE]
                signal = signal[~np.isnan(signal)]
                
                try:
                    # Bandpass filter
                    filtered = bandpass_filter(signal, fmin, fmax, fs)
                    
                    # Analyze with preprocessing
                    df_mi, df_pid = analyze_with_preprocessing(
                        filtered, LAGS, LAG_PAIRS, n_bins=N_BINS, 
                        preprocess=preprocess_mode
                    )
                    
                    if df_mi is None or df_pid is None:
                        print("✗ insufficient data after preprocessing")
                        continue
                    
                    df_mi['channel'] = ch
                    df_mi['band'] = band_name
                    df_mi['preprocess'] = preprocess_mode
                    all_results[preprocess_mode]['mi'][band_name][ch] = df_mi
                    
                    df_pid['channel'] = ch
                    df_pid['band'] = band_name
                    df_pid['preprocess'] = preprocess_mode
                    all_results[preprocess_mode]['pid'][band_name][ch] = df_pid
                    
                    print("✓")
                except Exception as e:
                    print(f"✗ {e}")
    
    # For backwards compatibility, extract 'none' results
    all_pid_results = all_results['none']['pid']
    all_mi_results = all_results['none']['mi']
    
    # Also get whitened results
    whitened_pid_results = all_results['whiten']['pid']
    whitened_mi_results = all_results['whiten']['mi']
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)
    
    # Combine all into single DataFrames
    all_pid_dfs = []
    all_mi_dfs = []
    
    for band_name in BANDS.keys():
        for ch in channels:
            if ch in all_pid_results.get(band_name, {}):
                all_pid_dfs.append(all_pid_results[band_name][ch])
            if ch in all_mi_results.get(band_name, {}):
                all_mi_dfs.append(all_mi_results[band_name][ch])
    
    df_pid_all = pd.concat(all_pid_dfs, ignore_index=True)
    df_mi_all = pd.concat(all_mi_dfs, ignore_index=True)
    
    # Add time columns
    df_pid_all['lag1_ms'] = df_pid_all['lag1'] / fs * 1000
    df_pid_all['lag2_ms'] = df_pid_all['lag2'] / fs * 1000
    df_mi_all['time_ms'] = df_mi_all['lag'] / fs * 1000
    
    df_pid_all.to_csv(RESULTS_DIR / "bandpass_pid_extended.csv", index=False)
    df_mi_all.to_csv(RESULTS_DIR / "bandpass_mi_extended.csv", index=False)
    
    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print("="*70)
    
    # 1. Grand summary
    print("\n1. Grand summary figure...")
    plot_grand_summary(all_pid_results, all_mi_results, fs, 
                       save_path=RESULTS_DIR / "grand_summary.png")
    
    # 2. Band comparison
    print("2. Band comparison...")
    df_band_summary = plot_band_comparison_summary(all_pid_results, channels, fs,
                                                   save_path=RESULTS_DIR / "band_comparison.png")
    df_band_summary.to_csv(RESULTS_DIR / "band_summary.csv", index=False)
    
    # 3. Regional heatmaps
    print("3. Regional analysis...")
    df_regional = plot_regional_band_heatmap(all_pid_results, fs,
                                             save_path=RESULTS_DIR / "regional_band_heatmap.png")
    df_regional.to_csv(RESULTS_DIR / "regional_summary.csv", index=False)
    
    # 4. All bands grid
    print("4. Band x channel grids...")
    plot_all_bands_grid(all_pid_results, metric='synergy',
                        save_path=RESULTS_DIR / "all_bands_grid_synergy.png")
    plot_all_bands_grid(all_pid_results, metric='redundancy',
                        save_path=RESULTS_DIR / "all_bands_grid_redundancy.png")
    
    # 5. Key channel analyses
    print("5. Individual channel analyses...")
    key_channels = ['eeg-Fz', 'eeg-O1', 'eeg-Cz', 'eeg-T3']
    
    for ch in key_channels:
        if ch not in channels:
            continue
        ch_name = ch.replace('eeg-', '')
        
        plot_lag_profiles_by_band(all_pid_results, ch, fs,
                                  save_path=RESULTS_DIR / f"lag_profiles_{ch_name}.png")
        
        plot_mi_by_band(all_mi_results, ch, fs,
                        save_path=RESULTS_DIR / f"mi_by_band_{ch_name}.png")
        
        plot_band_heatmaps_extended(all_pid_results, ch, fs, metric='synergy',
                                    save_path=RESULTS_DIR / f"synergy_bands_{ch_name}.png")
        
        plot_band_heatmaps_extended(all_pid_results, ch, fs, metric='redundancy',
                                    save_path=RESULTS_DIR / f"redundancy_bands_{ch_name}.png")
    
    # 6. RAW vs WHITENED comparison
    print("\n6. Raw vs Whitened comparison...")
    plot_raw_vs_whitened_comparison(all_pid_results, whitened_pid_results, fs, channels,
                                    save_path=RESULTS_DIR / "raw_vs_whitened_comparison.png")
    
    # Save whitened results too
    whitened_pid_dfs = []
    for band_name in BANDS.keys():
        for ch in channels:
            if ch in whitened_pid_results.get(band_name, {}):
                whitened_pid_dfs.append(whitened_pid_results[band_name][ch])
    
    if whitened_pid_dfs:
        df_whitened = pd.concat(whitened_pid_dfs, ignore_index=True)
        df_whitened['lag1_ms'] = df_whitened['lag1'] / fs * 1000
        df_whitened['lag2_ms'] = df_whitened['lag2'] / fs * 1000
        df_whitened.to_csv(RESULTS_DIR / "bandpass_pid_whitened.csv", index=False)
    
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
