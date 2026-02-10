"""
EEG Bandpass Temporal PID Analysis
===================================

Analyze the temporal structure of EEG signals across frequency bands
using PID on time-delay embeddings.

Frequency Bands:
- Delta: 1-4 Hz
- Theta: 4-8 Hz  
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Gamma: 30-50 Hz

For each band and each channel:
- Bandpass filter the signal
- Discretize and compute PID for all lag pairs
- Compare temporal fingerprints across bands and regions

Usage:
    python eeg_bandpass_temporal_pid.py
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

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "eeg_bandpass"
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
    
    # Ensure frequencies are valid
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = sig.butter(order, [low, high], btype='band')
    
    # Use filtfilt for zero-phase filtering
    try:
        filtered = sig.filtfilt(b, a, data)
    except ValueError:
        # If signal is too short, use lfilter instead
        filtered = sig.lfilter(b, a, data)
    
    return filtered


def compute_envelope(data):
    """Compute amplitude envelope using Hilbert transform."""
    analytic = sig.hilbert(data)
    return np.abs(analytic)


# =============================================================================
# PID FUNCTIONS
# =============================================================================

def discretize_timeseries(x, n_bins=4, method='quantile'):
    """Discretize continuous time series into symbols."""
    x = np.asarray(x)
    
    # Handle NaNs and infinities
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
    """Build a dit Distribution from time series with specified lags."""
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


def compute_pid_summary(dist, pid_class=PID_MMI):
    """Extract key PID values from a distribution."""
    if dist is None:
        return {'redundancy': np.nan, 'unique_0': np.nan, 'unique_1': np.nan, 'synergy': np.nan}
    
    try:
        pid = pid_class(dist)
    except Exception:
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


def compute_lag_sweep(signal, max_lag=15, n_bins=4):
    """Compute PID for all lag pairs for a single signal."""
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    
    results = []
    for lag1 in range(1, max_lag):
        for lag2 in range(lag1 + 1, max_lag + 1):
            dist = build_embedding_distribution(signal_discrete, lags=[lag1, lag2])
            pid = compute_pid_summary(dist)
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            results.append(pid)
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_all_bands_channels(df, channels, fs, max_lag=15, n_bins=4, subsample=None):
    """
    Analyze all frequency bands for all channels.
    
    Returns nested dict: results[band][channel] = DataFrame
    """
    all_results = {}
    
    for band_name, (fmin, fmax) in BANDS.items():
        print(f"\n{'='*60}")
        print(f"Processing band: {band_name.upper()} ({fmin}-{fmax} Hz)")
        print("="*60)
        
        band_results = {}
        
        for i, ch in enumerate(channels):
            print(f"  Channel {i+1}/{len(channels)}: {ch}", end=" ")
            
            # Get signal
            signal = df[ch].values
            if subsample is not None and len(signal) > subsample:
                signal = signal[:subsample]
            
            # Remove NaNs
            signal = signal[~np.isnan(signal)]
            
            # Bandpass filter
            try:
                filtered = bandpass_filter(signal, fmin, fmax, fs)
                
                # Optional: use envelope for amplitude dynamics
                # filtered = compute_envelope(filtered)
                
                # Compute PID
                ch_results = compute_lag_sweep(filtered, max_lag=max_lag, n_bins=n_bins)
                ch_results['channel'] = ch
                ch_results['band'] = band_name
                band_results[ch] = ch_results
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        all_results[band_name] = band_results
    
    return all_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_band_comparison_heatmaps(all_results, channel, save_path=None):
    """Create side-by-side heatmaps for all bands for one channel."""
    n_bands = len(BANDS)
    fig, axes = plt.subplots(2, n_bands, figsize=(4*n_bands, 8))
    
    for idx, (band_name, band_results) in enumerate(all_results.items()):
        if channel not in band_results:
            continue
            
        df = band_results[channel]
        max_lag = int(df['lag2'].max())
        lags = list(range(1, max_lag + 1))
        n_lags = len(lags)
        
        # Redundancy
        red_matrix = np.full((n_lags, n_lags), np.nan)
        syn_matrix = np.full((n_lags, n_lags), np.nan)
        
        for _, row in df.iterrows():
            i = int(row['lag1']) - 1
            j = int(row['lag2']) - 1
            red_matrix[i, j] = row['redundancy']
            syn_matrix[i, j] = row['synergy']
        
        # Plot redundancy
        mask = np.isnan(red_matrix)
        sns.heatmap(red_matrix, ax=axes[0, idx], cmap='Greens', mask=mask,
                    xticklabels=5, yticklabels=5, cbar_kws={'label': 'bits'})
        axes[0, idx].set_title(f'{band_name.upper()}\nRedundancy')
        axes[0, idx].set_xlabel('Lag 2')
        axes[0, idx].set_ylabel('Lag 1' if idx == 0 else '')
        
        # Plot synergy
        sns.heatmap(syn_matrix, ax=axes[1, idx], cmap='Reds', mask=mask,
                    xticklabels=5, yticklabels=5, cbar_kws={'label': 'bits'})
        axes[1, idx].set_title('Synergy')
        axes[1, idx].set_xlabel('Lag 2')
        axes[1, idx].set_ylabel('Lag 1' if idx == 0 else '')
    
    plt.suptitle(f'Temporal PID Across Frequency Bands: {channel}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_band_summary_bars(all_results, channels, save_path=None):
    """Bar chart comparing mean PID across bands."""
    summary = []
    
    for band_name, band_results in all_results.items():
        for ch, df in band_results.items():
            summary.append({
                'band': band_name,
                'channel': ch.replace('eeg-', ''),
                'mean_redundancy': df['redundancy'].mean(),
                'mean_synergy': df['synergy'].mean(),
                'max_synergy': df['synergy'].max(),
                'syn_red_ratio': df['synergy'].mean() / max(df['redundancy'].mean(), 1e-10)
            })
    
    df_summary = pd.DataFrame(summary)
    
    # Aggregate by band
    band_agg = df_summary.groupby('band').agg({
        'mean_redundancy': ['mean', 'std'],
        'mean_synergy': ['mean', 'std'],
        'syn_red_ratio': ['mean', 'std']
    }).reset_index()
    band_agg.columns = ['band', 'red_mean', 'red_std', 'syn_mean', 'syn_std', 'ratio_mean', 'ratio_std']
    
    # Order bands by frequency
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_agg['band'] = pd.Categorical(band_agg['band'], categories=band_order, ordered=True)
    band_agg = band_agg.sort_values('band')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(band_agg))
    colors = [BAND_COLORS[b] for b in band_agg['band']]
    
    # Redundancy
    axes[0].bar(x, band_agg['red_mean'], yerr=band_agg['red_std'], 
                color=colors, capsize=5, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b.upper() for b in band_agg['band']])
    axes[0].set_ylabel('Mean Redundancy (bits)')
    axes[0].set_title('Redundancy by Frequency Band')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synergy
    axes[1].bar(x, band_agg['syn_mean'], yerr=band_agg['syn_std'], 
                color=colors, capsize=5, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([b.upper() for b in band_agg['band']])
    axes[1].set_ylabel('Mean Synergy (bits)')
    axes[1].set_title('Synergy by Frequency Band')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Ratio
    axes[2].bar(x, band_agg['ratio_mean'], yerr=band_agg['ratio_std'], 
                color=colors, capsize=5, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([b.upper() for b in band_agg['band']])
    axes[2].set_ylabel('Synergy / Redundancy Ratio')
    axes[2].set_title('Synergy-Redundancy Balance')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Temporal PID Summary Across Frequency Bands', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_summary


def plot_band_channel_heatmap(all_results, metric='synergy', save_path=None):
    """Heatmap of metric values: bands x channels."""
    # Build matrix
    bands = list(BANDS.keys())
    channels = list(list(all_results.values())[0].keys())
    
    matrix = np.zeros((len(bands), len(channels)))
    
    for i, band in enumerate(bands):
        for j, ch in enumerate(channels):
            if ch in all_results[band]:
                matrix[i, j] = all_results[band][ch][metric].mean()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    cmap = 'Reds' if metric == 'synergy' else 'Greens'
    sns.heatmap(matrix, ax=ax, cmap=cmap, annot=True, fmt='.2f',
                xticklabels=[ch.replace('eeg-', '') for ch in channels],
                yticklabels=[b.upper() for b in bands],
                cbar_kws={'label': f'Mean {metric.title()} (bits)'})
    
    ax.set_xlabel('Channel')
    ax.set_ylabel('Frequency Band')
    ax.set_title(f'Mean {metric.title()} Across Bands and Channels')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_regional_band_comparison(all_results, save_path=None):
    """Compare PID across brain regions for each band."""
    # Aggregate by region
    summary = []
    
    for band_name, band_results in all_results.items():
        for ch, df in band_results.items():
            ch_short = ch.replace('eeg-', '')
            
            # Find region
            region = 'Other'
            for reg_name, reg_channels in REGIONS.items():
                if ch_short in reg_channels:
                    region = reg_name
                    break
            
            summary.append({
                'band': band_name,
                'channel': ch_short,
                'region': region,
                'mean_redundancy': df['redundancy'].mean(),
                'mean_synergy': df['synergy'].mean(),
                'syn_red_ratio': df['synergy'].mean() / max(df['redundancy'].mean(), 1e-10)
            })
    
    df_summary = pd.DataFrame(summary)
    
    # Aggregate by band and region
    agg = df_summary.groupby(['band', 'region']).agg({
        'mean_redundancy': 'mean',
        'mean_synergy': 'mean',
        'syn_red_ratio': 'mean'
    }).reset_index()
    
    # Create pivot tables
    pivot_syn = agg.pivot(index='region', columns='band', values='mean_synergy')
    pivot_red = agg.pivot(index='region', columns='band', values='mean_redundancy')
    pivot_ratio = agg.pivot(index='region', columns='band', values='syn_red_ratio')
    
    # Reorder columns and rows
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    region_order = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital', 'Reference']
    
    for pivot in [pivot_syn, pivot_red, pivot_ratio]:
        pivot = pivot.reindex(columns=[b for b in band_order if b in pivot.columns])
        pivot = pivot.reindex([r for r in region_order if r in pivot.index])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Synergy heatmap
    sns.heatmap(pivot_syn.reindex(columns=[b for b in band_order if b in pivot_syn.columns])
                        .reindex([r for r in region_order if r in pivot_syn.index]),
                ax=axes[0], cmap='Reds', annot=True, fmt='.3f',
                cbar_kws={'label': 'bits'})
    axes[0].set_title('Mean Synergy')
    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('Brain Region')
    
    # Redundancy heatmap
    sns.heatmap(pivot_red.reindex(columns=[b for b in band_order if b in pivot_red.columns])
                        .reindex([r for r in region_order if r in pivot_red.index]),
                ax=axes[1], cmap='Greens', annot=True, fmt='.2f',
                cbar_kws={'label': 'bits'})
    axes[1].set_title('Mean Redundancy')
    axes[1].set_xlabel('Frequency Band')
    axes[1].set_ylabel('')
    
    # Ratio heatmap
    sns.heatmap(pivot_ratio.reindex(columns=[b for b in band_order if b in pivot_ratio.columns])
                          .reindex([r for r in region_order if r in pivot_ratio.index]),
                ax=axes[2], cmap='Purples', annot=True, fmt='.3f',
                cbar_kws={'label': 'ratio'})
    axes[2].set_title('Synergy/Redundancy Ratio')
    axes[2].set_xlabel('Frequency Band')
    axes[2].set_ylabel('')
    
    plt.suptitle('Temporal PID: Brain Regions × Frequency Bands', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_summary


def plot_lag_profile_by_band(all_results, channel, save_path=None):
    """Plot PID vs lag distance for each band for one channel."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (band_name, band_results) in enumerate(all_results.items()):
        if channel not in band_results:
            continue
            
        ax = axes[idx]
        df = band_results[channel].copy()
        df['lag_diff'] = df['lag2'] - df['lag1']
        
        grouped = df.groupby('lag_diff').agg({
            'redundancy': 'mean',
            'synergy': 'mean',
            'unique_0': 'mean',
            'unique_1': 'mean'
        }).reset_index()
        
        ax.plot(grouped['lag_diff'], grouped['redundancy'], 'o-', 
                label='Redundancy', color='green', linewidth=2)
        ax.plot(grouped['lag_diff'], grouped['synergy'], 's-', 
                label='Synergy', color='red', linewidth=2)
        ax.plot(grouped['lag_diff'], grouped['unique_0'] + grouped['unique_1'], '^-', 
                label='Total Unique', color='blue', linewidth=2)
        
        ax.set_xlabel('Lag Difference')
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'{band_name.upper()} ({BANDS[band_name][0]}-{BANDS[band_name][1]} Hz)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.suptitle(f'PID vs Lag Distance by Frequency Band: {channel}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_synergy_peaks_by_band(all_results, save_path=None):
    """Find and plot where synergy peaks for each band and channel."""
    summary = []
    
    for band_name, band_results in all_results.items():
        for ch, df in band_results.items():
            # Find peak synergy
            peak_idx = df['synergy'].idxmax()
            peak_row = df.loc[peak_idx]
            
            summary.append({
                'band': band_name,
                'channel': ch.replace('eeg-', ''),
                'peak_lag1': int(peak_row['lag1']),
                'peak_lag2': int(peak_row['lag2']),
                'peak_lag_diff': int(peak_row['lag2'] - peak_row['lag1']),
                'peak_synergy': peak_row['synergy']
            })
    
    df_summary = pd.DataFrame(summary)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot of peak lag difference by band
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    df_summary['band'] = pd.Categorical(df_summary['band'], categories=band_order, ordered=True)
    
    colors = [BAND_COLORS[b] for b in band_order]
    box_data = [df_summary[df_summary['band'] == b]['peak_lag_diff'].values for b in band_order]
    
    bp = axes[0].boxplot(box_data, labels=[b.upper() for b in band_order], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    axes[0].set_xlabel('Frequency Band')
    axes[0].set_ylabel('Peak Synergy Lag Difference')
    axes[0].set_title('Where Does Synergy Peak?')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Scatter of peak synergy value vs lag difference
    for band in band_order:
        band_data = df_summary[df_summary['band'] == band]
        axes[1].scatter(band_data['peak_lag_diff'], band_data['peak_synergy'],
                       color=BAND_COLORS[band], label=band.upper(), alpha=0.7, s=50)
    
    axes[1].set_xlabel('Peak Synergy Lag Difference')
    axes[1].set_ylabel('Peak Synergy Value (bits)')
    axes[1].set_title('Peak Synergy: Value vs Timescale')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle('Synergy Peak Analysis by Frequency Band', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return df_summary


def plot_grand_summary(all_results, fs, save_path=None):
    """Create a comprehensive summary figure."""
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # --- Panel A: Band comparison bars ---
    ax_bars = fig.add_subplot(gs[0, :2])
    
    band_summary = []
    for band_name, band_results in all_results.items():
        reds = [df['redundancy'].mean() for df in band_results.values()]
        syns = [df['synergy'].mean() for df in band_results.values()]
        band_summary.append({
            'band': band_name,
            'redundancy': np.mean(reds),
            'synergy': np.mean(syns),
            'ratio': np.mean(syns) / max(np.mean(reds), 1e-10)
        })
    
    df_bands = pd.DataFrame(band_summary)
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    df_bands['band'] = pd.Categorical(df_bands['band'], categories=band_order, ordered=True)
    df_bands = df_bands.sort_values('band')
    
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
    
    # --- Panel B: Synergy/Redundancy ratio ---
    ax_ratio = fig.add_subplot(gs[0, 2:])
    
    colors = [BAND_COLORS[b] for b in df_bands['band']]
    ax_ratio.bar(x, df_bands['ratio'], color=colors, alpha=0.8)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([b.upper() for b in df_bands['band']])
    ax_ratio.set_ylabel('Synergy / Redundancy')
    ax_ratio.set_title('B) Synergy-Redundancy Balance')
    ax_ratio.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Balanced')
    ax_ratio.grid(axis='y', alpha=0.3)
    
    # --- Panel C: Region x Band heatmap (Synergy) ---
    ax_heat_syn = fig.add_subplot(gs[1, :2])
    
    region_band_data = []
    for band_name, band_results in all_results.items():
        for ch, df in band_results.items():
            ch_short = ch.replace('eeg-', '')
            region = 'Other'
            for reg_name, reg_channels in REGIONS.items():
                if ch_short in reg_channels:
                    region = reg_name
                    break
            region_band_data.append({
                'band': band_name, 'region': region, 
                'synergy': df['synergy'].mean(),
                'redundancy': df['redundancy'].mean()
            })
    
    df_rb = pd.DataFrame(region_band_data)
    pivot_syn = df_rb.groupby(['region', 'band'])['synergy'].mean().unstack()
    
    region_order = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
    pivot_syn = pivot_syn.reindex([r for r in region_order if r in pivot_syn.index])
    pivot_syn = pivot_syn[[b for b in band_order if b in pivot_syn.columns]]
    
    sns.heatmap(pivot_syn, ax=ax_heat_syn, cmap='Reds', annot=True, fmt='.3f',
                cbar_kws={'label': 'bits'})
    ax_heat_syn.set_title('C) Mean Synergy: Region × Band')
    ax_heat_syn.set_xlabel('Frequency Band')
    ax_heat_syn.set_ylabel('Brain Region')
    
    # --- Panel D: Region x Band heatmap (Ratio) ---
    ax_heat_ratio = fig.add_subplot(gs[1, 2:])
    
    pivot_ratio = df_rb.groupby(['region', 'band']).apply(
        lambda x: x['synergy'].mean() / max(x['redundancy'].mean(), 1e-10)
    ).unstack()
    pivot_ratio = pivot_ratio.reindex([r for r in region_order if r in pivot_ratio.index])
    pivot_ratio = pivot_ratio[[b for b in band_order if b in pivot_ratio.columns]]
    
    sns.heatmap(pivot_ratio, ax=ax_heat_ratio, cmap='Purples', annot=True, fmt='.3f',
                cbar_kws={'label': 'ratio'})
    ax_heat_ratio.set_title('D) Synergy/Redundancy Ratio: Region × Band')
    ax_heat_ratio.set_xlabel('Frequency Band')
    ax_heat_ratio.set_ylabel('')
    
    # --- Panel E: Lag profiles for alpha (example) ---
    ax_lag_alpha = fig.add_subplot(gs[2, :2])
    
    if 'alpha' in all_results:
        # Average across all channels for alpha
        all_lag_data = []
        for ch, df in all_results['alpha'].items():
            df = df.copy()
            df['lag_diff'] = df['lag2'] - df['lag1']
            all_lag_data.append(df)
        
        combined = pd.concat(all_lag_data)
        grouped = combined.groupby('lag_diff').agg({
            'redundancy': ['mean', 'std'],
            'synergy': ['mean', 'std']
        }).reset_index()
        grouped.columns = ['lag_diff', 'red_mean', 'red_std', 'syn_mean', 'syn_std']
        
        ax_lag_alpha.errorbar(grouped['lag_diff'], grouped['red_mean'], 
                              yerr=grouped['red_std'], fmt='o-', color='green',
                              label='Redundancy', capsize=3)
        ax_lag_alpha.errorbar(grouped['lag_diff'], grouped['syn_mean'], 
                              yerr=grouped['syn_std'], fmt='s-', color='red',
                              label='Synergy', capsize=3)
        ax_lag_alpha.set_xlabel('Lag Difference (samples)')
        ax_lag_alpha.set_ylabel('Information (bits)')
        ax_lag_alpha.set_title(f'E) Alpha Band: PID vs Lag Distance (all channels)')
        ax_lag_alpha.legend()
        ax_lag_alpha.grid(alpha=0.3)
        
        # Add time axis on top
        ax_lag_alpha_time = ax_lag_alpha.twiny()
        lag_diffs = grouped['lag_diff'].values
        ax_lag_alpha_time.set_xlim(ax_lag_alpha.get_xlim())
        ax_lag_alpha_time.set_xticks(lag_diffs[::2])
        ax_lag_alpha_time.set_xticklabels([f'{int(l/fs*1000)}ms' for l in lag_diffs[::2]])
        ax_lag_alpha_time.set_xlabel('Time (ms)')
    
    # --- Panel F: Peak synergy lag by band ---
    ax_peaks = fig.add_subplot(gs[2, 2:])
    
    peak_data = []
    for band_name, band_results in all_results.items():
        for ch, df in band_results.items():
            peak_idx = df['synergy'].idxmax()
            peak_row = df.loc[peak_idx]
            peak_data.append({
                'band': band_name,
                'peak_lag_diff': int(peak_row['lag2'] - peak_row['lag1']),
                'peak_synergy': peak_row['synergy']
            })
    
    df_peaks = pd.DataFrame(peak_data)
    
    for band in band_order:
        band_data = df_peaks[df_peaks['band'] == band]
        # Convert to ms
        peak_times = band_data['peak_lag_diff'] / fs * 1000
        ax_peaks.scatter(peak_times, band_data['peak_synergy'],
                        color=BAND_COLORS[band], label=band.upper(), alpha=0.6, s=40)
    
    ax_peaks.set_xlabel('Peak Synergy Timescale (ms)')
    ax_peaks.set_ylabel('Peak Synergy (bits)')
    ax_peaks.set_title('F) Synergy Peaks: Magnitude vs Timescale')
    ax_peaks.legend(loc='upper right')
    ax_peaks.grid(alpha=0.3)
    
    plt.suptitle('EEG Temporal PID: Frequency Band Analysis Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_all_bands_grid(all_results, metric='synergy', save_path=None):
    """Grid of heatmaps: rows=bands, cols=sample of channels."""
    bands = list(BANDS.keys())
    channels = list(list(all_results.values())[0].keys())
    
    # Select subset of channels for visibility
    selected_channels = channels[:10] if len(channels) > 10 else channels
    
    n_bands = len(bands)
    n_cols = len(selected_channels)
    
    fig, axes = plt.subplots(n_bands, n_cols, figsize=(2*n_cols, 2*n_bands))
    
    # Get global colorbar limits
    all_values = []
    for band_results in all_results.values():
        for df in band_results.values():
            all_values.extend(df[metric].dropna().values)
    vmin, vmax = np.percentile(all_values, [5, 95])
    
    cmap = 'Reds' if metric == 'synergy' else 'Greens'
    
    for i, band in enumerate(bands):
        for j, ch in enumerate(selected_channels):
            ax = axes[i, j]
            
            if ch in all_results[band]:
                df = all_results[band][ch]
                max_lag = int(df['lag2'].max())
                n_lags = max_lag
                
                matrix = np.full((n_lags, n_lags), np.nan)
                for _, row in df.iterrows():
                    ii = int(row['lag1']) - 1
                    jj = int(row['lag2']) - 1
                    matrix[ii, jj] = row[metric]
                
                mask = np.isnan(matrix)
                sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask,
                            vmin=vmin, vmax=vmax,
                            xticklabels=False, yticklabels=False, cbar=False)
            
            if i == 0:
                ax.set_title(ch.replace('eeg-', ''), fontsize=9)
            if j == 0:
                ax.set_ylabel(band.upper(), fontsize=10)
    
    plt.suptitle(f'{metric.title()} Heatmaps: Frequency Bands × Channels', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EEG BANDPASS TEMPORAL PID ANALYSIS")
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
    MAX_LAG = 15
    N_BINS = 4
    SUBSAMPLE = 50000
    
    print(f"\nParameters:")
    print(f"  Max lag: {MAX_LAG} samples (~{MAX_LAG/fs*1000:.1f} ms)")
    print(f"  Bins: {N_BINS}")
    print(f"  Samples used: {SUBSAMPLE}")
    print(f"  Bands: {list(BANDS.keys())}")
    
    # Run analysis
    all_results = analyze_all_bands_channels(
        df, channels, fs,
        max_lag=MAX_LAG,
        n_bins=N_BINS,
        subsample=SUBSAMPLE
    )
    
    # Save raw results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)
    
    all_dfs = []
    for band_name, band_results in all_results.items():
        for ch, ch_df in band_results.items():
            all_dfs.append(ch_df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / "bandpass_pid_all.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'bandpass_pid_all.csv'}")
    
    # Generate figures
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print("="*70)
    
    # 1. Grand summary figure
    print("\n1. Grand summary figure...")
    plot_grand_summary(all_results, fs, save_path=RESULTS_DIR / "grand_summary.png")
    
    # 2. Band comparison bars
    print("2. Band comparison bars...")
    df_summary = plot_band_summary_bars(all_results, channels, 
                                        save_path=RESULTS_DIR / "band_comparison_bars.png")
    df_summary.to_csv(RESULTS_DIR / "band_channel_summary.csv", index=False)
    
    # 3. Band x Channel heatmaps
    print("3. Band x Channel heatmaps...")
    plot_band_channel_heatmap(all_results, metric='synergy',
                              save_path=RESULTS_DIR / "band_channel_synergy.png")
    plot_band_channel_heatmap(all_results, metric='redundancy',
                              save_path=RESULTS_DIR / "band_channel_redundancy.png")
    
    # 4. Regional comparison
    print("4. Regional comparison...")
    df_regional = plot_regional_band_comparison(all_results,
                                                save_path=RESULTS_DIR / "regional_band_comparison.png")
    df_regional.to_csv(RESULTS_DIR / "regional_summary.csv", index=False)
    
    # 5. Synergy peak analysis
    print("5. Synergy peak analysis...")
    df_peaks = plot_synergy_peaks_by_band(all_results,
                                          save_path=RESULTS_DIR / "synergy_peaks.png")
    df_peaks.to_csv(RESULTS_DIR / "synergy_peaks.csv", index=False)
    
    # 6. Grid of all bands/channels
    print("6. Band x Channel grids...")
    plot_all_bands_grid(all_results, metric='synergy',
                        save_path=RESULTS_DIR / "all_bands_grid_synergy.png")
    plot_all_bands_grid(all_results, metric='redundancy',
                        save_path=RESULTS_DIR / "all_bands_grid_redundancy.png")
    
    # 7. Individual channel band comparisons (for a few key channels)
    print("7. Individual channel analyses...")
    key_channels = ['eeg-Fz', 'eeg-O1', 'eeg-T3', 'eeg-Cz']
    for ch in key_channels:
        if ch in channels:
            ch_name = ch.replace('eeg-', '')
            plot_band_comparison_heatmaps(all_results, ch,
                                          save_path=RESULTS_DIR / f"channel_{ch_name}_bands.png")
            plot_lag_profile_by_band(all_results, ch,
                                     save_path=RESULTS_DIR / f"channel_{ch_name}_lag_profiles.png")
    
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
