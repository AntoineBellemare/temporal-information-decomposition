"""
EEG Temporal PID Analysis - Extended Version
=============================================

Key improvements over basic version:
1. Extended lags up to seconds (not just milliseconds)
2. Multiple binning strategies compared
3. Single-lag mutual information alongside PID
4. Downsampling option for slow dynamics
5. Clear time-axis labeling

Usage:
    python eeg_temporal_pid_extended.py
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
from scipy.ndimage import uniform_filter1d

import dit
from dit.pid import PID_MMI
from dit import Distribution
from dit.shannon import mutual_information

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "eeg_extended"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
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


def downsample_signal(x, factor):
    """Downsample signal by averaging."""
    # Anti-aliasing filter first
    x_filtered = uniform_filter1d(x, size=factor)
    return x_filtered[::factor]


def compute_envelope(x, fs, lowpass_freq=10):
    """Compute amplitude envelope using Hilbert transform."""
    analytic = sig.hilbert(x)
    envelope = np.abs(analytic)
    
    # Lowpass filter the envelope
    nyq = fs / 2
    if lowpass_freq < nyq:
        b, a = sig.butter(2, lowpass_freq / nyq, btype='low')
        envelope = sig.filtfilt(b, a, envelope)
    
    return envelope


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


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_lag_mi(signal, lags, n_bins=4):
    """
    Compute mutual information for each single lag.
    
    This answers: "How much does lag-k tell about the present?"
    """
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    
    results = []
    for lag in lags:
        mi = compute_mutual_information(signal_discrete, lag)
        results.append({'lag': lag, 'MI': mi})
    
    return pd.DataFrame(results)


def analyze_pid_lag_pairs(signal, lag_pairs, n_bins=4):
    """
    Compute PID for specified lag pairs.
    """
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    
    results = []
    for lag1, lag2 in lag_pairs:
        dist = build_embedding_distribution(signal_discrete, lags=[lag1, lag2])
        pid = compute_pid_summary(dist)
        pid['lag1'] = lag1
        pid['lag2'] = lag2
        results.append(pid)
    
    return pd.DataFrame(results)


def analyze_binning_sensitivity(signal, lags, bin_sizes=[3, 4, 6, 8]):
    """
    Compare different bin sizes for a fixed set of lags.
    """
    results = []
    
    for n_bins in bin_sizes:
        df_mi = analyze_single_lag_mi(signal, lags, n_bins=n_bins)
        df_mi['n_bins'] = n_bins
        results.append(df_mi)
    
    return pd.concat(results, ignore_index=True)


def create_lag_schedule(fs, timescales_ms):
    """
    Create lag values corresponding to specific timescales.
    
    Parameters:
    -----------
    fs : float
        Sampling rate in Hz
    timescales_ms : list
        List of timescales in milliseconds
    
    Returns:
    --------
    lags : list of int
        Corresponding lag values
    """
    lags = [max(1, int(t * fs / 1000)) for t in timescales_ms]
    return sorted(list(set(lags)))  # Remove duplicates


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_single_lag_mi(df_mi, fs, channel_name, save_path=None):
    """
    Plot mutual information as function of lag (with time axis).
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    lags = df_mi['lag'].values
    mi_values = df_mi['MI'].values
    
    # Plot
    ax.plot(lags, mi_values, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.fill_between(lags, 0, mi_values, alpha=0.3, color='steelblue')
    
    ax.set_xlabel('Lag (samples)')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title(f'Single-Lag Predictive Information: {channel_name}')
    ax.grid(alpha=0.3)
    
    # Add time axis on top
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    
    # Select reasonable tick positions
    lag_ticks = lags[::max(1, len(lags)//10)]
    time_labels = [f'{l/fs*1000:.0f}ms' if l/fs < 1 else f'{l/fs:.1f}s' for l in lag_ticks]
    
    ax_time.set_xticks(lag_ticks)
    ax_time.set_xticklabels(time_labels)
    ax_time.set_xlabel('Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_lag_profiles(df_pid, fs, channel_name, save_path=None):
    """
    Plot how PID changes with lag distance (like eeg_temporal_pid_analysis).
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute lag difference and average over lag pairs with same distance
    df = df_pid.copy()
    df['lag_diff'] = df['lag2'] - df['lag1']
    
    grouped = df.groupby('lag_diff').agg({
        'redundancy': ['mean', 'std'],
        'synergy': ['mean', 'std'],
        'unique_0': 'mean',
        'unique_1': 'mean'
    }).reset_index()
    grouped.columns = ['lag_diff', 'red_mean', 'red_std', 'syn_mean', 'syn_std', 'uniq0', 'uniq1']
    grouped['total_unique'] = grouped['uniq0'] + grouped['uniq1']
    
    # Convert to time
    lag_diffs = grouped['lag_diff'].values
    time_ms = lag_diffs / fs * 1000
    
    ax.errorbar(time_ms, grouped['red_mean'], yerr=grouped['red_std'], 
                fmt='o-', color='green', label='Redundancy', capsize=3, linewidth=2)
    ax.errorbar(time_ms, grouped['syn_mean'], yerr=grouped['syn_std'], 
                fmt='s-', color='red', label='Synergy', capsize=3, linewidth=2)
    ax.plot(time_ms, grouped['total_unique'], '^-', color='blue', 
            label='Total Unique', linewidth=2)
    
    ax.set_xlabel('Lag Difference (ms)')
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'PID vs Lag Distance: {channel_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add sample axis on top
    ax_samples = ax.twiny()
    ax_samples.set_xlim(ax.get_xlim())
    sample_ticks = time_ms[::max(1, len(time_ms)//8)]
    ax_samples.set_xticks(sample_ticks)
    ax_samples.set_xticklabels([f'{int(t*fs/1000)}' for t in sample_ticks])
    ax_samples.set_xlabel('Lag Difference (samples)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_binning_comparison(df_binning, fs, channel_name, save_path=None):
    """
    Compare MI across different bin sizes.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, df_binning['n_bins'].nunique()))
    
    for i, (n_bins, group) in enumerate(df_binning.groupby('n_bins')):
        ax.plot(group['lag'], group['MI'], 'o-', color=colors[i], 
                label=f'{n_bins} bins', linewidth=2, markersize=5)
    
    ax.set_xlabel('Lag (samples)')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title(f'Effect of Binning on MI: {channel_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Add time axis
    ax_time = ax.twiny()
    lags = df_binning['lag'].unique()
    ax_time.set_xlim(ax.get_xlim())
    lag_ticks = lags[::max(1, len(lags)//8)]
    time_labels = [f'{l/fs*1000:.0f}ms' if l/fs < 1 else f'{l/fs:.1f}s' for l in lag_ticks]
    ax_time.set_xticks(lag_ticks)
    ax_time.set_xticklabels(time_labels)
    ax_time.set_xlabel('Time')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_pid_at_timescales(df_pid, fs, channel_name, save_path=None):
    """
    Plot PID components at different timescale pairs.
    """
    # Group by lag1 and show how PID changes with lag2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Unique lag1 values
    lag1_values = sorted(df_pid['lag1'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(lag1_values)))
    
    components = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Redundancy', 'Synergy', 'Unique (Lag 1)', 'Unique (Lag 2)']
    
    for ax, comp, title in zip(axes.flat, components, titles):
        for i, lag1 in enumerate(lag1_values):
            subset = df_pid[df_pid['lag1'] == lag1]
            lag1_time = lag1 / fs * 1000
            label = f'Lag1={lag1_time:.0f}ms' if lag1_time < 1000 else f'Lag1={lag1/fs:.1f}s'
            ax.plot(subset['lag2'], subset[comp], 'o-', color=colors[i], 
                    label=label, alpha=0.7)
        
        ax.set_xlabel('Lag 2 (samples)')
        ax.set_ylabel(f'{title} (bits)')
        ax.set_title(title)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'PID Components vs Lag Pairs: {channel_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_timescale_heatmap(df_pid, fs, channel_name, metric='synergy', save_path=None):
    """
    Heatmap with TIME labels instead of lag numbers.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get unique lags
    lag1_vals = sorted(df_pid['lag1'].unique())
    lag2_vals = sorted(df_pid['lag2'].unique())
    
    # Create matrix
    n1, n2 = len(lag1_vals), len(lag2_vals)
    matrix = np.full((n1, n2), np.nan)
    
    for _, row in df_pid.iterrows():
        i = lag1_vals.index(int(row['lag1']))
        j = lag2_vals.index(int(row['lag2']))
        matrix[i, j] = row[metric]
    
    # Create time labels
    def lag_to_time_label(lag):
        t_ms = lag / fs * 1000
        if t_ms < 1000:
            return f'{t_ms:.0f}ms'
        else:
            return f'{t_ms/1000:.1f}s'
    
    time_labels_1 = [lag_to_time_label(l) for l in lag1_vals]
    time_labels_2 = [lag_to_time_label(l) for l in lag2_vals]
    
    cmap = 'Reds' if metric == 'synergy' else 'Greens' if metric == 'redundancy' else 'Blues'
    
    mask = np.isnan(matrix)
    sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask, annot=True, fmt='.3f',
                xticklabels=time_labels_2, yticklabels=time_labels_1,
                cbar_kws={'label': f'{metric.title()} (bits)'})
    
    ax.set_xlabel('Lag 2 (time)')
    ax.set_ylabel('Lag 1 (time)')
    ax.set_title(f'{metric.title()} at Different Timescales: {channel_name}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_comprehensive_summary(df_mi, df_pid, fs, channel_name, save_path=None):
    """
    Create a comprehensive summary figure.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # --- Panel A: Single-lag MI ---
    ax_mi = fig.add_subplot(gs[0, :])
    
    lags = df_mi['lag'].values
    ax_mi.plot(lags, df_mi['MI'].values, 'o-', color='steelblue', linewidth=2)
    ax_mi.fill_between(lags, 0, df_mi['MI'].values, alpha=0.3, color='steelblue')
    ax_mi.set_xlabel('Lag')
    ax_mi.set_ylabel('MI (bits)')
    ax_mi.set_title('A) Single-Lag Mutual Information: How much does each lag predict the present?')
    ax_mi.grid(alpha=0.3)
    
    # Add time labels
    ax_mi_time = ax_mi.twiny()
    ax_mi_time.set_xlim(ax_mi.get_xlim())
    lag_ticks = lags[::max(1, len(lags)//10)]
    time_labels = [f'{l/fs*1000:.0f}ms' if l/fs < 1 else f'{l/fs:.1f}s' for l in lag_ticks]
    ax_mi_time.set_xticks(lag_ticks)
    ax_mi_time.set_xticklabels(time_labels)
    
    # --- Panel B: Redundancy heatmap ---
    ax_red = fig.add_subplot(gs[1, 0])
    
    lag1_vals = sorted(df_pid['lag1'].unique())
    lag2_vals = sorted(df_pid['lag2'].unique())
    matrix_red = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
    for _, row in df_pid.iterrows():
        i = lag1_vals.index(int(row['lag1']))
        j = lag2_vals.index(int(row['lag2']))
        matrix_red[i, j] = row['redundancy']
    
    time_labels_1 = [f'{l/fs*1000:.0f}' for l in lag1_vals]
    time_labels_2 = [f'{l/fs*1000:.0f}' for l in lag2_vals]
    
    sns.heatmap(matrix_red, ax=ax_red, cmap='Greens', 
                xticklabels=time_labels_2, yticklabels=time_labels_1,
                cbar_kws={'label': 'bits'})
    ax_red.set_xlabel('Lag 2 (ms)')
    ax_red.set_ylabel('Lag 1 (ms)')
    ax_red.set_title('B) Redundancy')
    
    # --- Panel C: Synergy heatmap ---
    ax_syn = fig.add_subplot(gs[1, 1])
    
    matrix_syn = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
    for _, row in df_pid.iterrows():
        i = lag1_vals.index(int(row['lag1']))
        j = lag2_vals.index(int(row['lag2']))
        matrix_syn[i, j] = row['synergy']
    
    sns.heatmap(matrix_syn, ax=ax_syn, cmap='Reds',
                xticklabels=time_labels_2, yticklabels=time_labels_1,
                cbar_kws={'label': 'bits'})
    ax_syn.set_xlabel('Lag 2 (ms)')
    ax_syn.set_ylabel('Lag 1 (ms)')
    ax_syn.set_title('C) Synergy')
    
    # --- Panel D: Ratio heatmap ---
    ax_ratio = fig.add_subplot(gs[1, 2])
    
    matrix_ratio = matrix_syn / np.maximum(matrix_red, 1e-10)
    
    sns.heatmap(matrix_ratio, ax=ax_ratio, cmap='Purples',
                xticklabels=time_labels_2, yticklabels=time_labels_1,
                cbar_kws={'label': 'ratio'})
    ax_ratio.set_xlabel('Lag 2 (ms)')
    ax_ratio.set_ylabel('Lag 1 (ms)')
    ax_ratio.set_title('D) Synergy/Redundancy Ratio')
    
    # --- Panel E: Summary interpretation ---
    ax_interp = fig.add_subplot(gs[2, :])
    ax_interp.axis('off')
    
    # Find key statistics
    max_mi_lag = df_mi.loc[df_mi['MI'].idxmax(), 'lag']
    max_mi_time = max_mi_lag / fs * 1000
    
    max_syn_idx = df_pid['synergy'].idxmax()
    max_syn_lags = (df_pid.loc[max_syn_idx, 'lag1'], df_pid.loc[max_syn_idx, 'lag2'])
    max_syn_times = (max_syn_lags[0]/fs*1000, max_syn_lags[1]/fs*1000)
    
    max_red_idx = df_pid['redundancy'].idxmax()
    max_red_lags = (df_pid.loc[max_red_idx, 'lag1'], df_pid.loc[max_red_idx, 'lag2'])
    max_red_times = (max_red_lags[0]/fs*1000, max_red_lags[1]/fs*1000)
    
    interpretation = f"""
    INTERPRETATION SUMMARY: {channel_name}
    {'='*60}
    
    SINGLE-LAG ANALYSIS (Panel A):
    • Peak MI at lag {int(max_mi_lag)} ({max_mi_time:.1f} ms)
    • This is the timescale with maximum predictive power
    
    REDUNDANCY (Panel B):
    • Peak at lags ({int(max_red_lags[0])}, {int(max_red_lags[1])}) = ({max_red_times[0]:.1f}, {max_red_times[1]:.1f}) ms
    • These lags carry OVERLAPPING information about the present
    • High redundancy = smooth, autocorrelated signal
    
    SYNERGY (Panel C):
    • Peak at lags ({int(max_syn_lags[0])}, {int(max_syn_lags[1])}) = ({max_syn_times[0]:.1f}, {max_syn_times[1]:.1f}) ms
    • These lags carry COMPLEMENTARY information (need BOTH to predict)
    • High synergy = nonlinear temporal dependencies
    
    WHAT THIS MEANS:
    • If redundancy >> synergy: Signal is mostly smooth/predictable
    • If synergy is notable: There are nonlinear dynamics at those timescales
    • If ratio varies with lag: Different temporal mechanisms at different scales
    """
    
    ax_interp.text(0.02, 0.95, interpretation, transform=ax_interp.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Temporal PID Analysis: {channel_name}', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EEG TEMPORAL PID ANALYSIS - EXTENDED VERSION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    eeg_file = DATA_DIR / "test_eeg_dsi.csv"
    print(f"\nLoading: {eeg_file}")
    df = pd.read_csv(eeg_file)
    
    # Get EEG channels
    channels = [c for c in df.columns if c.startswith('eeg-')]
    print(f"Found {len(channels)} EEG channels")
    
    # Estimate sampling rate
    try:
        ts = pd.to_datetime(df['timestamp'])
        dt = (ts.iloc[1] - ts.iloc[0]).total_seconds()
        fs = 1 / dt
    except:
        fs = 300
    print(f"Sampling rate: {fs:.1f} Hz")
    
    # Parameters
    SUBSAMPLE = 100000  # More samples for longer lags
    N_BINS = 4
    
    # Define timescales to analyze (in milliseconds)
    TIMESCALES_MS = [3, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    
    # Convert to lags
    LAGS = create_lag_schedule(fs, TIMESCALES_MS)
    print(f"\nTimescales (ms): {TIMESCALES_MS}")
    print(f"Corresponding lags: {LAGS}")
    print(f"Time range: {LAGS[0]/fs*1000:.1f} ms to {LAGS[-1]/fs*1000:.1f} ms ({LAGS[-1]/fs:.2f} s)")
    
    # Create ALL lag pairs (not just adjacent)
    LAG_PAIRS = []
    for i, lag1 in enumerate(LAGS):
        for lag2 in LAGS[i+1:]:  # ALL pairs, not just adjacent
            if lag2 > lag1:
                LAG_PAIRS.append((lag1, lag2))
    print(f"Number of lag pairs for PID: {len(LAG_PAIRS)}")
    
    # Analyze ALL channels (not just a few)
    channels_to_analyze = channels  # ALL 20 channels
    print(f"Channels to analyze: {len(channels_to_analyze)}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print("="*70)
    
    all_mi_results = []
    all_pid_results = []
    
    for ch in channels_to_analyze:
        print(f"\n--- Analyzing {ch} ---")
        
        # Get signal
        signal = df[ch].values[:SUBSAMPLE]
        signal = signal[~np.isnan(signal)]
        
        # 1. Single-lag mutual information
        print("  Computing single-lag MI...")
        df_mi = analyze_single_lag_mi(signal, LAGS, n_bins=N_BINS)
        df_mi['channel'] = ch
        all_mi_results.append(df_mi)
        
        # 2. PID for lag pairs
        print("  Computing PID for lag pairs...")
        df_pid = analyze_pid_lag_pairs(signal, LAG_PAIRS, n_bins=N_BINS)
        df_pid['channel'] = ch
        all_pid_results.append(df_pid)
        
        # 3. Binning sensitivity (optional)
        print("  Testing binning sensitivity...")
        df_binning = analyze_binning_sensitivity(signal, LAGS[:8], bin_sizes=[3, 4, 6, 8])
        
        # Generate figures for this channel
        ch_name = ch.replace('eeg-', '')
        
        plot_single_lag_mi(df_mi, fs, ch, 
                          save_path=RESULTS_DIR / f"mi_single_lag_{ch_name}.png")
        
        plot_lag_profiles(df_pid, fs, ch,
                         save_path=RESULTS_DIR / f"lag_profiles_{ch_name}.png")
        
        plot_timescale_heatmap(df_pid, fs, ch, metric='synergy',
                              save_path=RESULTS_DIR / f"synergy_heatmap_{ch_name}.png")
        
        plot_timescale_heatmap(df_pid, fs, ch, metric='redundancy',
                              save_path=RESULTS_DIR / f"redundancy_heatmap_{ch_name}.png")
        
        plot_comprehensive_summary(df_mi, df_pid, fs, ch,
                                  save_path=RESULTS_DIR / f"comprehensive_{ch_name}.png")
    
    # Combine and save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)
    
    df_mi_all = pd.concat(all_mi_results, ignore_index=True)
    df_pid_all = pd.concat(all_pid_results, ignore_index=True)
    
    # Add time columns
    df_mi_all['time_ms'] = df_mi_all['lag'] / fs * 1000
    df_pid_all['lag1_ms'] = df_pid_all['lag1'] / fs * 1000
    df_pid_all['lag2_ms'] = df_pid_all['lag2'] / fs * 1000
    
    df_mi_all.to_csv(RESULTS_DIR / "mutual_information_all.csv", index=False)
    df_pid_all.to_csv(RESULTS_DIR / "pid_all.csv", index=False)
    
    # =========================================================================
    # SUMMARY FIGURES
    # =========================================================================
    print("\n--- Generating summary figures ---")
    
    # 1. Grid of redundancy heatmaps for all channels
    print("  Creating redundancy grid...")
    n_channels = len(channels_to_analyze)
    n_cols = 5
    n_rows = int(np.ceil(n_channels / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    for idx, ch in enumerate(channels_to_analyze):
        ax = axes[idx]
        df_ch = df_pid_all[df_pid_all['channel'] == ch]
        
        lag1_vals = sorted(df_ch['lag1'].unique())
        lag2_vals = sorted(df_ch['lag2'].unique())
        
        matrix = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
        for _, row in df_ch.iterrows():
            i = lag1_vals.index(row['lag1'])
            j = lag2_vals.index(row['lag2'])
            matrix[i, j] = row['redundancy']
        
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, cmap='Greens', mask=mask,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax.set_title(ch.replace('eeg-', ''), fontsize=10)
    
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Temporal Redundancy Heatmaps: All Channels (Extended Lags)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_channels_redundancy_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Grid of synergy heatmaps for all channels
    print("  Creating synergy grid...")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    for idx, ch in enumerate(channels_to_analyze):
        ax = axes[idx]
        df_ch = df_pid_all[df_pid_all['channel'] == ch]
        
        lag1_vals = sorted(df_ch['lag1'].unique())
        lag2_vals = sorted(df_ch['lag2'].unique())
        
        matrix = np.full((len(lag1_vals), len(lag2_vals)), np.nan)
        for _, row in df_ch.iterrows():
            i = lag1_vals.index(row['lag1'])
            j = lag2_vals.index(row['lag2'])
            matrix[i, j] = row['synergy']
        
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, cmap='Reds', mask=mask,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax.set_title(ch.replace('eeg-', ''), fontsize=10)
    
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Temporal Synergy Heatmaps: All Channels (Extended Lags)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "all_channels_synergy_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Lag profiles comparison (4 selected channels)
    print("  Creating lag profile comparison...")
    selected_channels = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    selected_channels = [c for c in selected_channels if c in channels_to_analyze]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, ch in zip(axes.flat, selected_channels):
        df_ch = df_pid_all[df_pid_all['channel'] == ch].copy()
        df_ch['lag_diff'] = df_ch['lag2'] - df_ch['lag1']
        
        grouped = df_ch.groupby('lag_diff').agg({
            'redundancy': 'mean',
            'synergy': 'mean',
            'unique_0': 'mean',
            'unique_1': 'mean'
        }).reset_index()
        grouped['total_unique'] = grouped['unique_0'] + grouped['unique_1']
        
        time_ms = grouped['lag_diff'] / fs * 1000
        
        ax.plot(time_ms, grouped['redundancy'], 'o-', color='green', 
                label='Redundancy', linewidth=2)
        ax.plot(time_ms, grouped['synergy'], 's-', color='red', 
                label='Synergy', linewidth=2)
        ax.plot(time_ms, grouped['total_unique'], '^-', color='blue', 
                label='Total Unique', linewidth=2)
        
        ax.set_xlabel('Lag Difference (ms)')
        ax.set_ylabel('Information (bits)')
        ax.set_title(ch.replace('eeg-', ''))
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('PID vs Lag Distance: Selected Channels (Extended to 1 second)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lag_profiles_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Channel comparison bars
    print("  Creating channel comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Aggregate by channel
    summary = df_pid_all.groupby('channel').agg({
        'redundancy': 'mean',
        'synergy': 'mean',
        'unique_0': 'mean',
        'unique_1': 'mean'
    }).reset_index()
    summary['syn_red_ratio'] = summary['synergy'] / summary['redundancy'].replace(0, 1e-10)
    summary = summary.sort_values('syn_red_ratio', ascending=False)
    
    x = np.arange(len(summary))
    
    # Redundancy
    axes[0].bar(x, summary['redundancy'], color='green', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.replace('eeg-', '') for c in summary['channel']], rotation=45, ha='right')
    axes[0].set_ylabel('Mean Redundancy (bits)')
    axes[0].set_title('Redundancy by Channel')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synergy
    axes[1].bar(x, summary['synergy'], color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.replace('eeg-', '') for c in summary['channel']], rotation=45, ha='right')
    axes[1].set_ylabel('Mean Synergy (bits)')
    axes[1].set_title('Synergy by Channel')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Ratio
    axes[2].bar(x, summary['syn_red_ratio'], color='purple', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c.replace('eeg-', '') for c in summary['channel']], rotation=45, ha='right')
    axes[2].set_ylabel('Synergy / Redundancy')
    axes[2].set_title('Synergy-Redundancy Balance')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Channel Comparison (Extended Lags: 3ms to 1s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "channel_comparison_bars.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. MI comparison across channels
    print("  Creating MI comparison...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for ch, group in df_mi_all.groupby('channel'):
        ch_name = ch.replace('eeg-', '')
        ax.plot(group['time_ms'], group['MI'], 'o-', label=ch_name, linewidth=1.5, markersize=4, alpha=0.7)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mutual Information (bits)')
    ax.set_title('Single-Lag MI Across All Channels (Extended to 1 second)')
    ax.legend(loc='upper right', ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mi_all_channels.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary table
    summary.to_csv(RESULTS_DIR / "channel_summary.csv", index=False)
    
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
