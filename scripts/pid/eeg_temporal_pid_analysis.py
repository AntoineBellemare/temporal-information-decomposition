"""
EEG Temporal PID Analysis
=========================

Analyze the temporal structure of EEG signals using PID on time-delay embeddings.

For each channel:
- Discretize the continuous signal
- Compute PID for all lag pairs up to max_lag
- Extract redundancy, unique, synergy components
- Compare "temporal fingerprints" across brain regions

Includes two strategies to separate genuine structure from autocorrelation:
1. Non-overlapping embeddings: Use lags where windows don't overlap
2. AR baseline comparison: Compare to matched AR(1) surrogate

Usage:
    python eeg_temporal_pid_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from datetime import datetime
from scipy.linalg import toeplitz, solve
import warnings
warnings.filterwarnings('ignore')

import dit
from dit.pid import PID_MMI
from dit import Distribution

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "eeg_basic"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def discretize_timeseries(x, n_bins=4, method='quantile'):
    """Discretize continuous time series into symbols."""
    x = np.asarray(x)
    
    # Handle NaNs
    if np.any(np.isnan(x)):
        x = x[~np.isnan(x)]
    
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
        # Ensure unique bin edges
        bins = np.unique(bins)
        if len(bins) < 3:
            # Fallback to uniform if quantiles fail
            bins = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
        else:
            bins[0] -= 1e-10
            bins[-1] += 1e-10
    else:
        bins = np.linspace(x.min() - 1e-10, x.max() + 1e-10, n_bins + 1)
    
    return np.digitize(x, bins[1:-1])


# =============================================================================
# CORRELATION TIME ESTIMATION
# =============================================================================

def estimate_correlation_time(signal, fs=300, threshold=0.1):
    """
    Estimate the correlation time of a signal.
    
    The correlation time (τ_corr) is defined as the lag at which the
    autocorrelation drops below a threshold (e.g., 0.1 or 1/e ≈ 0.37).
    
    Parameters:
    -----------
    signal : array-like
        Input time series
    fs : float
        Sampling frequency in Hz
    threshold : float
        Autocorrelation threshold (default 0.1 = 10% correlation remaining)
        
    Returns:
    --------
    tau_corr_samples : int
        Correlation time in samples
    tau_corr_ms : float
        Correlation time in milliseconds
    autocorr : array
        Full autocorrelation function (for plotting)
    """
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)
    signal = signal[~np.isnan(signal)]
    
    n = len(signal)
    max_lag = min(n // 2, int(fs * 2))  # Up to 2 seconds
    
    # Compute normalized autocorrelation
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:n-1+max_lag] / autocorr[n-1]  # Normalize and take positive lags
    
    # Find where autocorrelation drops below threshold
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    
    if len(below_threshold) > 0:
        tau_corr_samples = below_threshold[0]
    else:
        tau_corr_samples = max_lag  # Never drops below threshold
    
    tau_corr_ms = tau_corr_samples / fs * 1000
    
    return tau_corr_samples, tau_corr_ms, autocorr


# =============================================================================
# STRATEGY 1: BEYOND-CORRELATION-TIME ANALYSIS (TRUE NON-OVERLAP)
# =============================================================================

def compute_pid_beyond_correlation(signal, tau_corr, fs=300, n_bins=4, n_lags=10):
    """
    Compute PID using lags BEYOND the correlation time.
    
    This ensures the time points being compared are statistically independent
    (autocorrelation ≈ 0), not just smooth continuations of each other.
    
    Parameters:
    -----------
    signal : array-like
        Input time series
    tau_corr : int
        Correlation time in samples (from estimate_correlation_time)
    fs : float
        Sampling frequency
    n_bins : int
        Number of discretization bins
    n_lags : int
        Number of lag pairs to compute beyond tau_corr
        
    Returns:
    --------
    DataFrame with PID values at lags beyond correlation time
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    
    # Discretize
    signal_disc = discretize_timeseries(signal, n_bins=n_bins)
    
    # Define lags starting from tau_corr
    # Use logarithmic spacing for efficiency
    min_lag = max(tau_corr, 1)
    max_lag = min(len(signal) // 4, min_lag * 10)  # Don't exceed signal length
    
    if max_lag <= min_lag:
        return pd.DataFrame()  # Signal too short
    
    lags = np.unique(np.geomspace(min_lag, max_lag, n_lags).astype(int))
    
    results = []
    
    for i, lag1 in enumerate(lags[:-1]):
        for lag2 in lags[i+1:]:
            # Build distribution
            max_lag_pair = max(lag1, lag2)
            n = len(signal_disc)
            
            if n <= max_lag_pair:
                continue
            
            outcomes = []
            for t in range(max_lag_pair, n):
                src1 = int(signal_disc[t - lag1])
                src2 = int(signal_disc[t - lag2])
                target = int(signal_disc[t])
                outcomes.append(f"{src1}{src2}{target}")
            
            if len(outcomes) < 50:  # Need sufficient samples
                continue
            
            counts = Counter(outcomes)
            total = sum(counts.values())
            outcomes_list = list(counts.keys())
            probs = [counts[o] / total for o in outcomes_list]
            
            try:
                dist = Distribution(outcomes_list, probs)
                pid = compute_pid_summary(dist)
            except:
                continue
            
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            pid['lag1_ms'] = lag1 / fs * 1000
            pid['lag2_ms'] = lag2 / fs * 1000
            pid['lag_diff'] = lag2 - lag1
            pid['lag_diff_ms'] = (lag2 - lag1) / fs * 1000
            results.append(pid)
    
    return pd.DataFrame(results)


def compute_pid_within_correlation(signal, tau_corr, fs=300, n_bins=4, n_lags=8):
    """
    Compute PID using lags WITHIN the correlation time (overlapping regime).
    
    These lags are where autocorrelation is still significant, so PID
    will be dominated by this smooth temporal structure.
    
    Uses logarithmic spacing for efficiency (not all pairs).
    
    Parameters:
    -----------
    signal : array-like
        Input time series
    tau_corr : int
        Correlation time in samples
    fs : float
        Sampling frequency
    n_bins : int
        Number of discretization bins
    n_lags : int
        Number of lag values to sample (not all pairs!)
        
    Returns:
    --------
    DataFrame with PID values at lags within correlation time
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    
    # Discretize
    signal_disc = discretize_timeseries(signal, n_bins=n_bins)
    
    # Use logarithmically spaced lags within tau_corr for efficiency
    max_lag = max(tau_corr, 2)
    lags = np.unique(np.geomspace(1, max_lag, n_lags).astype(int))
    
    results = []
    
    for i, lag1 in enumerate(lags[:-1]):
        for lag2 in lags[i+1:]:
            n = len(signal_disc)
            
            if n <= lag2:
                continue
            
            outcomes = []
            for t in range(lag2, n):
                src1 = int(signal_disc[t - lag1])
                src2 = int(signal_disc[t - lag2])
                target = int(signal_disc[t])
                outcomes.append(f"{src1}{src2}{target}")
            
            if len(outcomes) < 50:
                continue
            
            counts = Counter(outcomes)
            total = sum(counts.values())
            outcomes_list = list(counts.keys())
            probs = [counts[o] / total for o in outcomes_list]
            
            try:
                dist = Distribution(outcomes_list, probs)
                pid = compute_pid_summary(dist)
            except:
                continue
            
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            pid['lag1_ms'] = lag1 / fs * 1000
            pid['lag2_ms'] = lag2 / fs * 1000
            pid['lag_diff'] = lag2 - lag1
            pid['lag_diff_ms'] = (lag2 - lag1) / fs * 1000
            results.append(pid)
    
    return pd.DataFrame(results)


# =============================================================================
# STRATEGY 2: AR(1) BASELINE COMPARISON
# =============================================================================

def fit_ar1(signal):
    """
    Fit AR(1) model to signal and return coefficient.
    
    AR(1): x[t] = phi * x[t-1] + noise
    
    Returns:
    --------
    phi : float
        AR(1) coefficient (lag-1 autocorrelation)
    sigma : float
        Noise standard deviation
    """
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)
    signal = signal[~np.isnan(signal)]
    
    # Lag-1 autocorrelation = AR(1) coefficient
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:] / autocorr[n-1]  # Normalize
    
    phi = autocorr[1] if len(autocorr) > 1 else 0.0
    
    # Noise variance: Var(residual) = Var(x) * (1 - phi^2)
    sigma = np.std(signal) * np.sqrt(max(0, 1 - phi**2))
    
    return phi, sigma


def generate_ar1_surrogate(n_samples, phi, sigma, seed=None):
    """
    Generate AR(1) surrogate with matched autocorrelation structure.
    
    This serves as a null model for "what would PID look like if the
    signal were just linear autocorrelation with no higher-order structure?"
    """
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.randn(n_samples) * sigma
    x = np.zeros(n_samples)
    x[0] = noise[0]
    
    for t in range(1, n_samples):
        x[t] = phi * x[t-1] + noise[t]
    
    return x


def compute_ar1_baseline_comparison(signal, max_lag=15, n_bins=4, n_surrogates=5):
    """
    Compare actual PID to AR(1) surrogate baseline.
    
    For each lag pair, compute:
    - Actual PID from the real signal
    - Mean PID from AR(1) surrogates with matched autocorrelation
    - "Excess PID" = actual - surrogate (genuine non-AR structure)
    
    Parameters:
    -----------
    signal : array-like
        Input time series
    max_lag : int
        Maximum lag to consider
    n_bins : int
        Number of discretization bins
    n_surrogates : int
        Number of AR(1) surrogates to average
        
    Returns:
    --------
    df_actual : DataFrame
        PID from actual signal
    df_surrogate : DataFrame
        Mean PID from surrogates
    df_excess : DataFrame  
        Excess PID (actual - surrogate)
    ar1_params : dict
        Fitted AR(1) parameters
    """
    signal = np.asarray(signal)
    signal = signal[~np.isnan(signal)]
    
    # Fit AR(1) to signal
    phi, sigma = fit_ar1(signal)
    ar1_params = {'phi': phi, 'sigma': sigma}
    
    # Compute actual PID
    signal_discrete = discretize_timeseries(signal, n_bins=n_bins)
    actual_results = []
    
    for lag1 in range(1, max_lag):
        for lag2 in range(lag1 + 1, max_lag + 1):
            dist = build_embedding_distribution(signal_discrete, lags=[lag1, lag2])
            pid = compute_pid_summary(dist)
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            actual_results.append(pid)
    
    df_actual = pd.DataFrame(actual_results)
    
    # Compute surrogate PID (average over multiple surrogates)
    surrogate_pids = {key: [] for key in ['redundancy', 'unique_0', 'unique_1', 'synergy']}
    surrogate_meta = []
    
    for i in range(n_surrogates):
        surrogate = generate_ar1_surrogate(len(signal), phi, sigma, seed=42+i)
        surrogate_discrete = discretize_timeseries(surrogate, n_bins=n_bins)
        
        for lag1 in range(1, max_lag):
            for lag2 in range(lag1 + 1, max_lag + 1):
                dist = build_embedding_distribution(surrogate_discrete, lags=[lag1, lag2])
                pid = compute_pid_summary(dist)
                
                if i == 0:  # Only store metadata once
                    surrogate_meta.append({'lag1': lag1, 'lag2': lag2})
                
                for key in surrogate_pids:
                    if len(surrogate_pids[key]) <= len(surrogate_meta) - 1:
                        surrogate_pids[key].append([])
                    idx = (lag1 - 1) * (max_lag - lag1 // 2) + (lag2 - lag1 - 1)
                    # Build up lists
    
    # Recompute more cleanly
    surrogate_accum = []
    for i in range(n_surrogates):
        surrogate = generate_ar1_surrogate(len(signal), phi, sigma, seed=42+i)
        surrogate_discrete = discretize_timeseries(surrogate, n_bins=n_bins)
        
        surr_results = []
        for lag1 in range(1, max_lag):
            for lag2 in range(lag1 + 1, max_lag + 1):
                dist = build_embedding_distribution(surrogate_discrete, lags=[lag1, lag2])
                pid = compute_pid_summary(dist)
                pid['lag1'] = lag1
                pid['lag2'] = lag2
                surr_results.append(pid)
        
        surrogate_accum.append(pd.DataFrame(surr_results))
    
    # Average surrogates
    df_surrogate = surrogate_accum[0].copy()
    for key in ['redundancy', 'unique_0', 'unique_1', 'synergy']:
        vals = np.array([s[key].values for s in surrogate_accum])
        df_surrogate[key] = vals.mean(axis=0)
    
    # Compute excess (actual - surrogate baseline)
    df_excess = df_actual.copy()
    for key in ['redundancy', 'unique_0', 'unique_1', 'synergy']:
        df_excess[key] = df_actual[key].values - df_surrogate[key].values
    
    return df_actual, df_surrogate, df_excess, ar1_params


def build_embedding_distribution(x, lags=[1, 2]):
    """
    Build a dit Distribution from time series with specified lags.
    Sources: x[t - lag] for each lag
    Target: x[t]
    """
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
    except Exception as e:
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


def compute_channel_lag_sweep(signal, max_lag=15, n_bins=4, verbose=True):
    """
    Compute PID for all lag pairs for a single channel.
    
    Returns DataFrame with columns: lag1, lag2, redundancy, unique_0, unique_1, synergy
    """
    # Discretize
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


def compute_all_channels(df, channels, max_lag=15, n_bins=4, subsample=None):
    """
    Compute PID lag sweep for all channels.
    
    Parameters:
    -----------
    df : DataFrame with channel columns
    channels : list of channel names
    max_lag : maximum lag to consider
    n_bins : number of bins for discretization
    subsample : if not None, use only this many samples (for speed)
    
    Returns:
    --------
    all_results : dict of {channel: DataFrame}
    """
    all_results = {}
    
    for i, ch in enumerate(channels):
        print(f"  Processing channel {i+1}/{len(channels)}: {ch}")
        
        signal = df[ch].values
        
        # Subsample if requested
        if subsample is not None and len(signal) > subsample:
            signal = signal[:subsample]
        
        # Remove any NaNs
        signal = signal[~np.isnan(signal)]
        
        # Compute PID
        ch_results = compute_channel_lag_sweep(signal, max_lag=max_lag, n_bins=n_bins)
        ch_results['channel'] = ch
        all_results[ch] = ch_results
    
    return all_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_channel_heatmaps(df_channel, channel_name, save_path=None):
    """Create 2x2 heatmap for one channel."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    max_lag = int(df_channel['lag2'].max())
    lags = list(range(1, max_lag + 1))
    n_lags = len(lags)
    
    components = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Redundancy', 'Synergy', 'Unique (Lag 1)', 'Unique (Lag 2)']
    cmaps = ['Greens', 'Reds', 'Blues', 'Purples']
    
    for ax, comp, title, cmap in zip(axes.flat, components, titles, cmaps):
        matrix = np.full((n_lags, n_lags), np.nan)
        
        for _, row in df_channel.iterrows():
            i = int(row['lag1']) - 1
            j = int(row['lag2']) - 1
            matrix[i, j] = row[comp]
        
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask,
                    xticklabels=lags, yticklabels=lags,
                    cbar_kws={'label': 'bits'})
        ax.set_title(title)
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1')
    
    plt.suptitle(f'Temporal PID: {channel_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_channel_comparison(all_results, metric='redundancy', save_path=None):
    """Compare one PID metric across all channels."""
    
    # Aggregate mean across lags for each channel
    summary = []
    for ch, df in all_results.items():
        summary.append({
            'channel': ch,
            'mean': df[metric].mean(),
            'max': df[metric].max(),
            'std': df[metric].std()
        })
    
    df_summary = pd.DataFrame(summary).sort_values('mean', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_summary))
    ax.bar(x, df_summary['mean'], yerr=df_summary['std'], capsize=3, 
           color='steelblue', alpha=0.8, label='Mean ± Std')
    ax.scatter(x, df_summary['max'], color='red', marker='*', s=100, 
               label='Max', zorder=5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([ch.replace('eeg-', '') for ch in df_summary['channel']], 
                       rotation=45, ha='right')
    ax.set_xlabel('Channel')
    ax.set_ylabel(f'{metric.title()} (bits)')
    ax.set_title(f'Temporal {metric.title()} Across EEG Channels')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_channels_grid(all_results, metric='redundancy', save_path=None):
    """Create grid of heatmaps for all channels."""
    channels = list(all_results.keys())
    n_channels = len(channels)
    
    # Determine grid size
    n_cols = 5
    n_rows = int(np.ceil(n_channels / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    # Get global min/max for consistent colorbar
    all_values = []
    for df in all_results.values():
        all_values.extend(df[metric].dropna().values)
    vmin, vmax = np.percentile(all_values, [5, 95])
    
    for idx, (ch, df) in enumerate(all_results.items()):
        ax = axes[idx]
        
        max_lag = int(df['lag2'].max())
        lags = list(range(1, max_lag + 1))
        n_lags = len(lags)
        
        matrix = np.full((n_lags, n_lags), np.nan)
        for _, row in df.iterrows():
            i = int(row['lag1']) - 1
            j = int(row['lag2']) - 1
            matrix[i, j] = row[metric]
        
        mask = np.isnan(matrix)
        cmap = 'Greens' if metric == 'redundancy' else 'Reds' if metric == 'synergy' else 'Blues'
        
        sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask,
                    vmin=vmin, vmax=vmax,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax.set_title(ch.replace('eeg-', ''), fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide empty axes
    for idx in range(n_channels, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Temporal {metric.title()} Heatmaps Across All EEG Channels', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_synergy_redundancy_balance(all_results, save_path=None):
    """Plot synergy/redundancy ratio for each channel."""
    
    summary = []
    for ch, df in all_results.items():
        # Avoid division by zero
        ratio_mean = (df['synergy'] / df['redundancy'].replace(0, np.nan)).mean()
        ratio_max = (df['synergy'] / df['redundancy'].replace(0, np.nan)).max()
        
        summary.append({
            'channel': ch,
            'mean_ratio': ratio_mean,
            'max_ratio': ratio_max,
            'mean_redundancy': df['redundancy'].mean(),
            'mean_synergy': df['synergy'].mean()
        })
    
    df_summary = pd.DataFrame(summary).sort_values('mean_ratio', ascending=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Synergy/Redundancy ratio
    x = np.arange(len(df_summary))
    axes[0].bar(x, df_summary['mean_ratio'], color='purple', alpha=0.7)
    axes[0].axhline(1.0, color='gray', linestyle='--', label='Balanced (S=R)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ch.replace('eeg-', '') for ch in df_summary['channel']], 
                            rotation=45, ha='right')
    axes[0].set_ylabel('Synergy / Redundancy')
    axes[0].set_title('Synergy-Redundancy Balance')
    axes[0].legend()
    
    # Plot 2: Scatter of mean synergy vs mean redundancy
    axes[1].scatter(df_summary['mean_redundancy'], df_summary['mean_synergy'], 
                    s=100, alpha=0.7)
    for _, row in df_summary.iterrows():
        axes[1].annotate(row['channel'].replace('eeg-', ''), 
                        (row['mean_redundancy'], row['mean_synergy']),
                        fontsize=8, alpha=0.7)
    axes[1].set_xlabel('Mean Redundancy (bits)')
    axes[1].set_ylabel('Mean Synergy (bits)')
    axes[1].set_title('Synergy vs Redundancy by Channel')
    
    # Add diagonal line for reference
    lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
            max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
    axes[1].plot(lims, lims, 'k--', alpha=0.3, label='S=R line')
    
    # Plot 3: Topographic-style ordering (anterior to posterior)
    # Standard 10-20 ordering
    topo_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                  'T3', 'C3', 'Cz', 'C4', 'T4',
                  'T5', 'P3', 'P4', 'T6',
                  'O1', 'O2', 'A1', 'A2']
    
    # Map channels
    ordered_channels = []
    ordered_ratios = []
    for ch in topo_order:
        full_ch = f'eeg-{ch}'
        if full_ch in df_summary['channel'].values:
            ordered_channels.append(ch)
            ordered_ratios.append(df_summary[df_summary['channel'] == full_ch]['mean_ratio'].values[0])
    
    x = np.arange(len(ordered_channels))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(ordered_channels)))
    axes[2].bar(x, ordered_ratios, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(ordered_channels, rotation=45, ha='right')
    axes[2].set_ylabel('Synergy / Redundancy')
    axes[2].set_title('Topographic Order (Anterior → Posterior)')
    axes[2].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('EEG Temporal PID: Synergy-Redundancy Balance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_lag_profiles(all_results, selected_channels=None, save_path=None):
    """Plot how PID changes with lag distance for selected channels."""
    
    if selected_channels is None:
        # Pick a few representative channels
        selected_channels = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, ch in zip(axes.flat, selected_channels):
        if ch not in all_results:
            continue
            
        df = all_results[ch]
        
        # Average over lag pairs with same "lag distance" (lag2 - lag1)
        df = df.copy()
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
        
        ax.set_xlabel('Lag Difference (lag2 - lag1)')
        ax.set_ylabel('Information (bits)')
        ax.set_title(ch.replace('eeg-', ''))
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('PID vs Lag Distance: Selected Channels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# VISUALIZATION: CORRELATION TIME & OVERLAP COMPARISON
# =============================================================================

def plot_autocorrelation_analysis(signal, fs, channel_name, tau_corr, autocorr, save_path=None):
    """
    Plot autocorrelation function and mark the correlation time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Autocorrelation function
    lags_ms = np.arange(len(autocorr)) / fs * 1000
    axes[0].plot(lags_ms, autocorr, 'b-', linewidth=1.5)
    axes[0].axhline(0.1, color='r', linestyle='--', label='Threshold (0.1)')
    axes[0].axhline(-0.1, color='r', linestyle='--')
    axes[0].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[0].axvline(tau_corr / fs * 1000, color='green', linestyle='-', linewidth=2,
                    label=f'τ_corr = {tau_corr/fs*1000:.1f} ms')
    axes[0].fill_betweenx([-0.2, 1.0], 0, tau_corr / fs * 1000, 
                          color='red', alpha=0.1, label='Overlapping regime')
    axes[0].fill_betweenx([-0.2, 1.0], tau_corr / fs * 1000, lags_ms[-1], 
                          color='green', alpha=0.1, label='Non-overlapping regime')
    axes[0].set_xlabel('Lag (ms)')
    axes[0].set_ylabel('Autocorrelation')
    axes[0].set_title(f'Autocorrelation Function: {channel_name}')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlim([0, min(500, lags_ms[-1])])
    axes[0].set_ylim([-0.2, 1.0])
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Zoom on early lags (log scale for lag)
    axes[1].semilogx(lags_ms[1:], autocorr[1:], 'b-', linewidth=1.5)
    axes[1].axhline(0.1, color='r', linestyle='--')
    axes[1].axhline(-0.1, color='r', linestyle='--')
    axes[1].axhline(0, color='gray', linestyle='-', alpha=0.5)
    axes[1].axvline(tau_corr / fs * 1000, color='green', linestyle='-', linewidth=2)
    axes[1].set_xlabel('Lag (ms, log scale)')
    axes[1].set_ylabel('Autocorrelation')
    axes[1].set_title(f'τ_corr = {tau_corr} samples = {tau_corr/fs*1000:.1f} ms')
    axes[1].set_xlim([1, min(1000, lags_ms[-1])])
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_overlap_comparison(within_results, beyond_results, tau_corr, fs,
                            channel_name, save_path=None):
    """
    Compare PID from within-correlation-time (overlapping) vs 
    beyond-correlation-time (non-overlapping) regimes.
    
    This is the key comparison: does PID change when we move to
    statistically independent time points?
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    tau_corr_ms = tau_corr / fs * 1000
    
    # Top row: Individual profiles
    if len(within_results) > 0:
        df_within = within_results.copy()
        within_grouped = df_within.groupby('lag_diff_ms').agg({
            'redundancy': ['mean', 'std'], 
            'synergy': ['mean', 'std'], 
            'unique_0': ['mean', 'std'], 
            'unique_1': ['mean', 'std']
        }).reset_index()
        within_grouped.columns = ['lag_diff_ms', 
                                   'redundancy_mean', 'redundancy_std',
                                   'synergy_mean', 'synergy_std',
                                   'unique_0_mean', 'unique_0_std',
                                   'unique_1_mean', 'unique_1_std']
        within_grouped['unique_mean'] = within_grouped['unique_0_mean'] + within_grouped['unique_1_mean']
    else:
        within_grouped = pd.DataFrame()
    
    if len(beyond_results) > 0:
        df_beyond = beyond_results.copy()
        beyond_grouped = df_beyond.groupby('lag_diff_ms').agg({
            'redundancy': ['mean', 'std'], 
            'synergy': ['mean', 'std'], 
            'unique_0': ['mean', 'std'], 
            'unique_1': ['mean', 'std']
        }).reset_index()
        beyond_grouped.columns = ['lag_diff_ms', 
                                   'redundancy_mean', 'redundancy_std',
                                   'synergy_mean', 'synergy_std',
                                   'unique_0_mean', 'unique_0_std',
                                   'unique_1_mean', 'unique_1_std']
        beyond_grouped['unique_mean'] = beyond_grouped['unique_0_mean'] + beyond_grouped['unique_1_mean']
    else:
        beyond_grouped = pd.DataFrame()
    
    metrics = [('redundancy_mean', 'green', 'Redundancy'),
               ('synergy_mean', 'red', 'Synergy'),
               ('unique_mean', 'blue', 'Unique')]
    
    for ax, (metric, color, label) in zip(axes[0], metrics):
        if len(within_grouped) > 0:
            ax.plot(within_grouped['lag_diff_ms'], within_grouped[metric], 'o-',
                    color=color, label=f'Within τ_corr (<{tau_corr_ms:.0f}ms)', 
                    linewidth=2, alpha=0.8)
        if len(beyond_grouped) > 0:
            ax.plot(beyond_grouped['lag_diff_ms'], beyond_grouped[metric], 's--',
                    color=color, label=f'Beyond τ_corr (>{tau_corr_ms:.0f}ms)', 
                    linewidth=2, alpha=0.8, markerfacecolor='white', markersize=8)
        
        ax.axvline(tau_corr_ms, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Lag Difference (ms)')
        ax.set_ylabel('Information (bits)')
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Bottom row: Summary comparisons
    # Plot 4: Bar comparison of means
    if len(within_grouped) > 0 and len(beyond_grouped) > 0:
        within_means = [within_grouped['redundancy_mean'].mean(),
                        within_grouped['synergy_mean'].mean(),
                        within_grouped['unique_mean'].mean()]
        beyond_means = [beyond_grouped['redundancy_mean'].mean(),
                        beyond_grouped['synergy_mean'].mean(),
                        beyond_grouped['unique_mean'].mean()]
        
        x = np.arange(3)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, within_means, width, label=f'Within τ_corr', 
                       color=['green', 'red', 'blue'], alpha=0.6)
        axes[1, 0].bar(x + width/2, beyond_means, width, label=f'Beyond τ_corr',
                       color=['green', 'red', 'blue'], alpha=0.9, hatch='//')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['Redundancy', 'Synergy', 'Unique'])
        axes[1, 0].set_ylabel('Mean Information (bits)')
        axes[1, 0].set_title('Mean PID: Within vs Beyond τ_corr')
        axes[1, 0].legend()
        
        # Plot 5: Ratio comparison
        ratios_within = within_grouped['synergy_mean'] / within_grouped['redundancy_mean'].replace(0, np.nan)
        ratios_beyond = beyond_grouped['synergy_mean'] / beyond_grouped['redundancy_mean'].replace(0, np.nan)
        
        axes[1, 1].hist(ratios_within.dropna(), bins=20, alpha=0.6, color='red', 
                        label=f'Within τ_corr (mean={ratios_within.mean():.2f})')
        axes[1, 1].hist(ratios_beyond.dropna(), bins=20, alpha=0.6, color='blue',
                        label=f'Beyond τ_corr (mean={ratios_beyond.mean():.2f})')
        axes[1, 1].axvline(1.0, color='gray', linestyle='--', label='S=R')
        axes[1, 1].set_xlabel('Synergy / Redundancy')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Synergy-Redundancy Ratio Distribution')
        axes[1, 1].legend(fontsize=8)
        
        # Plot 6: Key insight text
        axes[1, 2].axis('off')
        
        # Compute change
        red_change = (beyond_means[0] - within_means[0]) / max(within_means[0], 1e-10) * 100
        syn_change = (beyond_means[1] - within_means[1]) / max(within_means[1], 1e-10) * 100
        uniq_change = (beyond_means[2] - within_means[2]) / max(within_means[2], 1e-10) * 100
        
        insight_text = f"""
KEY FINDINGS: {channel_name}
{'='*40}

Correlation time: τ_corr = {tau_corr_ms:.1f} ms

WITHIN τ_corr (overlapping, autocorrelated):
  • Redundancy: {within_means[0]:.3f} bits
  • Synergy:    {within_means[1]:.3f} bits  
  • Unique:     {within_means[2]:.3f} bits
  • S/R ratio:  {ratios_within.mean():.3f}

BEYOND τ_corr (non-overlapping, independent):
  • Redundancy: {beyond_means[0]:.3f} bits ({red_change:+.1f}%)
  • Synergy:    {beyond_means[1]:.3f} bits ({syn_change:+.1f}%)
  • Unique:     {beyond_means[2]:.3f} bits ({uniq_change:+.1f}%)
  • S/R ratio:  {ratios_beyond.mean():.3f}

INTERPRETATION:
{'Genuine structure' if beyond_means[0] > 0.01 else 'Mostly autocorrelation'}
"""
        axes[1, 2].text(0.05, 0.95, insight_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Overlapping vs Non-Overlapping PID: {channel_name}\n'
                 f'(Correlation time τ_corr = {tau_corr_ms:.1f} ms)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ar_baseline_comparison(df_actual, df_surrogate, df_excess, ar1_params, 
                                channel_name, save_path=None):
    """
    Visualize actual PID vs AR(1) baseline and the excess (genuine structure).
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    phi = ar1_params['phi']
    
    # Top row: Actual vs Surrogate heatmaps
    max_lag = int(df_actual['lag2'].max())
    lags = list(range(1, max_lag + 1))
    n_lags = len(lags)
    
    components = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles_top = ['Redundancy (Actual)', 'Synergy (Actual)', 
                  'Unique₁ (Actual)', 'Unique₂ (Actual)']
    cmaps = ['Greens', 'Reds', 'Blues', 'Purples']
    
    for idx, (comp, title, cmap) in enumerate(zip(components, titles_top, cmaps)):
        ax = axes[0, idx]
        matrix = np.full((n_lags, n_lags), np.nan)
        
        for _, row in df_actual.iterrows():
            i = int(row['lag1']) - 1
            j = int(row['lag2']) - 1
            matrix[i, j] = row[comp]
        
        mask = np.isnan(matrix)
        sns.heatmap(matrix, ax=ax, cmap=cmap, mask=mask, fmt='.3f',
                    xticklabels=lags[::2], yticklabels=lags[::2], cbar=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1')
    
    # Bottom row: Excess over AR(1) baseline
    titles_bot = ['Excess Redundancy', 'Excess Synergy', 
                  'Excess Unique₁', 'Excess Unique₂']
    
    for idx, (comp, title, cmap) in enumerate(zip(components, titles_bot, cmaps)):
        ax = axes[1, idx]
        matrix = np.full((n_lags, n_lags), np.nan)
        
        for _, row in df_excess.iterrows():
            i = int(row['lag1']) - 1
            j = int(row['lag2']) - 1
            matrix[i, j] = row[comp]
        
        mask = np.isnan(matrix)
        
        # Use diverging colormap centered at 0 for excess
        vmax = np.nanmax(np.abs(matrix))
        sns.heatmap(matrix, ax=ax, cmap='RdBu_r', mask=mask, 
                    center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=lags[::2], yticklabels=lags[::2], cbar=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Lag 2')
        ax.set_ylabel('Lag 1')
    
    plt.suptitle(f'AR(1) Baseline Comparison: {channel_name}\n'
                 f'(φ = {phi:.3f}, Excess = Actual − AR(1) Surrogate)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_excess_summary(all_excess_results, save_path=None):
    """
    Summary plot showing mean excess PID across all channels.
    
    Positive excess = genuine structure beyond AR(1) autocorrelation.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    components = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Excess Redundancy', 'Excess Synergy', 'Excess Unique₁', 'Excess Unique₂']
    colors = ['green', 'red', 'blue', 'purple']
    
    for ax, comp, title, color in zip(axes, components, titles, colors):
        means = []
        stds = []
        channels = []
        
        for ch, df_excess in all_excess_results.items():
            means.append(df_excess[comp].mean())
            stds.append(df_excess[comp].std())
            channels.append(ch.replace('eeg-', ''))
        
        x = np.arange(len(channels))
        bars = ax.bar(x, means, yerr=stds, capsize=2, color=color, alpha=0.7)
        
        # Highlight significant excess (> 0)
        for bar, mean in zip(bars, means):
            if mean > 0:
                bar.set_edgecolor('black')
                bar.set_linewidth(2)
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Excess (bits)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Genuine Structure: Excess PID Over AR(1) Baseline\n'
                 '(Positive = structure beyond linear autocorrelation)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_overlap_summary_all_channels(within_results, beyond_results, correlation_times, 
                                       save_path=None):
    """
    Summary plot comparing within vs beyond correlation time across ALL channels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    channels = []
    within_red = []
    within_syn = []
    beyond_red = []
    beyond_syn = []
    tau_values = []
    
    for ch in within_results.keys():
        if len(within_results[ch]) > 0 and len(beyond_results[ch]) > 0:
            channels.append(ch.replace('eeg-', ''))
            within_red.append(within_results[ch]['redundancy'].mean())
            within_syn.append(within_results[ch]['synergy'].mean())
            beyond_red.append(beyond_results[ch]['redundancy'].mean())
            beyond_syn.append(beyond_results[ch]['synergy'].mean())
            tau_values.append(correlation_times[ch]['tau_ms'])
    
    x = np.arange(len(channels))
    width = 0.35
    
    # Plot 1: Redundancy comparison
    axes[0, 0].bar(x - width/2, within_red, width, label='Within τ_corr', color='green', alpha=0.6)
    axes[0, 0].bar(x + width/2, beyond_red, width, label='Beyond τ_corr', color='green', alpha=0.9, hatch='//')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(channels, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Redundancy (bits)')
    axes[0, 0].set_title('Redundancy: Within vs Beyond τ_corr')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Synergy comparison
    axes[0, 1].bar(x - width/2, within_syn, width, label='Within τ_corr', color='red', alpha=0.6)
    axes[0, 1].bar(x + width/2, beyond_syn, width, label='Beyond τ_corr', color='red', alpha=0.9, hatch='//')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(channels, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Synergy (bits)')
    axes[0, 1].set_title('Synergy: Within vs Beyond τ_corr')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Change in PID (beyond - within)
    red_change = np.array(beyond_red) - np.array(within_red)
    syn_change = np.array(beyond_syn) - np.array(within_syn)
    
    axes[1, 0].bar(x - width/2, red_change, width, label='ΔRedundancy', color='green', alpha=0.7)
    axes[1, 0].bar(x + width/2, syn_change, width, label='ΔSynergy', color='red', alpha=0.7)
    axes[1, 0].axhline(0, color='gray', linestyle='--')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(channels, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Change (bits)')
    axes[1, 0].set_title('Change: Beyond − Within τ_corr\n(Negative = reduced by autocorrelation)')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Correlation times
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(channels)))
    axes[1, 1].bar(x, tau_values, color=colors)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(channels, rotation=45, ha='right')
    axes[1, 1].set_ylabel('τ_corr (ms)')
    axes[1, 1].set_title('Correlation Time by Channel')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_tau = np.mean(tau_values)
    axes[1, 1].axhline(mean_tau, color='red', linestyle='--', 
                       label=f'Mean = {mean_tau:.1f} ms')
    axes[1, 1].legend()
    
    plt.suptitle('Overlap Analysis Summary: All Channels\n'
                 'Comparing PID within vs beyond correlation time', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("="*70)
    print("EEG TEMPORAL PID ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Load data
    eeg_file = DATA_DIR / "test_eeg_dsi.csv"
    print(f"\nLoading: {eeg_file}")
    df = pd.read_csv(eeg_file)
    
    # Get EEG channels (exclude timestamp)
    channels = [c for c in df.columns if c.startswith('eeg-')]
    print(f"Found {len(channels)} EEG channels: {channels}")
    print(f"Total samples: {len(df)}")
    
    # Estimate sampling rate from timestamps
    try:
        ts = pd.to_datetime(df['timestamp'])
        dt = (ts.iloc[1] - ts.iloc[0]).total_seconds()
        fs = 1 / dt
        print(f"Estimated sampling rate: {fs:.1f} Hz")
    except:
        fs = 300  # Default assumption
        print(f"Assuming sampling rate: {fs} Hz")
    
    # Parameters
    MAX_LAG = 15  # At 300 Hz, lag 15 = 50ms
    N_BINS = 4
    SUBSAMPLE = 20000  # Reduced for speed (still statistically robust)
    
    print(f"\nParameters:")
    print(f"  Max lag: {MAX_LAG} samples (~{MAX_LAG/fs*1000:.1f} ms)")
    print(f"  Bins: {N_BINS}")
    print(f"  Samples used: {SUBSAMPLE}")
    
    # Compute PID for all channels
    print(f"\n{'='*70}")
    print("COMPUTING TEMPORAL PID FOR ALL CHANNELS")
    print("="*70)
    
    all_results = compute_all_channels(
        df, channels, 
        max_lag=MAX_LAG, 
        n_bins=N_BINS, 
        subsample=SUBSAMPLE
    )
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print("="*70)
    
    # Combine all results into one DataFrame
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(RESULTS_DIR / "eeg_temporal_pid_all_channels.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'eeg_temporal_pid_all_channels.csv'}")
    
    # Generate figures
    print(f"\n{'='*70}")
    print("GENERATING FIGURES")
    print("="*70)
    
    # 1. Heatmaps for each channel
    print("\nGenerating individual channel heatmaps...")
    for ch, df_ch in all_results.items():
        ch_name = ch.replace('eeg-', '')
        plot_channel_heatmaps(df_ch, ch, 
                              save_path=RESULTS_DIR / f"heatmap_{ch_name}.png")
    
    # 2. Channel comparison for each metric
    print("\nGenerating channel comparison plots...")
    for metric in ['redundancy', 'synergy', 'unique_0']:
        plot_channel_comparison(all_results, metric=metric,
                               save_path=RESULTS_DIR / f"channel_comparison_{metric}.png")
    
    # 3. Grid of all channels
    print("\nGenerating channel grid plots...")
    for metric in ['redundancy', 'synergy']:
        plot_all_channels_grid(all_results, metric=metric,
                              save_path=RESULTS_DIR / f"all_channels_grid_{metric}.png")
    
    # 4. Synergy-redundancy balance
    print("\nGenerating synergy-redundancy balance plot...")
    plot_synergy_redundancy_balance(all_results, 
                                    save_path=RESULTS_DIR / "synergy_redundancy_balance.png")
    
    # 5. Lag profiles for selected channels
    print("\nGenerating lag profile plots...")
    plot_lag_profiles(all_results, 
                     selected_channels=['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3'],
                     save_path=RESULTS_DIR / "lag_profiles.png")
    
    # =========================================================================
    # STRATEGY 1: CORRELATION TIME & OVERLAP ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("STRATEGY 1: CORRELATION TIME & OVERLAP ANALYSIS")
    print("="*70)
    print("Estimating correlation time and comparing overlapping vs non-overlapping regimes...")
    
    correlation_times = {}
    within_results = {}
    beyond_results = {}
    
    for i, ch in enumerate(channels):
        print(f"  Processing {i+1}/{len(channels)}: {ch}...")
        signal = df[ch].values[:SUBSAMPLE]
        signal = signal[~np.isnan(signal)]
        
        # Estimate correlation time
        tau_corr, tau_corr_ms, autocorr = estimate_correlation_time(signal, fs=fs, threshold=0.1)
        correlation_times[ch] = {'tau_samples': tau_corr, 'tau_ms': tau_corr_ms}
        
        # Compute PID within and beyond correlation time
        within_results[ch] = compute_pid_within_correlation(signal, tau_corr, fs=fs, n_bins=N_BINS)
        beyond_results[ch] = compute_pid_beyond_correlation(signal, tau_corr, fs=fs, n_bins=N_BINS)
        
        # Generate plots for first 4 channels
        if i < 4:
            # Autocorrelation plot
            plot_autocorrelation_analysis(
                signal, fs, ch.replace('eeg-', ''), tau_corr, autocorr,
                save_path=RESULTS_DIR / f"autocorr_{ch.replace('eeg-', '')}.png"
            )
            
            # Overlap comparison plot
            plot_overlap_comparison(
                within_results[ch], beyond_results[ch], tau_corr, fs,
                ch.replace('eeg-', ''),
                save_path=RESULTS_DIR / f"overlap_comparison_{ch.replace('eeg-', '')}.png"
            )
    
    # Print correlation time summary
    print("\n  Correlation times by channel:")
    for ch, times in correlation_times.items():
        print(f"    {ch.replace('eeg-', '')}: τ_corr = {times['tau_samples']} samples = {times['tau_ms']:.1f} ms")
    
    # Save correlation times
    corr_time_df = pd.DataFrame([
        {'channel': ch, 'tau_samples': v['tau_samples'], 'tau_ms': v['tau_ms']}
        for ch, v in correlation_times.items()
    ])
    corr_time_df.to_csv(RESULTS_DIR / "correlation_times.csv", index=False)
    
    # Summary comparison across all channels
    print("\n  Generating overlap summary...")
    
    within_summary = []
    beyond_summary = []
    for ch in channels:
        if len(within_results[ch]) > 0:
            within_summary.append({
                'channel': ch,
                'regime': 'within',
                'redundancy': within_results[ch]['redundancy'].mean(),
                'synergy': within_results[ch]['synergy'].mean(),
                'unique': within_results[ch]['unique_0'].mean() + within_results[ch]['unique_1'].mean()
            })
        if len(beyond_results[ch]) > 0:
            beyond_summary.append({
                'channel': ch,
                'regime': 'beyond',
                'redundancy': beyond_results[ch]['redundancy'].mean(),
                'synergy': beyond_results[ch]['synergy'].mean(),
                'unique': beyond_results[ch]['unique_0'].mean() + beyond_results[ch]['unique_1'].mean()
            })
    
    summary_df = pd.concat([pd.DataFrame(within_summary), pd.DataFrame(beyond_summary)])
    summary_df.to_csv(RESULTS_DIR / "overlap_comparison_summary.csv", index=False)
    
    # Generate summary plot for all channels
    plot_overlap_summary_all_channels(
        within_results, beyond_results, correlation_times,
        save_path=RESULTS_DIR / "overlap_summary_all_channels.png"
    )
    
    print("  Overlap analysis complete.")
    
    # =========================================================================
    # STRATEGY 2: AR(1) BASELINE COMPARISON
    # =========================================================================
    print(f"\n{'='*70}")
    print("STRATEGY 2: AR(1) BASELINE COMPARISON")
    print("="*70)
    print("Comparing actual PID to AR(1) surrogate to identify genuine structure...")
    
    all_excess_results = {}
    all_ar1_params = {}
    
    for i, ch in enumerate(channels):
        print(f"  Processing {i+1}/{len(channels)}: {ch}...")
        signal = df[ch].values[:SUBSAMPLE]
        signal = signal[~np.isnan(signal)]
        
        df_actual, df_surrogate, df_excess, ar1_params = compute_ar1_baseline_comparison(
            signal, max_lag=MAX_LAG, n_bins=N_BINS, n_surrogates=5
        )
        
        all_excess_results[ch] = df_excess
        all_ar1_params[ch] = ar1_params
        
        # Generate detailed plot for first 4 channels
        if i < 4:
            plot_ar_baseline_comparison(
                df_actual, df_surrogate, df_excess, ar1_params,
                ch.replace('eeg-', ''),
                save_path=RESULTS_DIR / f"ar_baseline_{ch.replace('eeg-', '')}.png"
            )
    
    # Summary plot of excess across all channels
    print("\nGenerating excess PID summary...")
    plot_excess_summary(all_excess_results, save_path=RESULTS_DIR / "excess_pid_summary.png")
    
    # Save excess results
    excess_combined = pd.concat([df.assign(channel=ch) for ch, df in all_excess_results.items()])
    excess_combined.to_csv(RESULTS_DIR / "excess_pid_all_channels.csv", index=False)
    
    # Print AR(1) parameters
    print("\nAR(1) coefficients (φ) by channel:")
    for ch, params in all_ar1_params.items():
        print(f"  {ch.replace('eeg-', '')}: φ = {params['phi']:.3f}")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print("="*70)
    
    summary = []
    for ch, df_ch in all_results.items():
        summary.append({
            'channel': ch.replace('eeg-', ''),
            'mean_redundancy': df_ch['redundancy'].mean(),
            'mean_synergy': df_ch['synergy'].mean(),
            'mean_unique': df_ch['unique_0'].mean() + df_ch['unique_1'].mean(),
            'syn_red_ratio': df_ch['synergy'].mean() / max(df_ch['redundancy'].mean(), 1e-10)
        })
    
    df_summary = pd.DataFrame(summary).sort_values('syn_red_ratio', ascending=False)
    print("\nChannels ranked by Synergy/Redundancy ratio:")
    print(df_summary.to_string(index=False))
    
    df_summary.to_csv(RESULTS_DIR / "channel_summary.csv", index=False)
    
    # Excess PID summary
    print("\n--- Excess PID (Genuine Structure Beyond AR(1)) ---")
    excess_summary = []
    for ch, df_excess in all_excess_results.items():
        excess_summary.append({
            'channel': ch.replace('eeg-', ''),
            'excess_redundancy': df_excess['redundancy'].mean(),
            'excess_synergy': df_excess['synergy'].mean(),
            'excess_unique': df_excess['unique_0'].mean() + df_excess['unique_1'].mean(),
            'ar1_phi': all_ar1_params[ch]['phi']
        })
    
    df_excess_summary = pd.DataFrame(excess_summary).sort_values('excess_synergy', ascending=False)
    print("\nChannels ranked by Excess Synergy (genuine non-AR structure):")
    print(df_excess_summary.to_string(index=False))
    
    df_excess_summary.to_csv(RESULTS_DIR / "channel_excess_summary.csv", index=False)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"\nFiles created:")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"  - {f.name}")
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
