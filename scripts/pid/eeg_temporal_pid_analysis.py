"""
EEG Temporal PID Analysis
=========================

Analyze the temporal structure of EEG signals using PID on time-delay embeddings.

For each channel:
- Discretize the continuous signal
- Compute PID for all lag pairs up to max_lag
- Extract redundancy, unique, synergy components
- Compare "temporal fingerprints" across brain regions

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
    SUBSAMPLE = 50000  # Use first 50k samples for speed (can increase later)
    
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
