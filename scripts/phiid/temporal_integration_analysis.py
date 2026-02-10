"""
Temporal Integration Analysis - Advanced PhiID Insights
=========================================================

Goes beyond divergence plots to extract:
1. Temporal Integration Index (TII) - scalar measure of scale-invariance
2. Dominant Timescale Detection - which lag carries most information
3. Cross-Lag Coupling - how timescales interact
4. Integration State Segmentation - identify distinct temporal regimes
5. Complexity Metrics - entropy of multi-scale dynamics
6. Predictive Relationships - do short lags predict long lags?

Usage:
    python temporal_integration_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import signal as sig
from scipy.stats import zscore, entropy, pearsonr, spearmanr
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "temporal_integration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATOM_NAMES = list(PhiID_atoms_abbr)

# All metrics
DYNAMICS_GROUPS = {
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
    "Causal_density": ["xtr", "ytr", "sty", "str", "xty", "ytx", "stx"],
    "Integrated_info": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


def load_eeg_data(filepath, fs=300, max_duration=60):
    """Load EEG data."""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    eeg_cols = [c for c in df.columns if c.startswith('eeg-') and 
                'trigger' not in c.lower() and 'A1' not in c and 'A2' not in c]
    
    max_samples = int(max_duration * fs)
    if len(df) > max_samples:
        df = df.iloc[:max_samples]
    
    print(f"  {len(df)} samples, {len(df)/fs:.1f}s, channels: {len(eeg_cols)}")
    return df[eeg_cols], fs, eeg_cols


def compute_phiid_multilags(signal, lags, tau, window_size, step_size):
    """Compute PhiID for multiple lags efficiently."""
    n = len(signal)
    max_lag = max(lags)
    n_windows = (n - window_size - max_lag) // step_size
    
    # Store atom series for each lag
    results = {lag: {name: np.zeros(n_windows) for name in ATOM_NAMES} for lag in lags}
    time_indices = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_size
        time_indices[i] = start + window_size // 2
        
        for lag in lags:
            end = start + window_size + lag
            if end > n:
                continue
            
            segment = signal[start:end]
            src = segment[:-lag]
            tgt = segment[lag:]
            
            try:
                atoms_res, _ = calc_PhiID(src, tgt, tau, kind='gaussian', redundancy='MMI')
                for name in ATOM_NAMES:
                    if name in atoms_res:
                        val = np.nanmean(atoms_res[name])
                        results[lag][name][i] = val if np.isfinite(val) else 0.0
            except:
                pass
    
    return results, time_indices, n_windows


def compute_metric_series(atom_series_by_lag, metric_atoms):
    """Compute a grouped metric (sum of atoms) for each lag."""
    lags = sorted(atom_series_by_lag.keys())
    n_times = len(atom_series_by_lag[lags[0]][ATOM_NAMES[0]])
    
    metric_by_lag = {}
    for lag in lags:
        metric_by_lag[lag] = sum(atom_series_by_lag[lag][a] for a in metric_atoms)
    
    return metric_by_lag


# =============================================================================
# ADVANCED ANALYSIS FUNCTIONS
# =============================================================================

def compute_temporal_integration_index(metric_by_lag):
    """
    Temporal Integration Index (TII):
    TII = 1 / (1 + normalized_divergence)
    
    High TII (→1): Scale-invariant, unified temporal dynamics
    Low TII (→0): Multi-scale, fragmented dynamics
    """
    lags = sorted(metric_by_lag.keys())
    values = np.array([metric_by_lag[lag] for lag in lags])
    
    # Divergence = std across lags at each time
    divergence = np.std(values, axis=0)
    mean_value = np.mean(values, axis=0)
    
    # Coefficient of variation (normalized divergence)
    cv = np.where(np.abs(mean_value) > 0.01, 
                  divergence / np.abs(mean_value), 
                  divergence)
    
    # TII = 1 / (1 + CV)
    tii = 1 / (1 + cv)
    
    return tii, divergence, mean_value


def compute_dominant_timescale(metric_by_lag, time_indices):
    """
    Find which lag dominates at each timepoint.
    Returns: dominant_lag array, dominance_strength
    """
    lags = sorted(metric_by_lag.keys())
    values = np.array([metric_by_lag[lag] for lag in lags])
    
    # Dominant = lag with highest absolute value
    dominant_idx = np.argmax(np.abs(values), axis=0)
    dominant_lag = np.array([lags[i] for i in dominant_idx])
    
    # Dominance strength = how much stronger than mean
    max_val = np.max(np.abs(values), axis=0)
    mean_val = np.mean(np.abs(values), axis=0)
    dominance_strength = np.where(mean_val > 0.01, max_val / mean_val, 1.0)
    
    return dominant_lag, dominance_strength


def compute_cross_lag_coupling(metric_by_lag):
    """
    Compute correlation between different lags over time.
    Returns: coupling matrix, mean coupling strength
    """
    lags = sorted(metric_by_lag.keys())
    n_lags = len(lags)
    
    coupling_matrix = np.zeros((n_lags, n_lags))
    
    for i, lag1 in enumerate(lags):
        for j, lag2 in enumerate(lags):
            if i != j:
                valid = (np.isfinite(metric_by_lag[lag1]) & 
                        np.isfinite(metric_by_lag[lag2]))
                if np.sum(valid) > 10:
                    r, _ = pearsonr(metric_by_lag[lag1][valid], 
                                   metric_by_lag[lag2][valid])
                    coupling_matrix[i, j] = r if np.isfinite(r) else 0
            else:
                coupling_matrix[i, j] = 1.0
    
    # Mean coupling (off-diagonal)
    mask = ~np.eye(n_lags, dtype=bool)
    mean_coupling = np.mean(coupling_matrix[mask])
    
    return coupling_matrix, mean_coupling, lags


def compute_lag_gradient(metric_by_lag):
    """
    Compute how metric changes across lags (short→long).
    Positive gradient: increases with lag
    Negative gradient: decreases with lag
    """
    lags = sorted(metric_by_lag.keys())
    values = np.array([metric_by_lag[lag] for lag in lags])
    
    # Log-lag for proper timescale comparison
    log_lags = np.log(lags)
    
    # Linear regression slope at each timepoint
    n_times = values.shape[1]
    gradients = np.zeros(n_times)
    
    for t in range(n_times):
        y = values[:, t]
        if np.all(np.isfinite(y)):
            # Simple linear regression
            slope = np.polyfit(log_lags, y, 1)[0]
            gradients[t] = slope
    
    return gradients


def segment_integration_states(tii, n_states=3):
    """
    Segment time series into distinct integration states using clustering.
    """
    # Prepare features: TII and its derivative
    tii_smooth = uniform_filter1d(tii, size=5)
    tii_diff = np.gradient(tii_smooth)
    
    features = np.column_stack([tii_smooth, tii_diff])
    valid = np.all(np.isfinite(features), axis=1)
    
    if np.sum(valid) < n_states * 10:
        return np.zeros(len(tii), dtype=int), None
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features[valid])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    labels_valid = kmeans.fit_predict(features_scaled)
    
    # Map back to full time series
    labels = np.zeros(len(tii), dtype=int)
    labels[valid] = labels_valid
    
    # Compute state characteristics
    state_info = {}
    for state in range(n_states):
        mask = labels == state
        state_info[state] = {
            'mean_tii': np.mean(tii[mask]),
            'fraction': np.mean(mask),
        }
    
    # Reorder states by mean TII (0=low, n-1=high)
    state_order = sorted(state_info.keys(), key=lambda x: state_info[x]['mean_tii'])
    label_map = {old: new for new, old in enumerate(state_order)}
    labels = np.array([label_map[l] for l in labels])
    
    return labels, state_info


def compute_complexity_metrics(metric_by_lag, tii):
    """
    Compute complexity measures of temporal dynamics.
    """
    lags = sorted(metric_by_lag.keys())
    values = np.array([metric_by_lag[lag] for lag in lags])
    
    n_times = values.shape[1]
    
    # 1. Spectral entropy of lag distribution at each time
    spectral_entropy = np.zeros(n_times)
    for t in range(n_times):
        dist = np.abs(values[:, t])
        dist = dist / (np.sum(dist) + 1e-10)
        spectral_entropy[t] = entropy(dist + 1e-10)
    
    # 2. TII entropy (how variable is integration over time?)
    tii_binned = np.digitize(tii, bins=np.linspace(0, 1, 11)) - 1
    tii_hist = np.bincount(tii_binned, minlength=10) / len(tii)
    tii_entropy = entropy(tii_hist + 1e-10)
    
    # 3. Transition entropy (how often does dominant scale change?)
    dominant_lag, _ = compute_dominant_timescale(metric_by_lag, None)
    transitions = np.diff(dominant_lag) != 0
    transition_rate = np.mean(transitions)
    
    return {
        'spectral_entropy': spectral_entropy,
        'tii_entropy': tii_entropy,
        'transition_rate': transition_rate,
        'mean_spectral_entropy': np.mean(spectral_entropy),
    }


def compute_predictive_relationships(metric_by_lag, max_pred_lag=10):
    """
    Can short-lag dynamics predict long-lag dynamics?
    Computes lagged correlations.
    """
    lags = sorted(metric_by_lag.keys())
    shortest = lags[0]
    longest = lags[-1]
    
    short_series = metric_by_lag[shortest]
    long_series = metric_by_lag[longest]
    
    # Cross-correlation at different temporal offsets
    pred_correlations = []
    for offset in range(-max_pred_lag, max_pred_lag + 1):
        if offset < 0:
            s = short_series[-offset:]
            l = long_series[:offset]
        elif offset > 0:
            s = short_series[:-offset]
            l = long_series[offset:]
        else:
            s, l = short_series, long_series
        
        valid = np.isfinite(s) & np.isfinite(l)
        if np.sum(valid) > 10:
            r, _ = pearsonr(s[valid], l[valid])
            pred_correlations.append({'offset': offset, 'correlation': r})
    
    df_pred = pd.DataFrame(pred_correlations)
    
    # Best predictive offset
    if len(df_pred) > 0:
        best_idx = df_pred['correlation'].abs().idxmax()
        best_offset = df_pred.loc[best_idx, 'offset']
        best_corr = df_pred.loc[best_idx, 'correlation']
    else:
        best_offset, best_corr = 0, 0
    
    return df_pred, best_offset, best_corr


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_integration_analysis(results, channel, fs, save_dir):
    """Comprehensive integration analysis plot."""
    
    time_sec = results['time_indices'] / fs
    
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Row 1: TII time series and states
    ax1 = fig.add_subplot(gs[0, :2])
    tii = results['tii']
    states = results['states']
    colors = ['red', 'yellow', 'green']  # Low, medium, high integration
    for state in range(3):
        mask = states == state
        ax1.fill_between(time_sec, 0, 1, where=mask, alpha=0.3, 
                        color=colors[state], label=f'State {state}')
    ax1.plot(time_sec, tii, 'k-', linewidth=1.5, label='TII')
    ax1.set_ylabel('TII')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Temporal Integration Index (TII) with States')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # Row 1 right: TII distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(tii, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    ax2.axvline(np.mean(tii), color='red', linestyle='--', label=f'Mean: {np.mean(tii):.3f}')
    ax2.set_xlabel('TII')
    ax2.set_ylabel('Count')
    ax2.set_title('TII Distribution')
    ax2.legend()
    
    # Row 2: Dominant timescale
    ax3 = fig.add_subplot(gs[1, :2])
    dom_lag = results['dominant_lag']
    ax3.scatter(time_sec, dom_lag, c=results['dominance_strength'], 
               cmap='hot', s=10, alpha=0.7)
    ax3.set_ylabel('Dominant Lag (samples)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Dominant Timescale Over Time (color = strength)')
    ax3.grid(alpha=0.3)
    
    # Row 2 right: Lag gradient
    ax4 = fig.add_subplot(gs[1, 2])
    gradient = results['gradient']
    ax4.hist(gradient, bins=30, color='coral', edgecolor='white', alpha=0.7)
    ax4.axvline(0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Lag Gradient')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Gradient Distribution\n(+: increases with lag)')
    
    # Row 3: Cross-lag coupling matrix
    ax5 = fig.add_subplot(gs[2, 0])
    coupling = results['coupling_matrix']
    lags = results['lags']
    im = ax5.imshow(coupling, cmap='RdBu_r', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(lags)))
    ax5.set_yticks(range(len(lags)))
    ax5.set_xticklabels([f'{l}' for l in lags])
    ax5.set_yticklabels([f'{l}' for l in lags])
    ax5.set_xlabel('Lag (samples)')
    ax5.set_ylabel('Lag (samples)')
    ax5.set_title(f'Cross-Lag Coupling\n(mean r = {results["mean_coupling"]:.3f})')
    plt.colorbar(im, ax=ax5)
    
    # Row 3: Spectral entropy over time
    ax6 = fig.add_subplot(gs[2, 1:])
    se = results['complexity']['spectral_entropy']
    ax6.fill_between(time_sec, se, alpha=0.5, color='purple')
    ax6.plot(time_sec, se, color='darkviolet', linewidth=1)
    ax6.set_ylabel('Spectral Entropy')
    ax6.set_xlabel('Time (s)')
    ax6.set_title(f'Lag Distribution Entropy (mean: {results["complexity"]["mean_spectral_entropy"]:.3f})')
    ax6.grid(alpha=0.3)
    
    # Row 4: Predictive relationships
    ax7 = fig.add_subplot(gs[3, 0])
    df_pred = results['predictive']['df']
    ax7.bar(df_pred['offset'], df_pred['correlation'], color='teal', alpha=0.7)
    ax7.axhline(0, color='black', linewidth=1)
    ax7.axvline(results['predictive']['best_offset'], color='red', linestyle='--',
               label=f'Best: {results["predictive"]["best_offset"]}')
    ax7.set_xlabel('Temporal Offset')
    ax7.set_ylabel('Correlation (short→long lag)')
    ax7.set_title('Cross-Scale Prediction')
    ax7.legend()
    
    # Row 4: Summary metrics
    ax8 = fig.add_subplot(gs[3, 1:])
    ax8.axis('off')
    
    summary_text = f"""
    TEMPORAL INTEGRATION SUMMARY: {channel}
    {'='*50}
    
    INTEGRATION INDEX (TII)
      Mean TII:        {np.mean(tii):.4f}
      TII Std:         {np.std(tii):.4f}
      TII Entropy:     {results['complexity']['tii_entropy']:.4f}
    
    MULTI-SCALE DYNAMICS
      Mean Coupling:   {results['mean_coupling']:.4f}
      Transition Rate: {results['complexity']['transition_rate']:.4f}
      Spectral Entropy: {results['complexity']['mean_spectral_entropy']:.4f}
    
    PREDICTIVE STRUCTURE
      Best Offset:     {results['predictive']['best_offset']} windows
      Best Correlation: {results['predictive']['best_corr']:.4f}
    
    INTERPRETATION
    • High TII (→1): Unified dynamics across timescales
    • High Coupling: Timescales are synchronized
    • High Transition Rate: Rapidly switching dominance
    • Positive Best Offset: Short lags lead long lags
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Temporal Integration Analysis: {channel}', fontsize=14, fontweight='bold')
    plt.savefig(save_dir / f"integration_analysis_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_summary(all_channel_results, fs, save_dir):
    """Compare integration metrics across channels."""
    
    channels = list(all_channel_results.keys())
    
    # Extract key metrics for each channel
    summary_data = []
    for ch in channels:
        r = all_channel_results[ch]
        summary_data.append({
            'channel': ch,
            'mean_tii': np.mean(r['tii']),
            'std_tii': np.std(r['tii']),
            'tii_entropy': r['complexity']['tii_entropy'],
            'mean_coupling': r['mean_coupling'],
            'transition_rate': r['complexity']['transition_rate'],
            'spectral_entropy': r['complexity']['mean_spectral_entropy'],
            'predictive_offset': r['predictive']['best_offset'],
            'predictive_corr': r['predictive']['best_corr'],
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(save_dir / "integration_summary.csv", index=False)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    metrics = ['mean_tii', 'mean_coupling', 'transition_rate', 
               'spectral_entropy', 'tii_entropy', 'predictive_corr']
    titles = ['Mean TII', 'Cross-Lag Coupling', 'Transition Rate',
              'Spectral Entropy', 'TII Entropy', 'Predictive Correlation']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes.flatten()[idx]
        values = df_summary[metric].values
        ax.bar(channels, values, color=plt.cm.Set2(np.arange(len(channels))))
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Temporal Integration Comparison Across Channels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "channel_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return df_summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TEMPORAL INTEGRATION ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Find EEG file
    eeg_files = list(DATA_DIR.glob("*eeg*.csv")) + list(DATA_DIR.glob("*dsi*.csv"))
    if not eeg_files:
        print("ERROR: No EEG files found")
        return
    
    eeg_file = eeg_files[0]
    
    # Parameters
    FS = 300
    MAX_DURATION = 30  # seconds
    TAU = 1
    LAGS = [3, 9, 15, 30, 60]  # ~10, 30, 50, 100, 200 ms
    WINDOW = int(0.5 * FS)  # 500ms
    STEP = int(0.1 * FS)    # 100ms
    
    CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    
    # Load data
    eeg_data, fs, all_channels = load_eeg_data(eeg_file, FS, MAX_DURATION)
    channels = [c for c in CHANNELS if c in all_channels]
    if not channels:
        channels = all_channels[:4]
    
    print(f"\nParameters: tau={TAU}, lags={LAGS}, window={WINDOW}, step={STEP}")
    print(f"Channels: {channels}")
    
    all_channel_results = {}
    
    for channel in channels:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {channel}")
        print("=" * 50)
        
        signal = zscore(eeg_data[channel].dropna().values)
        
        # Compute PhiID for all lags
        print("  Computing PhiID across lags...")
        atom_series_by_lag, time_indices, n_windows = compute_phiid_multilags(
            signal, LAGS, TAU, WINDOW, STEP
        )
        
        # Get Storage metric for main analysis
        storage_by_lag = compute_metric_series(
            atom_series_by_lag, DYNAMICS_GROUPS["Storage"]
        )
        
        # Compute all derived metrics
        print("  Computing integration metrics...")
        
        # 1. TII
        tii, divergence, mean_value = compute_temporal_integration_index(storage_by_lag)
        
        # 2. Dominant timescale
        dominant_lag, dominance_strength = compute_dominant_timescale(storage_by_lag, time_indices)
        
        # 3. Cross-lag coupling
        coupling_matrix, mean_coupling, lags_list = compute_cross_lag_coupling(storage_by_lag)
        
        # 4. Gradient
        gradient = compute_lag_gradient(storage_by_lag)
        
        # 5. State segmentation
        states, state_info = segment_integration_states(tii, n_states=3)
        
        # 6. Complexity
        complexity = compute_complexity_metrics(storage_by_lag, tii)
        
        # 7. Predictive relationships
        df_pred, best_offset, best_corr = compute_predictive_relationships(storage_by_lag)
        
        # Store results
        results = {
            'time_indices': time_indices,
            'tii': tii,
            'divergence': divergence,
            'mean_value': mean_value,
            'dominant_lag': dominant_lag,
            'dominance_strength': dominance_strength,
            'coupling_matrix': coupling_matrix,
            'mean_coupling': mean_coupling,
            'lags': LAGS,
            'gradient': gradient,
            'states': states,
            'state_info': state_info,
            'complexity': complexity,
            'predictive': {'df': df_pred, 'best_offset': best_offset, 'best_corr': best_corr},
        }
        
        all_channel_results[channel] = results
        
        # Generate plots
        print("  Generating visualization...")
        plot_integration_analysis(results, channel, FS, RESULTS_DIR)
    
    # Cross-channel comparison
    print("\nGenerating comparison summary...")
    df_summary = create_comparison_summary(all_channel_results, FS, RESULTS_DIR)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df_summary.round(4).to_string(index=False))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
