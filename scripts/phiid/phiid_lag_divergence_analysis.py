"""
PhiID Lag Divergence Analysis
==============================

Explores when PhiID metrics across different lags DIVERGE (spread apart) 
vs CONVERGE (similar values) over time.

Key insight: If metrics for different lags have similar values at time t,
the signal structure is "scale-invariant" at that moment. If they diverge,
the signal has distinct temporal scales active.

This could reveal:
- Moments of high vs low temporal integration
- State transitions in the signal
- Frequency-band specific dynamics

Usage:
    python phiid_lag_divergence_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import signal as sig
from scipy.stats import zscore

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "lag_divergence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Atom names and groupings
ATOM_NAMES = list(PhiID_atoms_abbr)

DYNAMICS_GROUPS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Integrated": ["rts", "str", "sts"],  # synergy-related
}


def load_eeg_data():
    """Load EEG data from results or generate sample."""
    # Try to load from common EEG data location
    data_paths = [
        PROJECT_DIR / "data" / "eeg" / "eeg_sample.npy",
        PROJECT_DIR / "data" / "eeg_sample.npy",
    ]
    
    for path in data_paths:
        if path.exists():
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype == object:
                data = data.item()
            return data
    
    # Generate synthetic EEG-like data
    print("Generating synthetic EEG-like data...")
    np.random.seed(42)
    fs = 256
    duration = 60  # seconds
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # Create signal with multiple rhythms and transient events
    # Base 1/f noise
    freqs = np.fft.fftfreq(n_samples, 1/fs)
    pink_spectrum = np.where(np.abs(freqs) > 0, 1/np.sqrt(np.abs(freqs)), 0)
    pink_noise = np.real(np.fft.ifft(pink_spectrum * np.exp(2j * np.pi * np.random.rand(n_samples))))
    
    # Alpha rhythm (10 Hz) - amplitude modulated
    alpha_envelope = 1 + 0.5 * np.sin(2 * np.pi * 0.1 * t)  # 0.1 Hz modulation
    alpha = alpha_envelope * np.sin(2 * np.pi * 10 * t)
    
    # Beta bursts (20 Hz) - intermittent
    beta_on = (np.sin(2 * np.pi * 0.05 * t) > 0.5).astype(float)  # On/off
    beta = beta_on * np.sin(2 * np.pi * 20 * t) * 0.5
    
    # Delta waves (2 Hz)
    delta = 2 * np.sin(2 * np.pi * 2 * t)
    
    # Combine with varying weights over time
    signal = pink_noise + alpha + beta + delta
    signal = zscore(signal)
    
    return {
        'data': signal,
        'fs': fs,
        'channel': 'synthetic'
    }


def compute_phiid_timeseries(signal, extra_lag, tau, window_size, step_size, max_lag, kind='gaussian'):
    """
    Compute PhiID atoms over sliding windows.
    
    Returns time series of each atom.
    
    max_lag: The maximum lag being used (to ensure consistent window count)
    """
    n = len(signal)
    # Use max_lag to ensure consistent number of windows across all lags
    n_windows = (n - window_size - max_lag) // step_size
    
    # Initialize storage
    atom_series = {name: np.zeros(n_windows) for name in ATOM_NAMES}
    time_indices = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size + extra_lag
        
        if end > n:
            break
            
        segment = signal[start:end]
        src = segment[:-extra_lag]
        tgt = segment[extra_lag:]
        
        try:
            atoms_res, _ = calc_PhiID(src, tgt, tau, kind=kind, redundancy='MMI')
            
            for name in ATOM_NAMES:
                if name in atoms_res:
                    val = np.nanmean(atoms_res[name])
                    atom_series[name][i] = val if np.isfinite(val) else 0.0
                    
        except Exception:
            pass
        
        time_indices[i] = start + window_size // 2
    
    return atom_series, time_indices


def compute_divergence_metrics(atom_series_by_lag, time_indices):
    """
    Compute how much different lags diverge at each timepoint.
    
    Divergence = std across lags for each atom at each time
    Convergence = 1 / (1 + divergence)
    """
    lags = sorted(atom_series_by_lag.keys())
    n_times = len(time_indices)
    
    divergence = {name: np.zeros(n_times) for name in ATOM_NAMES}
    
    for name in ATOM_NAMES:
        # Stack values from all lags: shape (n_lags, n_times)
        values = np.array([atom_series_by_lag[lag][name] for lag in lags])
        # Std across lags at each time
        divergence[name] = np.std(values, axis=0)
    
    # Also compute for dynamics groups
    dynamics_divergence = {}
    for group_name, atoms in DYNAMICS_GROUPS.items():
        group_values = []
        for lag in lags:
            group_sum = sum(atom_series_by_lag[lag][a] for a in atoms)
            group_values.append(group_sum)
        group_values = np.array(group_values)
        dynamics_divergence[group_name] = np.std(group_values, axis=0)
    
    return divergence, dynamics_divergence


def analyze_divergence_states(divergence_series, threshold_percentile=75):
    """
    Identify periods of high divergence (spread) vs low divergence (convergence).
    """
    high_threshold = np.percentile(divergence_series, threshold_percentile)
    low_threshold = np.percentile(divergence_series, 100 - threshold_percentile)
    
    high_divergence_mask = divergence_series > high_threshold
    low_divergence_mask = divergence_series < low_threshold
    
    return high_divergence_mask, low_divergence_mask, high_threshold, low_threshold


def plot_divergence_analysis(atom_series_by_lag, divergence, dynamics_divergence, 
                             time_indices, fs, save_dir):
    """Create comprehensive divergence visualization."""
    
    lags = sorted(atom_series_by_lag.keys())
    time_sec = time_indices / fs
    
    # Figure 1: Raw atom values for each lag (for Storage dynamics)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel A: Storage for each lag
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lags)))
    for i, lag in enumerate(lags):
        storage = sum(atom_series_by_lag[lag][a] for a in DYNAMICS_GROUPS["Storage"])
        ax.plot(time_sec, storage, color=colors[i], alpha=0.7, 
                label=f'Lag {lag}', linewidth=1)
    ax.set_ylabel('Storage (bits)')
    ax.set_title('A) Storage Across Different Lags Over Time')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    # Panel B: Divergence (spread across lags)
    ax = axes[1]
    ax.fill_between(time_sec, dynamics_divergence["Storage"], alpha=0.5, color='coral')
    ax.plot(time_sec, dynamics_divergence["Storage"], color='red', linewidth=1.5)
    ax.set_ylabel('Divergence (std)')
    ax.set_title('B) Storage Divergence Across Lags (High = Spread, Low = Converged)')
    ax.grid(alpha=0.3)
    
    # Mark high/low divergence periods
    high_mask, low_mask, hi_thresh, lo_thresh = analyze_divergence_states(
        dynamics_divergence["Storage"])
    ax.axhline(hi_thresh, color='red', linestyle='--', alpha=0.5, label='High threshold')
    ax.axhline(lo_thresh, color='green', linestyle='--', alpha=0.5, label='Low threshold')
    ax.legend(loc='upper right')
    
    # Panel C: Signal itself (if available)
    ax = axes[2]
    # We don't have the raw signal here, so plot the mean across lags
    mean_storage = np.mean([sum(atom_series_by_lag[lag][a] for a in DYNAMICS_GROUPS["Storage"]) 
                           for lag in lags], axis=0)
    ax.plot(time_sec, mean_storage, color='blue', linewidth=1)
    ax.fill_between(time_sec, mean_storage - dynamics_divergence["Storage"],
                   mean_storage + dynamics_divergence["Storage"], alpha=0.3, color='blue')
    ax.set_ylabel('Mean Storage +/- std')
    ax.set_xlabel('Time (s)')
    ax.set_title('C) Mean Storage with Divergence Band')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "storage_divergence_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: All dynamics divergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, (name, div_series) in enumerate(dynamics_divergence.items()):
        ax = axes[idx]
        ax.fill_between(time_sec, div_series, alpha=0.5)
        ax.plot(time_sec, div_series, linewidth=1.5)
        ax.set_ylabel('Divergence')
        ax.set_title(f'{name} Divergence')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    plt.suptitle('Divergence Across Lags for Different Information Dynamics', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "all_dynamics_divergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Scatter - does high divergence correlate with high/low absolute values?
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, atoms) in enumerate(DYNAMICS_GROUPS.items()):
        ax = axes[idx]
        mean_values = np.mean([sum(atom_series_by_lag[lag][a] for a in atoms) 
                              for lag in lags], axis=0)
        div_values = dynamics_divergence[name]
        
        ax.scatter(mean_values, div_values, alpha=0.3, s=10)
        ax.set_xlabel(f'Mean {name} (bits)')
        ax.set_ylabel('Divergence (std across lags)')
        ax.set_title(name)
        ax.grid(alpha=0.3)
        
        # Add correlation
        valid = np.isfinite(mean_values) & np.isfinite(div_values)
        if np.sum(valid) > 10:
            corr = np.corrcoef(mean_values[valid], div_values[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top')
    
    plt.suptitle('Relationship: Mean Value vs Divergence Across Lags', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "value_vs_divergence_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Divergence heatmap across all atoms
    fig, ax = plt.subplots(figsize=(14, 6))
    
    div_matrix = np.array([divergence[name] for name in ATOM_NAMES])
    
    # Subsample for visualization
    step = max(1, len(time_sec) // 100)
    div_subsampled = div_matrix[:, ::step]
    time_subsampled = time_sec[::step]
    
    im = ax.imshow(div_subsampled, aspect='auto', cmap='hot',
                   extent=[time_subsampled[0], time_subsampled[-1], 0, len(ATOM_NAMES)])
    ax.set_yticks(np.arange(len(ATOM_NAMES)) + 0.5)
    ax.set_yticklabels(ATOM_NAMES)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PhiID Atom')
    ax.set_title('Divergence Across Lags for Each Atom Over Time')
    plt.colorbar(im, ax=ax, label='Divergence (std)')
    
    plt.tight_layout()
    plt.savefig(save_dir / "atom_divergence_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_convergence_states_comparison(atom_series_by_lag, divergence, dynamics_divergence,
                                       time_indices, fs, save_dir):
    """
    Compare what the signal looks like during convergent vs divergent states.
    """
    lags = sorted(atom_series_by_lag.keys())
    time_sec = time_indices / fs
    
    # Get high/low divergence periods for Storage
    storage_div = dynamics_divergence["Storage"]
    high_mask, low_mask, _, _ = analyze_divergence_states(storage_div)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Divergent periods (spread)
    for col, (name, atoms) in enumerate(list(DYNAMICS_GROUPS.items())[:3]):
        ax = axes[0, col]
        for lag in lags:
            values = sum(atom_series_by_lag[lag][a] for a in atoms)
            divergent_vals = values[high_mask]
            if len(divergent_vals) > 0:
                ax.hist(divergent_vals, bins=30, alpha=0.5, label=f'Lag {lag}', density=True)
        ax.set_xlabel(f'{name} (bits)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} - DIVERGENT periods')
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    # Row 2: Convergent periods
    for col, (name, atoms) in enumerate(list(DYNAMICS_GROUPS.items())[:3]):
        ax = axes[1, col]
        for lag in lags:
            values = sum(atom_series_by_lag[lag][a] for a in atoms)
            convergent_vals = values[low_mask]
            if len(convergent_vals) > 0:
                ax.hist(convergent_vals, bins=30, alpha=0.5, label=f'Lag {lag}', density=True)
        ax.set_xlabel(f'{name} (bits)')
        ax.set_ylabel('Density')
        ax.set_title(f'{name} - CONVERGENT periods')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Distribution of Metrics During Divergent vs Convergent States', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / "convergent_vs_divergent_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary statistics
    summary = []
    for name, atoms in DYNAMICS_GROUPS.items():
        for lag in lags:
            values = sum(atom_series_by_lag[lag][a] for a in atoms)
            summary.append({
                'metric': name,
                'lag': lag,
                'divergent_mean': np.mean(values[high_mask]) if np.sum(high_mask) > 0 else np.nan,
                'divergent_std': np.std(values[high_mask]) if np.sum(high_mask) > 0 else np.nan,
                'convergent_mean': np.mean(values[low_mask]) if np.sum(low_mask) > 0 else np.nan,
                'convergent_std': np.std(values[low_mask]) if np.sum(low_mask) > 0 else np.nan,
            })
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(save_dir / "divergent_vs_convergent_stats.csv", index=False)
    
    return df_summary


def main():
    print("=" * 70)
    print("PhiID LAG DIVERGENCE ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    eeg_data = load_eeg_data()
    signal = eeg_data['data']
    fs = eeg_data['fs']
    channel = eeg_data.get('channel', 'unknown')
    
    print(f"\nData: {len(signal)} samples, {fs} Hz, {len(signal)/fs:.1f} seconds")
    print(f"Channel: {channel}")
    
    # Parameters
    TAU = 1
    LAGS = [2, 5, 10, 20, 40]  # Multiple lags to compare
    WINDOW_SIZE = int(0.5 * fs)  # 500ms windows
    STEP_SIZE = int(0.1 * fs)    # 100ms steps
    
    print(f"\nParameters:")
    print(f"  tau: {TAU}")
    print(f"  Lags to compare: {LAGS}")
    print(f"  Window: {WINDOW_SIZE/fs*1000:.0f}ms, Step: {STEP_SIZE/fs*1000:.0f}ms")
    
    # Compute PhiID for each lag
    atom_series_by_lag = {}
    time_indices = None
    max_lag = max(LAGS)  # Use max lag to ensure consistent window count
    
    for lag in LAGS:
        print(f"\nProcessing lag = {lag} samples ({lag/fs*1000:.1f} ms)...")
        atoms, times = compute_phiid_timeseries(
            signal, lag, TAU, WINDOW_SIZE, STEP_SIZE, max_lag, kind='gaussian'
        )
        atom_series_by_lag[lag] = atoms
        if time_indices is None:
            time_indices = times
    
    print(f"\nComputed {len(time_indices)} time windows")
    
    # Compute divergence
    print("\nComputing divergence metrics...")
    divergence, dynamics_divergence = compute_divergence_metrics(
        atom_series_by_lag, time_indices
    )
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_divergence_analysis(
        atom_series_by_lag, divergence, dynamics_divergence,
        time_indices, fs, RESULTS_DIR
    )
    
    summary_df = plot_convergence_states_comparison(
        atom_series_by_lag, divergence, dynamics_divergence,
        time_indices, fs, RESULTS_DIR
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("DIVERGENT vs CONVERGENT STATE COMPARISON")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Key insight summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
    HIGH DIVERGENCE (spread across lags) indicates:
    - Different temporal scales are behaving differently
    - The signal has multi-scale structure at this moment
    - Possible state transition or event processing
    
    LOW DIVERGENCE (convergence across lags) indicates:
    - All temporal scales show similar information dynamics
    - Scale-invariant behavior (like 1/f noise)
    - Stationary / stable dynamics
    
    This can be used to:
    - Detect state transitions
    - Identify moments of scale-specific processing
    - Characterize signal complexity
    """)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
