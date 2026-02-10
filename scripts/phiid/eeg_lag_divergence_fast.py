"""
PhiID Lag Divergence Analysis - Real EEG Data (FAST VERSION)
==============================================================

Optimized for real-world EEG analysis:
- Fixed sampling rate (300 Hz for DSI-24)
- Uses subset of data for speed
- Progress tracking
- Proper parameter computation

Usage:
    python eeg_lag_divergence_fast.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import signal as sig
from scipy.stats import zscore
from tqdm import tqdm

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "eeg_divergence"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Atom names and groupings
ATOM_NAMES = list(PhiID_atoms_abbr)

# Information Dynamics metrics
DYNAMICS_GROUPS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Copy": ["xtx", "yty"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Downward_causation": ["sty", "stx", "str"],
    "Upward_causation": ["xts", "yts", "rts"],
}

# IIT-inspired metrics
IIT_METRICS = {
    "Info_storage": ["xtx", "yty", "rtr", "sts"],
    "Transfer_entropy": ["xty", "xtr", "str", "sty"],
    "Causal_density": ["xtr", "ytr", "sty", "str", "xty", "ytx", "stx"],
    "Integrated_info": ["rts", "xts", "sts", "sty", "str", "yts", "ytx", "stx", "xty"],
}


def load_eeg_data(filepath, fs=300, max_duration=60):
    """
    Load EEG data from CSV.
    
    Args:
        filepath: Path to CSV file
        fs: Sampling rate (Hz) - DSI-24 is ~300 Hz
        max_duration: Maximum seconds to load (for speed)
    """
    print(f"Loading EEG from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Get EEG columns (exclude timestamp and triggers)
    eeg_cols = [c for c in df.columns if c.startswith('eeg-') and 
                'trigger' not in c.lower() and 'A1' not in c and 'A2' not in c]
    
    # Limit data for speed
    max_samples = int(max_duration * fs)
    if len(df) > max_samples:
        print(f"  Using first {max_duration} seconds ({max_samples} samples)")
        df = df.iloc[:max_samples]
    
    print(f"  Channels: {eeg_cols}")
    print(f"  Samples: {len(df)}")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Duration: {len(df)/fs:.1f} seconds")
    
    return df[eeg_cols], fs, eeg_cols


def compute_phiid_timeseries(signal, extra_lag, tau, window_size, step_size, 
                              n_windows, kind='gaussian', desc=''):
    """
    Compute PhiID atoms over sliding windows with progress bar.
    """
    atom_series = {name: np.zeros(n_windows) for name in ATOM_NAMES}
    time_indices = np.zeros(n_windows)
    
    iterator = range(n_windows)
    if desc:
        iterator = tqdm(iterator, desc=desc, leave=False)
    
    for i in iterator:
        start = i * step_size
        end = start + window_size + extra_lag
        
        if end > len(signal):
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


def compute_divergence(atom_series_by_lag, n_times):
    """Compute divergence (std across lags) at each timepoint."""
    lags = sorted(atom_series_by_lag.keys())
    
    divergence = {name: np.zeros(n_times) for name in ATOM_NAMES}
    
    for name in ATOM_NAMES:
        values = np.array([atom_series_by_lag[lag][name] for lag in lags])
        divergence[name] = np.std(values, axis=0)
    
    # Compute for Information Dynamics groups
    dynamics_divergence = {}
    for group_name, atoms in DYNAMICS_GROUPS.items():
        group_values = []
        for lag in lags:
            group_sum = sum(atom_series_by_lag[lag][a] for a in atoms)
            group_values.append(group_sum)
        group_values = np.array(group_values)
        dynamics_divergence[group_name] = np.std(group_values, axis=0)
    
    # Compute for IIT metrics
    iit_divergence = {}
    for group_name, atoms in IIT_METRICS.items():
        group_values = []
        for lag in lags:
            group_sum = sum(atom_series_by_lag[lag][a] for a in atoms)
            group_values.append(group_sum)
        group_values = np.array(group_values)
        iit_divergence[group_name] = np.std(group_values, axis=0)
    
    return divergence, dynamics_divergence, iit_divergence


def plot_results(atom_series_by_lag, divergence, dynamics_divergence, iit_divergence,
                 time_indices, fs, channel, save_dir):
    """Generate all visualizations."""
    lags = sorted(atom_series_by_lag.keys())
    time_sec = time_indices / fs
    
    # Figure 1: Storage divergence analysis
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel A: Storage for each lag
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(lags)))
    for i, lag in enumerate(lags):
        storage = sum(atom_series_by_lag[lag][a] for a in DYNAMICS_GROUPS["Storage"])
        ax.plot(time_sec, storage, color=colors[i], alpha=0.7, 
                label=f'Lag {lag} ({lag/fs*1000:.0f}ms)', linewidth=1)
    ax.set_ylabel('Storage (bits)')
    ax.set_title(f'A) Storage Across Different Lags Over Time - {channel}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel B: Divergence
    ax = axes[1]
    ax.fill_between(time_sec, dynamics_divergence["Storage"], alpha=0.5, color='coral')
    ax.plot(time_sec, dynamics_divergence["Storage"], color='red', linewidth=1.5)
    ax.set_ylabel('Divergence (std)')
    ax.set_title('B) Storage Divergence Across Lags (High = Multi-scale, Low = Scale-invariant)')
    ax.grid(alpha=0.3)
    
    # Thresholds
    hi_thresh = np.percentile(dynamics_divergence["Storage"], 75)
    lo_thresh = np.percentile(dynamics_divergence["Storage"], 25)
    ax.axhline(hi_thresh, color='red', linestyle='--', alpha=0.5, label='High (75%)')
    ax.axhline(lo_thresh, color='green', linestyle='--', alpha=0.5, label='Low (25%)')
    ax.legend(loc='upper right')
    
    # Panel C: Mean with band
    ax = axes[2]
    mean_storage = np.mean([sum(atom_series_by_lag[lag][a] for a in DYNAMICS_GROUPS["Storage"]) 
                           for lag in lags], axis=0)
    std_storage = dynamics_divergence["Storage"]
    ax.plot(time_sec, mean_storage, color='blue', linewidth=1.5)
    ax.fill_between(time_sec, mean_storage - std_storage, mean_storage + std_storage, 
                   alpha=0.3, color='blue')
    ax.set_ylabel('Mean Storage +/- std')
    ax.set_xlabel('Time (s)')
    ax.set_title('C) Mean Storage with Divergence Band')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / f"storage_divergence_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 2: All Information Dynamics divergence (2x3 grid for 6 metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    
    for idx, (name, div_series) in enumerate(dynamics_divergence.items()):
        ax = axes[idx]
        ax.fill_between(time_sec, div_series, alpha=0.5)
        ax.plot(time_sec, div_series, linewidth=1.5)
        ax.set_ylabel('Divergence')
        ax.set_title(f'{name.replace("_", " ")}')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    plt.suptitle(f'Information Dynamics Divergence - {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"all_dynamics_divergence_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Correlation scatter (2x3 for 6 metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, (name, atoms) in enumerate(DYNAMICS_GROUPS.items()):
        ax = axes[idx]
        mean_values = np.mean([sum(atom_series_by_lag[lag][a] for a in atoms) 
                              for lag in lags], axis=0)
        div_values = dynamics_divergence[name]
        
        ax.scatter(mean_values, div_values, alpha=0.3, s=10)
        ax.set_xlabel(f'Mean {name.replace("_", " ")} (bits)')
        ax.set_ylabel('Divergence (std)')
        ax.set_title(name.replace("_", " "))
        ax.grid(alpha=0.3)
        
        valid = np.isfinite(mean_values) & np.isfinite(div_values)
        if np.sum(valid) > 10:
            corr = np.corrcoef(mean_values[valid], div_values[valid])[0, 1]
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                   fontsize=12, fontweight='bold', verticalalignment='top')
    
    plt.suptitle(f'Value vs Divergence - {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"value_vs_divergence_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Figure 4: IIT metrics divergence (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    
    for idx, (name, div_series) in enumerate(iit_divergence.items()):
        ax = axes[idx]
        ax.fill_between(time_sec, div_series, alpha=0.5, color='purple')
        ax.plot(time_sec, div_series, linewidth=1.5, color='darkviolet')
        ax.set_ylabel('Divergence')
        ax.set_title(f'{name.replace("_", " ")}')
        ax.grid(alpha=0.3)
        ax.set_xlabel('Time (s)')
    
    plt.suptitle(f'IIT Metrics Divergence - {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"iit_divergence_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()


def analyze_channel(signal, fs, channel_name, lags_samples, tau, 
                    window_samples, step_samples):
    """Full divergence analysis for one channel."""
    
    signal = zscore(signal[~np.isnan(signal)])
    n = len(signal)
    
    max_lag = max(lags_samples)
    n_windows = (n - window_samples - max_lag) // step_samples
    
    print(f"\n  Window: {window_samples} samples ({window_samples/fs*1000:.0f}ms)")
    print(f"  Step: {step_samples} samples ({step_samples/fs*1000:.0f}ms)")
    print(f"  Number of windows: {n_windows}")
    
    atom_series_by_lag = {}
    time_indices = None
    
    for lag in lags_samples:
        print(f"    Processing lag = {lag} samples ({lag/fs*1000:.1f}ms)...")
        atoms, times = compute_phiid_timeseries(
            signal, lag, tau, window_samples, step_samples, n_windows,
            kind='gaussian', desc=f'Lag {lag}'
        )
        atom_series_by_lag[lag] = atoms
        if time_indices is None:
            time_indices = times
    
    # Compute divergence
    divergence, dynamics_divergence, iit_divergence = compute_divergence(
        atom_series_by_lag, n_windows
    )
    
    return atom_series_by_lag, divergence, dynamics_divergence, iit_divergence, time_indices


def main():
    print("=" * 70)
    print("PhiID LAG DIVERGENCE - REAL EEG (FAST)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Find EEG file
    eeg_files = list(DATA_DIR.glob("*eeg*.csv")) + list(DATA_DIR.glob("*dsi*.csv"))
    
    if not eeg_files:
        print("ERROR: No EEG files found in data/")
        return
    
    eeg_file = eeg_files[0]
    
    # Parameters - FIXED for DSI-24
    FS = 300  # DSI-24 sampling rate
    MAX_DURATION = 30  # seconds to analyze (for speed)
    TAU = 1
    
    # Lags in samples (not ms!) - covering different timescales
    LAGS_SAMPLES = [3, 9, 15, 30, 60]  # ~10, 30, 50, 100, 200 ms at 300Hz
    WINDOW_SAMPLES = int(0.5 * FS)  # 500ms = 150 samples
    STEP_SAMPLES = int(0.1 * FS)    # 100ms = 30 samples
    
    # Channels to analyze
    CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    
    # Load data
    eeg_data, fs, all_channels = load_eeg_data(eeg_file, FS, MAX_DURATION)
    
    # Filter to available channels
    channels = [c for c in CHANNELS if c in all_channels]
    if not channels:
        channels = all_channels[:4]
    
    print(f"\nParameters:")
    print(f"  tau: {TAU}")
    print(f"  Lags (samples): {LAGS_SAMPLES}")
    print(f"  Lags (ms): {[int(l/FS*1000) for l in LAGS_SAMPLES]}")
    print(f"  Window: {WINDOW_SAMPLES} samples ({WINDOW_SAMPLES/FS*1000:.0f}ms)")
    print(f"  Step: {STEP_SAMPLES} samples ({STEP_SAMPLES/FS*1000:.0f}ms)")
    print(f"  Channels: {channels}")
    
    all_results = []
    
    for channel in channels:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {channel}")
        print("=" * 50)
        
        signal = eeg_data[channel].values
        
        atom_series_by_lag, divergence, dynamics_divergence, iit_divergence, time_indices = \
            analyze_channel(signal, FS, channel, LAGS_SAMPLES, TAU,
                          WINDOW_SAMPLES, STEP_SAMPLES)
        
        # Generate plots
        print(f"  Generating plots...")
        plot_results(atom_series_by_lag, divergence, dynamics_divergence, iit_divergence,
                    time_indices, FS, channel, RESULTS_DIR)
        
        # Save stats for Information Dynamics
        for name, div_series in dynamics_divergence.items():
            all_results.append({
                'channel': channel,
                'category': 'Info_Dynamics',
                'metric': name,
                'mean_divergence': np.mean(div_series),
                'std_divergence': np.std(div_series),
                'max_divergence': np.max(div_series),
                'min_divergence': np.min(div_series),
            })
        
        # Save stats for IIT metrics
        for name, div_series in iit_divergence.items():
            all_results.append({
                'channel': channel,
                'category': 'IIT',
                'metric': name,
                'mean_divergence': np.mean(div_series),
                'std_divergence': np.std(div_series),
                'max_divergence': np.max(div_series),
                'min_divergence': np.min(div_series),
            })
    
    # Summary
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(RESULTS_DIR / "divergence_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
