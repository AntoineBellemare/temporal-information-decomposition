"""
Cross-Biosignal Temporal PID Comparison
========================================

Compare temporal information structure across different biosignal types
using normalized timescales (τ-relative lags).

This script:
1. Loads multiple biosignal types (EEG, ECG, Respiration)
2. Auto-estimates characteristic timescale (τ) for each
3. Computes PID at τ-normalized lags for fair comparison
4. Generates "temporal information fingerprints" for each signal type

The key insight: by normalizing to each signal's intrinsic timescale,
we can meaningfully compare how information integrates across time
in fundamentally different physiological systems.

Usage:
    python biosignal_temporal_pid_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from scipy import signal as sig

# Use pip-installed dit (works with get_pi() API)
import dit
from dit.pid import PID_MMI
from dit import Distribution

# Setup paths
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# Import tau estimation
sys.path.insert(0, str(SCRIPT_DIR.parent))  # Add scripts/ to path for utils
from utils.estimate_tau import TauEstimator, estimate_tau, create_normalized_lags

# Configuration - already set above
# SCRIPT_DIR and PROJECT_DIR defined when adding dit to path
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "biosignal_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Signal type configurations
SIGNAL_CONFIGS = {
    'EEG': {
        'columns': ['EEG-ch1', 'EEG-ch2', 'EEG-ch3', 'EEG-ch4'],  # Sample channels
        'tau_method': 'autocorr',  # Use autocorrelation decay (faster, neural timescale)
        'color': '#1f77b4',
        'expected_tau_range': (0.01, 0.2),  # 10ms - 200ms
    },
    'ECG': {
        'columns': ['ecg'],
        'tau_method': 'autocorr',  # Use autocorr for aperiodic structure
        'color': '#d62728',
        'expected_tau_range': (0.25, 1.0),  # 250ms - 1000ms
    },
    'Respiration': {
        'columns': ['respiration'],
        'tau_method': 'autocorr',  # Autocorr decay timescale
        'color': '#2ca02c',
        'expected_tau_range': (0.5, 5.0),  # 500ms - 5s
    }
}


# =============================================================================
# SIGNAL PROCESSING
# =============================================================================

def preprocess_signal(signal: np.ndarray, fs: float, 
                      signal_type: str) -> np.ndarray:
    """
    Preprocess signal based on its type.
    
    - Remove DC offset
    - Apply appropriate bandpass filter
    - Handle NaNs
    """
    # Remove NaNs
    signal = np.asarray(signal).flatten()
    if np.any(~np.isfinite(signal)):
        # Interpolate NaNs
        mask = np.isfinite(signal)
        if np.sum(mask) < len(signal) * 0.5:
            return None  # Too many NaNs
        signal = np.interp(np.arange(len(signal)), 
                          np.arange(len(signal))[mask], 
                          signal[mask])
    
    # Remove DC
    signal = signal - np.mean(signal)
    
    # Type-specific filtering
    if signal_type == 'EEG':
        # Bandpass 1-45 Hz
        b, a = sig.butter(4, [1, 45], btype='band', fs=fs)
        signal = sig.filtfilt(b, a, signal)
    elif signal_type == 'ECG':
        # Bandpass 0.5-40 Hz
        b, a = sig.butter(4, [0.5, 40], btype='band', fs=fs)
        signal = sig.filtfilt(b, a, signal)
    elif signal_type == 'Respiration':
        # Lowpass 1 Hz (breathing is slow)
        b, a = sig.butter(4, 1, btype='low', fs=fs)
        signal = sig.filtfilt(b, a, signal)
    
    return signal


def discretize_signal(signal: np.ndarray, n_bins: int = 4) -> np.ndarray:
    """Discretize signal using quantile binning."""
    # Use quantile-based binning for even distribution
    percentiles = [100 * i / n_bins for i in range(1, n_bins)]  # e.g., [25, 50, 75] for 4 bins
    thresholds = np.percentile(signal, percentiles)
    return np.digitize(signal, thresholds)  # Returns 0, 1, 2, 3 for 4 bins


# =============================================================================
# PID FUNCTIONS
# =============================================================================

def build_embedding_distribution(x_discrete: np.ndarray, 
                                  lag1: int, lag2: int) -> Optional[Distribution]:
    """Build distribution for PID: I(X_{t-lag1}, X_{t-lag2} → X_t)."""
    max_lag = max(lag1, lag2)
    n = len(x_discrete)
    
    if n <= max_lag + 10:
        return None
    
    outcomes = []
    for t in range(max_lag, n):
        src1 = int(x_discrete[t - lag1])
        src2 = int(x_discrete[t - lag2])
        target = int(x_discrete[t])
        outcomes.append(f"{src1}{src2}{target}")
    
    counts = Counter(outcomes)
    total = sum(counts.values())
    
    outcomes_list = list(counts.keys())
    probs = [counts[o] / total for o in outcomes_list]
    
    return Distribution(outcomes_list, probs)


def compute_pid(dist: Distribution) -> Dict[str, float]:
    """Compute PID and return summary."""
    if dist is None:
        return {'redundancy': np.nan, 'unique_0': np.nan, 
                'unique_1': np.nan, 'synergy': np.nan, 'total_mi': np.nan}
    
    try:
        pid = PID_MMI(dist)
    except Exception as e:
        print(f"      PID error: {e}")
        return {'redundancy': np.nan, 'unique_0': np.nan, 
                'unique_1': np.nan, 'synergy': np.nan, 'total_mi': np.nan}
    
    summary = {'redundancy': 0.0, 'unique_0': 0.0, 'unique_1': 0.0, 'synergy': 0.0}
    
    # Use get_pi() method which works with both local and pip-installed dit
    # Iterate over lattice nodes to get all PID atoms
    for node in pid._lattice:
        try:
            val = float(pid.get_pi(node))
        except:
            val = 0.0
        
        # Identify which PID atom this is
        if len(node) == 2 and all(len(n) == 1 for n in node):
            # ((0,), (1,)) = redundancy
            summary['redundancy'] = val
        elif len(node) == 1 and len(node[0]) == 2:
            # ((0, 1),) = synergy
            summary['synergy'] = val
        elif node == ((0,),):
            summary['unique_0'] = val
        elif node == ((1,),):
            summary['unique_1'] = val
    
    # Compute total MI as sanity check
    summary['total_mi'] = summary['redundancy'] + summary['unique_0'] + summary['unique_1'] + summary['synergy']
    
    return summary


def compute_temporal_pid_profile(signal: np.ndarray, 
                                  tau: float, 
                                  fs: float,
                                  tau_multiples: List[float] = None,
                                  n_bins: int = 4) -> pd.DataFrame:
    """
    Compute PID at τ-normalized lags.
    
    For each pair of lags (τ_mult_1, τ_mult_2), compute PID.
    """
    if tau_multiples is None:
        tau_multiples = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    
    # Convert tau multiples to sample lags
    lags = {}
    for mult in tau_multiples:
        lag_samples = max(1, int(mult * tau * fs))
        lags[mult] = lag_samples
    
    # Discretize
    x_discrete = discretize_signal(signal, n_bins=n_bins)
    
    # Compute PID for all lag pairs
    results = []
    for i, mult1 in enumerate(tau_multiples):
        for mult2 in tau_multiples[i+1:]:
            lag1 = lags[mult1]
            lag2 = lags[mult2]
            
            dist = build_embedding_distribution(x_discrete, lag1, lag2)
            pid = compute_pid(dist)
            
            pid['tau_mult_1'] = mult1
            pid['tau_mult_2'] = mult2
            pid['lag1_samples'] = lag1
            pid['lag2_samples'] = lag2
            pid['lag1_ms'] = lag1 / fs * 1000
            pid['lag2_ms'] = lag2 / fs * 1000
            pid['lag_diff_tau'] = mult2 - mult1
            
            results.append(pid)
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS
# =============================================================================

@dataclass
class SignalAnalysis:
    """Container for signal analysis results."""
    signal_type: str
    channel: str
    tau: float
    tau_method: str
    tau_samples: int
    pid_profile: pd.DataFrame
    fs: float


def analyze_signal(signal: np.ndarray, 
                   signal_type: str, 
                   channel: str,
                   fs: float,
                   tau_method: str = 'auto',
                   tau_multiples: List[float] = None,
                   n_bins: int = 4) -> Optional[SignalAnalysis]:
    """
    Full analysis pipeline for a single signal.
    """
    # Preprocess
    processed = preprocess_signal(signal, fs, signal_type)
    if processed is None:
        return None
    
    # Check signal quality
    signal_std = np.std(processed)
    print(f"    Signal std: {signal_std:.4f}")
    
    if signal_std < 1e-10:
        print(f"    Signal has near-zero variance, skipping")
        return None
    
    # Normalize to unit variance for better discretization
    processed = processed / signal_std
    
    # Estimate tau
    estimator = TauEstimator(processed, fs)
    estimates = estimator.estimate_all(preferred_method=tau_method)
    tau = estimates.tau_best
    
    # Clamp tau to reasonable range for the signal type
    config = SIGNAL_CONFIGS.get(signal_type, {})
    tau_min, tau_max = config.get('expected_tau_range', (0.01, 10.0))
    tau_clamped = np.clip(tau, tau_min, tau_max)
    if tau != tau_clamped:
        print(f"    τ clamped: {tau*1000:.1f}ms → {tau_clamped*1000:.1f}ms")
        tau = tau_clamped
    
    print(f"    τ = {tau*1000:.1f} ms ({tau*fs:.1f} samples) via {estimates.method_used}")
    
    # Compute PID profile
    pid_profile = compute_temporal_pid_profile(
        processed, tau, fs, 
        tau_multiples=tau_multiples,
        n_bins=n_bins
    )
    
    # Check if we got any information
    mean_mi = pid_profile['total_mi'].mean()
    print(f"    Mean total MI: {mean_mi:.4f} bits")
    
    pid_profile['signal_type'] = signal_type
    pid_profile['channel'] = channel
    pid_profile['tau'] = tau
    pid_profile['tau_ms'] = tau * 1000
    
    return SignalAnalysis(
        signal_type=signal_type,
        channel=channel,
        tau=tau,
        tau_method=estimates.method_used,
        tau_samples=int(tau * fs),
        pid_profile=pid_profile,
        fs=fs
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_tau_comparison(analyses: List[SignalAnalysis], save_path: Path = None):
    """Bar chart comparing τ across signal types."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by signal type
    type_taus = {}
    for analysis in analyses:
        if analysis.signal_type not in type_taus:
            type_taus[analysis.signal_type] = []
        type_taus[analysis.signal_type].append(analysis.tau * 1000)  # Convert to ms
    
    types = list(type_taus.keys())
    means = [np.mean(type_taus[t]) for t in types]
    stds = [np.std(type_taus[t]) for t in types]
    colors = [SIGNAL_CONFIGS[t]['color'] for t in types]
    
    x = np.arange(len(types))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=12)
    ax.set_ylabel('Characteristic Timescale τ (ms)', fontsize=12)
    ax.set_title('Characteristic Timescales Across Biosignal Types', fontsize=14)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{mean:.0f}ms', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_pid_fingerprints(analyses: List[SignalAnalysis], save_path: Path = None):
    """
    Plot PID profiles vs normalized lag for each signal type.
    
    This is the key comparison: Synergy/Redundancy ratio vs τ-normalized lag.
    
    FIX: Use lag2 (longer lag) as x-axis with fixed lag1 at shortest timescale.
    This gives cleaner curves than grouping by lag difference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['redundancy', 'synergy', 'ratio']
    titles = ['Redundancy vs Longer Lag', 
              'Synergy vs Longer Lag',
              'Synergy/Redundancy Ratio (Fingerprint)']
    
    for ax_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[ax_idx]
        
        for signal_type in SIGNAL_CONFIGS.keys():
            type_analyses = [a for a in analyses if a.signal_type == signal_type]
            if not type_analyses:
                continue
            
            # Aggregate profiles
            all_profiles = pd.concat([a.pid_profile for a in type_analyses])
            
            # Filter to fixed shortest lag1 (0.1 tau) for cleaner curves
            min_lag1 = all_profiles['tau_mult_1'].min()
            fixed_lag1_data = all_profiles[all_profiles['tau_mult_1'] == min_lag1]
            
            # Group by lag2 (the longer lag)
            grouped = fixed_lag1_data.groupby('tau_mult_2').agg({
                'redundancy': ['mean', 'std'],
                'synergy': ['mean', 'std']
            }).reset_index()
            grouped.columns = ['tau_mult_2', 'red_mean', 'red_std', 'syn_mean', 'syn_std']
            
            # Compute ratio with protection against division by near-zero
            # Use minimum threshold to avoid instability
            grouped['ratio'] = grouped['syn_mean'] / np.maximum(grouped['red_mean'], 0.01)
            
            color = SIGNAL_CONFIGS[signal_type]['color']
            
            if metric == 'redundancy':
                ax.errorbar(grouped['tau_mult_2'], grouped['red_mean'],
                           yerr=grouped['red_std'], fmt='o-', color=color,
                           label=signal_type, capsize=3, linewidth=2, markersize=6)
            elif metric == 'synergy':
                ax.errorbar(grouped['tau_mult_2'], grouped['syn_mean'],
                           yerr=grouped['syn_std'], fmt='s-', color=color,
                           label=signal_type, capsize=3, linewidth=2, markersize=6)
            else:  # ratio
                ax.plot(grouped['tau_mult_2'], grouped['ratio'], 'o-',
                       color=color, label=signal_type, linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Longer Lag (tau units)', fontsize=11)
        ax.set_ylabel('Information (bits)' if metric != 'ratio' else 'Ratio', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
        
        if metric == 'ratio':
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim(0, 3)  # Clip for visibility
    
    plt.suptitle('Temporal Information Fingerprints (fixed short lag = 0.1 tau)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_pid_heatmaps_by_type(analyses: List[SignalAnalysis], 
                               metric: str = 'synergy',
                               save_path: Path = None):
    """Heatmaps of PID metric for each signal type."""
    signal_types = list(SIGNAL_CONFIGS.keys())
    n_types = len(signal_types)
    
    fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 4))
    if n_types == 1:
        axes = [axes]
    
    for idx, signal_type in enumerate(signal_types):
        ax = axes[idx]
        
        type_analyses = [a for a in analyses if a.signal_type == signal_type]
        if not type_analyses:
            ax.set_title(f'{signal_type}\nNo data')
            continue
        
        # Aggregate
        all_profiles = pd.concat([a.pid_profile for a in type_analyses])
        
        # Pivot for heatmap
        pivot = all_profiles.groupby(['tau_mult_1', 'tau_mult_2'])[metric].mean().unstack()
        
        cmap = 'Reds' if metric == 'synergy' else 'Greens'
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt='.3f',
                   cbar_kws={'label': 'bits'})
        ax.set_title(f'{signal_type}\nτ = {np.mean([a.tau for a in type_analyses])*1000:.0f}ms')
        ax.set_xlabel('Lag 2 (τ units)')
        ax.set_ylabel('Lag 1 (τ units)')
    
    plt.suptitle(f'{metric.title()} Across Lag Pairs (τ-normalized)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_grand_summary(analyses: List[SignalAnalysis], save_path: Path = None):
    """Comprehensive summary figure."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel A: τ comparison
    ax_tau = fig.add_subplot(gs[0, 0])
    type_taus = {}
    for analysis in analyses:
        if analysis.signal_type not in type_taus:
            type_taus[analysis.signal_type] = []
        type_taus[analysis.signal_type].append(analysis.tau * 1000)
    
    types = list(type_taus.keys())
    means = [np.mean(type_taus[t]) for t in types]
    colors = [SIGNAL_CONFIGS[t]['color'] for t in types]
    
    x = np.arange(len(types))
    ax_tau.bar(x, means, color=colors, alpha=0.8)
    ax_tau.set_xticks(x)
    ax_tau.set_xticklabels(types)
    ax_tau.set_ylabel('τ (ms)')
    ax_tau.set_title('A) Characteristic Timescales')
    ax_tau.set_yscale('log')
    for i, (xi, m) in enumerate(zip(x, means)):
        ax_tau.text(xi, m * 1.2, f'{m:.0f}ms', ha='center', fontsize=9)
    
    # Panel B: Redundancy profiles
    ax_red = fig.add_subplot(gs[0, 1])
    for signal_type in SIGNAL_CONFIGS.keys():
        type_analyses = [a for a in analyses if a.signal_type == signal_type]
        if not type_analyses:
            continue
        all_profiles = pd.concat([a.pid_profile for a in type_analyses])
        grouped = all_profiles.groupby('lag_diff_tau')['redundancy'].mean().reset_index()
        ax_red.plot(grouped['lag_diff_tau'], grouped['redundancy'], 'o-',
                   color=SIGNAL_CONFIGS[signal_type]['color'], 
                   label=signal_type, linewidth=2)
    ax_red.set_xlabel('Lag Difference (τ units)')
    ax_red.set_ylabel('Redundancy (bits)')
    ax_red.set_title('B) Redundancy vs Normalized Lag')
    ax_red.legend()
    ax_red.grid(alpha=0.3)
    ax_red.set_xscale('log')
    
    # Panel C: Synergy profiles
    ax_syn = fig.add_subplot(gs[0, 2])
    for signal_type in SIGNAL_CONFIGS.keys():
        type_analyses = [a for a in analyses if a.signal_type == signal_type]
        if not type_analyses:
            continue
        all_profiles = pd.concat([a.pid_profile for a in type_analyses])
        grouped = all_profiles.groupby('lag_diff_tau')['synergy'].mean().reset_index()
        ax_syn.plot(grouped['lag_diff_tau'], grouped['synergy'], 's-',
                   color=SIGNAL_CONFIGS[signal_type]['color'],
                   label=signal_type, linewidth=2)
    ax_syn.set_xlabel('Lag Difference (τ units)')
    ax_syn.set_ylabel('Synergy (bits)')
    ax_syn.set_title('C) Synergy vs Normalized Lag')
    ax_syn.legend()
    ax_syn.grid(alpha=0.3)
    ax_syn.set_xscale('log')
    
    # Panel D: Synergy/Redundancy Ratio (THE FINGERPRINT)
    ax_ratio = fig.add_subplot(gs[1, 0])
    for signal_type in SIGNAL_CONFIGS.keys():
        type_analyses = [a for a in analyses if a.signal_type == signal_type]
        if not type_analyses:
            continue
        all_profiles = pd.concat([a.pid_profile for a in type_analyses])
        grouped = all_profiles.groupby('lag_diff_tau').agg({
            'redundancy': 'mean', 'synergy': 'mean'
        }).reset_index()
        grouped['ratio'] = grouped['synergy'] / (grouped['redundancy'] + 1e-10)
        ax_ratio.plot(grouped['lag_diff_tau'], grouped['ratio'], 'o-',
                     color=SIGNAL_CONFIGS[signal_type]['color'],
                     label=signal_type, linewidth=2.5, markersize=8)
    ax_ratio.set_xlabel('Lag Difference (τ units)')
    ax_ratio.set_ylabel('Synergy / Redundancy')
    ax_ratio.set_title('D) Temporal Information Fingerprint')
    ax_ratio.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax_ratio.legend()
    ax_ratio.grid(alpha=0.3)
    ax_ratio.set_xscale('log')
    
    # Panel E: Summary statistics
    ax_summary = fig.add_subplot(gs[1, 1])
    
    summary_data = []
    for signal_type in SIGNAL_CONFIGS.keys():
        type_analyses = [a for a in analyses if a.signal_type == signal_type]
        if not type_analyses:
            continue
        all_profiles = pd.concat([a.pid_profile for a in type_analyses])
        summary_data.append({
            'Type': signal_type,
            'τ (ms)': np.mean([a.tau for a in type_analyses]) * 1000,
            'Mean Red': all_profiles['redundancy'].mean(),
            'Mean Syn': all_profiles['synergy'].mean(),
            'Syn/Red': all_profiles['synergy'].mean() / (all_profiles['redundancy'].mean() + 1e-10)
        })
    
    df_summary = pd.DataFrame(summary_data)
    ax_summary.axis('off')
    table = ax_summary.table(
        cellText=df_summary.round(3).values,
        colLabels=df_summary.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_summary.set_title('E) Summary Statistics')
    
    # Panel F: Interpretation
    ax_interp = fig.add_subplot(gs[1, 2])
    ax_interp.axis('off')
    
    interpretation = """
    INTERPRETATION
    ==============
    
    τ (Characteristic Timescale):
    • EEG: ~10-100ms (neural oscillations)
    • ECG: ~500-1000ms (heart rate)
    • Respiration: ~2000-5000ms (breathing)
    
    Synergy/Redundancy Ratio:
    • > 1: Nonlinear temporal integration
    • ≈ 1: Balanced structure
    • < 1: Smooth, predictable dynamics
    
    KEY QUESTIONS:
    1. Does ratio increase at larger τ?
       → Hierarchical temporal processing
    2. Which signal has highest ratio?
       → Most complex temporal dynamics
    3. Do shapes differ across types?
       → Different integration architectures
    """
    
    ax_interp.text(0.05, 0.95, interpretation, transform=ax_interp.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Cross-Biosignal Temporal PID Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CROSS-BIOSIGNAL TEMPORAL PID COMPARISON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load data
    data_file = DATA_DIR / "test_biosignals.csv"
    print(f"\nLoading: {data_file}")
    
    df = pd.read_csv(data_file, low_memory=False)
    print(f"Shape: {df.shape}")
    
    # Estimate sampling rate (assume 256 Hz if not specified)
    FS = 256  # Hz - adjust if different
    print(f"Assumed sampling rate: {FS} Hz")
    
    # Analysis parameters
    SUBSAMPLE = 100000  # Use subset for speed
    N_BINS = 4
    # Use smaller tau multiples to capture dynamics at/near characteristic timescale
    TAU_MULTIPLES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    
    print(f"\nParameters:")
    print(f"  Samples: {SUBSAMPLE}")
    print(f"  Bins: {N_BINS}")
    print(f"  Tau multiples: {TAU_MULTIPLES}")
    
    # Analyze each signal type
    all_analyses = []
    
    for signal_type, config in SIGNAL_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {signal_type}")
        print("="*60)
        
        for col in config['columns']:
            if col not in df.columns:
                print(f"  Column {col} not found, skipping")
                continue
            
            print(f"\n  Channel: {col}")
            
            # Get signal
            signal = df[col].values[:SUBSAMPLE]
            signal = signal.astype(float)
            
            # Skip if too many NaNs
            if np.sum(np.isfinite(signal)) < SUBSAMPLE * 0.5:
                print(f"    Too many NaNs, skipping")
                continue
            
            # Analyze
            analysis = analyze_signal(
                signal, signal_type, col, FS,
                tau_method=config['tau_method'],
                tau_multiples=TAU_MULTIPLES,
                n_bins=N_BINS
            )
            
            if analysis:
                all_analyses.append(analysis)
                print(f"    ✓ Computed {len(analysis.pid_profile)} lag pairs")
    
    print(f"\n{'='*70}")
    print(f"Completed {len(all_analyses)} signal analyses")
    print("="*70)
    
    # Combine results
    all_profiles = pd.concat([a.pid_profile for a in all_analyses], ignore_index=True)
    all_profiles.to_csv(RESULTS_DIR / "all_pid_profiles.csv", index=False)
    
    # Save tau summary
    tau_summary = pd.DataFrame([{
        'signal_type': a.signal_type,
        'channel': a.channel,
        'tau_seconds': a.tau,
        'tau_ms': a.tau * 1000,
        'tau_samples': a.tau_samples,
        'tau_method': a.tau_method
    } for a in all_analyses])
    tau_summary.to_csv(RESULTS_DIR / "tau_summary.csv", index=False)
    
    print("\n" + "="*70)
    print("GENERATING FIGURES")
    print("="*70)
    
    # Generate figures
    print("\n1. τ comparison...")
    plot_tau_comparison(all_analyses, RESULTS_DIR / "tau_comparison.png")
    
    print("2. PID fingerprints...")
    plot_pid_fingerprints(all_analyses, RESULTS_DIR / "pid_fingerprints.png")
    
    print("3. PID heatmaps...")
    plot_pid_heatmaps_by_type(all_analyses, 'synergy', 
                              RESULTS_DIR / "heatmaps_synergy.png")
    plot_pid_heatmaps_by_type(all_analyses, 'redundancy',
                              RESULTS_DIR / "heatmaps_redundancy.png")
    
    print("4. Grand summary...")
    plot_grand_summary(all_analyses, RESULTS_DIR / "grand_summary.png")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(tau_summary.to_string(index=False))
    
    print(f"\n\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
