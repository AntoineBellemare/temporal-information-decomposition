"""
EEG Temporal PhiID Analysis
===========================

Use PhiID (Integrated Information Decomposition) for temporal structure analysis.

PhiID differs from PID:
- PID: Decomposes I(X1, X2 → Y) for 3 variables
- PhiID: Decomposes MI between two time series (src, tgt), analyzing how their 
  pasts (at t-tau) relate to their futures (at t)

Understanding calc_PhiID(src, tgt, tau):
  - Creates 4 vectors: src_past, tgt_past, src_future, tgt_future
  - src_past = src[:-tau], src_future = src[tau:]  (src at t-tau and t)
  - tgt_past = tgt[:-tau], tgt_future = tgt[tau:]  (tgt at t-tau and t)
  - tau is THE temporal delay used for the analysis
  
For TEMPORAL analysis of a SINGLE signal:
  - src = signal (the time series)
  - tgt = same signal shifted by an additional offset (creates spatial separation)
  - OR: tgt = signal itself (degenerate case, but valid - self-PhiID)
  
APPROACH 1: Self-PhiID (src = tgt = signal)
  - Analyzes how signal's own past relates to its future
  - X_past, Y_past are identical; X_future, Y_future are identical
  - Still gives 16 atoms due to the formalism, but many will be redundant/zero
  - tau directly controls the timescale

APPROACH 2: Bivariate embedding (recommended for richer decomposition)
  - Create two delayed versions as "pseudo-bivariate" system
  - src = signal, tgt = signal shifted by extra_lag
  - This creates distinct X and Y processes
  - tau = the PhiID embedding delay (typically small, e.g., 1)
  - extra_lag = the timescale separation we want to probe

We implement APPROACH 2: treating x[t] and x[t-extra_lag] as two coupled processes.

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

# Import PhiID
from phyid.calculate import calc_PhiID

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
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

BAND_COLORS = {
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


def create_lag_schedule(fs, timescales_ms):
    """Create lag values from timescales in ms."""
    lags = [max(1, int(t * fs / 1000)) for t in timescales_ms]
    return sorted(list(set(lags)))


# =============================================================================
# PhiID TEMPORAL ANALYSIS
# =============================================================================

def compute_temporal_phiid(signal, extra_lag, tau=1, kind='gaussian', redundancy='MMI'):
    """
    Compute PhiID for a signal treated as a bivariate process.
    
    We create a pseudo-bivariate system:
      - src = signal (X process)
      - tgt = signal shifted by extra_lag (Y process = X delayed)
    
    Then PhiID internally uses tau to create:
      - X_past = src[:-tau] = signal[:-tau]
      - X_future = src[tau:] = signal[tau:]
      - Y_past = tgt[:-tau] = signal[extra_lag:-tau] 
      - Y_future = tgt[tau:] = signal[extra_lag+tau:]
    
    This gives us the temporal structure at timescale 'extra_lag', with PhiID's
    internal tau controlling the embedding window.
    
    Parameters
    ----------
    signal : array
        1D time series (continuous values work best with kind='gaussian')
    extra_lag : int
        The timescale separation between X and Y processes (in samples)
        This is the "temporal offset" we want to probe
    tau : int
        PhiID's embedding delay (typically 1 or small)
        This creates the past/future distinction within each process
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
    n = len(signal)
    
    # Need enough samples: extra_lag + tau + some buffer
    if n <= extra_lag + tau + 10:
        return None, None, None
    
    # Create pseudo-bivariate system
    # src = X process (original signal, truncated to align)
    # tgt = Y process (signal shifted by extra_lag)
    # They must have the same length for PhiID
    
    # After the shift, we have:
    # src[t] corresponds to signal[t]
    # tgt[t] corresponds to signal[t + extra_lag]
    # So tgt is the "future" of src
    
    # Align: 
    # tgt = signal[extra_lag:] (the later part)
    # src = signal[:-extra_lag] (the earlier part)
    # Now src[t] and tgt[t] are separated by extra_lag samples
    
    src = signal[:-extra_lag].copy()  # Earlier (past-ish)
    tgt = signal[extra_lag:].copy()   # Later (future-ish)
    
    # Now PhiID will internally do:
    # src_past = src[:-tau], src_future = src[tau:]
    # tgt_past = tgt[:-tau], tgt_future = tgt[tau:]
    # 
    # Which corresponds to:
    # src_past = signal[:-extra_lag][:-tau] = signal[:-extra_lag-tau+?] 
    # tgt_future = signal[extra_lag:][tau:] = signal[extra_lag+tau:]
    #
    # The 4-way relationship captures:
    # - How does signal at (t-tau) relate to signal at (t)?
    # - How does signal at (t+extra_lag-tau) relate to signal at (t+extra_lag)?
    # - How do these two "processes" share/transfer information?
    
    try:
        atoms_res, _ = calc_PhiID(src, tgt, tau, kind=kind, redundancy=redundancy)
    except Exception as e:
        print(f"    PhiID error: {e}")
        return None, None, None
    
    # Average over time to get scalars
    atoms = {}
    for name in ATOM_NAMES:
        if name in atoms_res:
            atoms[name] = float(np.nanmean(atoms_res[name]))
        else:
            atoms[name] = np.nan
    
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


def analyze_channel_temporal_phiid(signal, extra_lags, tau=1, kind='gaussian', redundancy='MMI'):
    """
    Analyze a channel across multiple extra_lag values.
    
    Parameters
    ----------
    signal : array
        1D time series
    extra_lags : list of int
        List of timescale separations to probe (in samples)
    tau : int
        PhiID embedding delay (typically 1)
    kind : str
        'gaussian' or 'discrete'
    redundancy : str
        'MMI' or 'CCS'
    
    Returns DataFrames for atoms, dynamics, and IIT metrics.
    """
    all_atoms = []
    all_dynamics = []
    all_iit = []
    
    for extra_lag in extra_lags:
        atoms, dynamics, iit = compute_temporal_phiid(signal, extra_lag, tau, kind, redundancy)
        
        if atoms is None:
            continue
        
        atoms['extra_lag'] = extra_lag
        dynamics['extra_lag'] = extra_lag
        iit['extra_lag'] = extra_lag
        
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
    """Heatmap of all 16 atoms vs lag offset."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Pivot to matrix
    atom_cols = [c for c in df_atoms.columns if c != 'extra_lag']
    matrix = df_atoms[atom_cols].values.T
    
    time_labels = [f'{l/fs*1000:.0f}' for l in df_atoms['extra_lag']]
    
    sns.heatmap(matrix, ax=ax, cmap='RdBu_r', center=0,
                xticklabels=time_labels, yticklabels=atom_cols,
                cbar_kws={'label': 'bits'})
    
    ax.set_xlabel('Lag Offset (ms)')
    ax.set_ylabel('PhiID Atom')
    ax.set_title(f'PhiID Atoms vs Temporal Lag: {channel}')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_dynamics_summary(df_dynamics, channel, fs, save_path=None):
    """Plot information dynamics metrics vs lag."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    time_ms = df_dynamics['extra_lag'] / fs * 1000
    metrics = [c for c in df_dynamics.columns if c != 'extra_lag']
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(time_ms, df_dynamics[metric], 'o-', color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel('Lag Offset (ms)')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' '))
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    plt.suptitle(f'Information Dynamics: {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_iit_summary(df_iit, channel, fs, save_path=None):
    """Plot IIT metrics vs lag."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    time_ms = df_iit['extra_lag'] / fs * 1000
    metrics = [c for c in df_iit.columns if c != 'extra_lag']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(time_ms, df_iit[metric], 'o-', color=colors[i], linewidth=2, markersize=6)
        ax.set_xlabel('Lag Offset (ms)')
        ax.set_ylabel('bits')
        ax.set_title(metric.replace('_', ' '))
        ax.grid(alpha=0.3)
        ax.set_xscale('log')
    
    plt.suptitle(f'IIT Metrics: {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_band_comparison_phiid(all_results, metric, fs, save_path=None):
    """Compare a specific metric across bands."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for band_name, results in all_results.items():
        if results is None:
            continue
        
        # Average across channels
        dfs = [r['dynamics'] for r in results.values() if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        
        combined = pd.concat(dfs)
        grouped = combined.groupby('extra_lag')[metric].agg(['mean', 'std']).reset_index()
        
        time_ms = grouped['extra_lag'] / fs * 1000
        
        ax.errorbar(time_ms, grouped['mean'], yerr=grouped['std'],
                   fmt='o-', color=BAND_COLORS[band_name], label=band_name.upper(),
                   linewidth=2, capsize=3)
    
    ax.set_xlabel('Lag Offset (ms)')
    ax.set_ylabel(f'{metric} (bits)')
    ax.set_title(f'{metric.replace("_", " ")} Across Frequency Bands')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def plot_grand_summary_phiid(all_results, fs, save_path=None):
    """Grand summary of PhiID analysis."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    band_order = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Panel A: Storage across bands
    ax_storage = fig.add_subplot(gs[0, 0])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['dynamics'] for r in all_results[band_name].values() 
               if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('extra_lag')['Storage'].mean().reset_index()
        time_ms = grouped['extra_lag'] / fs * 1000
        ax_storage.plot(time_ms, grouped['Storage'], 'o-', color=BAND_COLORS[band_name],
                       label=band_name.upper(), linewidth=2)
    ax_storage.set_xlabel('Lag Offset (ms)')
    ax_storage.set_ylabel('Storage (bits)')
    ax_storage.set_title('A) Information Storage')
    ax_storage.legend(fontsize=8)
    ax_storage.grid(alpha=0.3)
    ax_storage.set_xscale('log')
    
    # Panel B: Transfer across bands
    ax_transfer = fig.add_subplot(gs[0, 1])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['dynamics'] for r in all_results[band_name].values() 
               if r is not None and r['dynamics'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('extra_lag')['Transfer'].mean().reset_index()
        time_ms = grouped['extra_lag'] / fs * 1000
        ax_transfer.plot(time_ms, grouped['Transfer'], 'o-', color=BAND_COLORS[band_name],
                        label=band_name.upper(), linewidth=2)
    ax_transfer.set_xlabel('Lag Offset (ms)')
    ax_transfer.set_ylabel('Transfer (bits)')
    ax_transfer.set_title('B) Information Transfer')
    ax_transfer.legend(fontsize=8)
    ax_transfer.grid(alpha=0.3)
    ax_transfer.set_xscale('log')
    
    # Panel C: Integrated Info
    ax_phi = fig.add_subplot(gs[0, 2])
    for band_name in band_order:
        if band_name not in all_results or all_results[band_name] is None:
            continue
        dfs = [r['iit'] for r in all_results[band_name].values() 
               if r is not None and r['iit'] is not None]
        if not dfs:
            continue
        combined = pd.concat(dfs)
        grouped = combined.groupby('extra_lag')['Integrated_info'].mean().reset_index()
        time_ms = grouped['extra_lag'] / fs * 1000
        ax_phi.plot(time_ms, grouped['Integrated_info'], 'o-', color=BAND_COLORS[band_name],
                   label=band_name.upper(), linewidth=2)
    ax_phi.set_xlabel('Lag Offset (ms)')
    ax_phi.set_ylabel('Φ (bits)')
    ax_phi.set_title('C) Integrated Information')
    ax_phi.legend(fontsize=8)
    ax_phi.grid(alpha=0.3)
    ax_phi.set_xscale('log')
    
    # Panel D: Downward vs Upward causation for alpha
    ax_causation = fig.add_subplot(gs[1, 0])
    if 'alpha' in all_results and all_results['alpha'] is not None:
        dfs = [r['dynamics'] for r in all_results['alpha'].values() 
               if r is not None and r['dynamics'] is not None]
        if dfs:
            combined = pd.concat(dfs)
            grouped = combined.groupby('extra_lag').agg({
                'Downward_causation': 'mean',
                'Upward_causation': 'mean'
            }).reset_index()
            time_ms = grouped['extra_lag'] / fs * 1000
            ax_causation.plot(time_ms, grouped['Downward_causation'], 'o-', 
                             color='purple', label='Downward', linewidth=2)
            ax_causation.plot(time_ms, grouped['Upward_causation'], 's-', 
                             color='orange', label='Upward', linewidth=2)
    ax_causation.set_xlabel('Lag Offset (ms)')
    ax_causation.set_ylabel('Causation (bits)')
    ax_causation.set_title('D) Causation Direction (Alpha)')
    ax_causation.legend()
    ax_causation.grid(alpha=0.3)
    ax_causation.set_xscale('log')
    
    # Panel E: Bars by band
    ax_bars = fig.add_subplot(gs[1, 1])
    
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
    
    # Panel F: Interpretation
    ax_interp = fig.add_subplot(gs[1, 2])
    ax_interp.axis('off')
    
    interpretation = """
    PhiID TEMPORAL ANALYSIS
    =======================
    
    WHAT WE'RE MEASURING:
    • Temporal PhiID: How info flows between
      a signal and its time-shifted self
    
    KEY METRICS:
    • Storage: Info preserved over time
    • Transfer: Info flowing forward in time  
    • Copy: Redundant temporal structure
    • Causation: Direction of info flow
    
    INTERPRETATION:
    • High Storage: Persistent dynamics
    • High Transfer: Active processing
    • High Φ: Integrated temporal structure
    
    BAND SIGNATURES:
    • Delta: High storage (slow persistence)
    • Alpha: Periodic transfer patterns
    • Gamma: Rapid transfer, low storage
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
    TAU = 1  # PhiID embedding delay (small - the extra_lag is our main timescale)
    KIND = 'gaussian'  # Use Gaussian for continuous EEG
    REDUNDANCY = 'MMI'
    
    # Extra lags (timescales to analyze) - this is the separation between X and Y processes
    TIMESCALES_MS = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
    EXTRA_LAGS = create_lag_schedule(fs, TIMESCALES_MS)
    
    # Select key channels for analysis
    KEY_CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    channels_to_analyze = [c for c in KEY_CHANNELS if c in channels]
    
    print(f"\nParameters:")
    print(f"  Samples: {SUBSAMPLE}")
    print(f"  Tau: {TAU}")
    print(f"  Kind: {KIND}")
    print(f"  Redundancy: {REDUNDANCY}")
    print(f"  Extra lags: {EXTRA_LAGS}")
    print(f"  Timescales: {TIMESCALES_MS} ms")
    print(f"  Channels: {channels_to_analyze}")
    
    # Storage
    all_results = {}  # {band: {channel: {atoms, dynamics, iit}}}
    
    # Process each band
    for band_name, (fmin, fmax) in BANDS.items():
        print(f"\n{'='*60}")
        print(f"Processing: {band_name.upper()} ({fmin}-{fmax} Hz)")
        print("="*60)
        
        all_results[band_name] = {}
        
        for ch in channels_to_analyze:
            print(f"  Channel: {ch}", end=" ")
            
            # Get signal
            signal = df[ch].values[:SUBSAMPLE]
            signal = signal[~np.isnan(signal)]
            
            try:
                # Bandpass filter
                filtered = bandpass_filter(signal, fmin, fmax, fs)
                
                # Analyze PhiID
                df_atoms, df_dynamics, df_iit = analyze_channel_temporal_phiid(
                    filtered, EXTRA_LAGS, TAU, kind=KIND, redundancy=REDUNDANCY
                )
                
                if df_atoms is None:
                    print("✗ failed")
                    all_results[band_name][ch] = None
                    continue
                
                all_results[band_name][ch] = {
                    'atoms': df_atoms,
                    'dynamics': df_dynamics,
                    'iit': df_iit
                }
                
                print(f"✓ ({len(df_atoms)} lags)")
                
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
    
    # 3. Individual channel plots for alpha
    print("3. Alpha band channel details...")
    if 'alpha' in all_results:
        for ch, results in all_results['alpha'].items():
            if results is None:
                continue
            ch_name = ch.replace('eeg-', '')
            
            plot_phiid_atoms_heatmap(results['atoms'], ch, fs,
                                     save_path=RESULTS_DIR / f"atoms_{ch_name}_alpha.png")
            plot_dynamics_summary(results['dynamics'], ch, fs,
                                  save_path=RESULTS_DIR / f"dynamics_{ch_name}_alpha.png")
            plot_iit_summary(results['iit'], ch, fs,
                             save_path=RESULTS_DIR / f"iit_{ch_name}_alpha.png")
    
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
