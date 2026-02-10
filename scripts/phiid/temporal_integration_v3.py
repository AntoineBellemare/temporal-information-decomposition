#!/usr/bin/env python3
"""
Temporal Integration Analysis v3 - Takens Embedding Approach

This version uses proper Takens delay embedding with direct 4-vector construction,
solving the two-lag problem present in v2.

Key improvement:
- v2: Irregular sampling (tau, extra_lag-tau, tau) with TWO lag parameters
- v3: Regular sampling (tau, tau, tau) with ONE lag parameter (tau_embed)

Timeline comparison:
- v2: t, t+τ, t+lag, t+lag+τ  (irregular)
- v3: t, t+τ, t+2τ, t+3τ     (perfectly regular)

Author: Temporal PhiID Project
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy.stats import zscore, entropy
from scipy.signal import hilbert
from scipy.linalg import lstsq
from sklearn.cluster import KMeans
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "integrated-info-decomp"))

# Import PhiID internal functions (bypassing calc_PhiID)
from phyid.calculate import (
    _get_entropy_four_vec,
    _get_coinfo_four_vec,
    _get_redundancy_four_vec,
    _get_double_redundancy_four_vec,
    _get_atoms_four_vec
)
from phyid.utils import PhiID_atoms_abbr

# Directories
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "phiid" / "temporal_integration_v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# PhiID atom names
ATOM_NAMES = ['rtr', 'rtx', 'rty', 'rts', 'xtr', 'xtx', 'xty', 'xts',
              'ytr', 'ytx', 'yty', 'yts', 'str', 'stx', 'sty', 'sts']

# Information Dynamics metric definitions
DYNAMICS_GROUPS = {
    'Storage': ['rtr', 'xtx', 'yty', 'sts'],
    'Copy': ['xtx', 'yty'],
    'Transfer': ['xty', 'ytx'],
    'Erasure': ['rtx', 'rty'],
    'Upward_causation': ['xts', 'yts', 'rts'],
    'Downward_causation': ['stx', 'sty', 'str'],
}


def load_eeg_data(file_path, target_fs, max_duration):
    """Load and preprocess EEG data."""
    df = pd.read_csv(file_path)
    
    eeg_cols = [c for c in df.columns if 'eeg' in c.lower()]
    if not eeg_cols:
        eeg_cols = [c for c in df.columns if c not in ['timestamp', 'time', 'sample']]
    
    eeg_data = df[eeg_cols].copy()
    
    # Use target_fs directly (skip timestamp parsing which can be unreliable)
    fs = target_fs
    
    # Limit duration
    max_samples = int(max_duration * fs)
    if len(eeg_data) > max_samples:
        eeg_data = eeg_data.iloc[:max_samples]
    
    print(f"Loaded {len(eeg_data)} samples at {fs} Hz ({len(eeg_data)/fs:.1f}s)")
    print(f"Channels: {eeg_cols}")
    
    return eeg_data, fs, eeg_cols


def takens_phiid_direct(signal, tau_embed, kind="gaussian", redundancy="MMI"):
    """
    Compute PhiID on Takens embedding with TRUE SINGLE lag parameter.
    
    Bypasses calc_PhiID to avoid the two-lag problem.
    Creates perfectly regular temporal sampling: t, t+τ, t+2τ, t+3τ
    
    Parameters
    ----------
    signal : array
        1D time series segment
    tau_embed : int
        Embedding delay in samples (the ONLY temporal parameter)
    kind : str
        'gaussian' or 'discrete'
    redundancy : str
        'MMI' or 'CCS'
    
    Returns
    -------
    atoms_res : dict
        PhiID atoms (16 atoms, each is a scalar mean value)
    """
    N = len(signal) - 3 * tau_embed
    
    if N < 10:  # Need minimum samples
        return None
    
    # Create 4D Takens embedding with PERFECTLY REGULAR spacing
    X = np.zeros((4, N))
    X[0] = signal[0:N]                          # p1 = x(t)       - "src_past"
    X[1] = signal[tau_embed:N+tau_embed]        # p2 = x(t+τ)     - "tgt_past"
    X[2] = signal[2*tau_embed:N+2*tau_embed]    # t1 = x(t+2τ)    - "src_future"
    X[3] = signal[3*tau_embed:N+3*tau_embed]    # t2 = x(t+3τ)    - "tgt_future"
    
    # Timeline: t, t+τ, t+2τ, t+3τ
    # Spacing:  [τ], [τ], [τ]  ← PERFECTLY REGULAR!
    
    # Normalize (same as calc_PhiID does)
    if kind == "gaussian":
        stds = np.std(X, axis=1, ddof=1, keepdims=True)
        stds = np.maximum(stds, 1e-10)  # Avoid division by zero
        X_norm = X / stds
        X_input = X_norm
    else:
        raise ValueError("Only 'gaussian' kind supported in this implementation")
    
    try:
        # Run PhiID computation pipeline directly
        h_res = _get_entropy_four_vec(X_input, kind=kind)
        I_res = _get_coinfo_four_vec(h_res)
        R_res = _get_redundancy_four_vec(redundancy, I_res)
        
        calc_res = {
            "h_res": h_res,
            "I_res": I_res,
            "R_res": R_res
        }
        
        rtr = _get_double_redundancy_four_vec(redundancy, calc_res)
        calc_res["rtr"] = rtr
        
        atoms_res = _get_atoms_four_vec(calc_res)
        
        # Convert to scalar means
        atoms_scalar = {}
        for name in ATOM_NAMES:
            if name in atoms_res:
                val = np.nanmean(atoms_res[name])
                atoms_scalar[name] = val if np.isfinite(val) else 0.0
            else:
                atoms_scalar[name] = 0.0
        
        return atoms_scalar
        
    except Exception as e:
        return None


def estimate_optimal_tau(signal, fs, max_tau_ms=100):
    """
    Estimate optimal embedding tau using first minimum of mutual information.
    
    Parameters
    ----------
    signal : array
        1D time series
    fs : int
        Sampling frequency
    max_tau_ms : float
        Maximum tau to consider in milliseconds
    
    Returns
    -------
    tau_opt : int
        Optimal tau in samples
    """
    max_tau_samples = int(max_tau_ms * fs / 1000)
    max_tau_samples = min(max_tau_samples, len(signal) // 4)
    
    # Use autocorrelation-based method (faster than MI)
    signal_centered = signal - np.mean(signal)
    acf = np.correlate(signal_centered, signal_centered, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]  # Normalize
    
    # Find first local minimum or zero crossing
    for i in range(1, min(max_tau_samples, len(acf) - 1)):
        if acf[i] < acf[i-1] and acf[i] < acf[i+1]:
            return i
        if acf[i] <= 0:
            return i
    
    # Fallback: use 1/4 of dominant period
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    
    # Ignore DC and very low frequencies
    fft[0] = 0
    if len(fft) > 5:
        fft[:3] = 0
    
    peak_idx = np.argmax(fft)
    if peak_idx > 0 and freqs[peak_idx] > 0:
        period_samples = int(fs / freqs[peak_idx])
        return max(1, period_samples // 4)
    
    return max(1, max_tau_samples // 4)


def compute_phiid_for_tau(signal, tau_embed, window_samples, step_samples, n_windows):
    """
    Compute PhiID atoms using Takens embedding for one tau value.
    
    Parameters
    ----------
    signal : array
        Full signal
    tau_embed : int
        Embedding delay (the ONLY temporal parameter)
    window_samples : int
        Analysis window size
    step_samples : int
        Step between windows
    n_windows : int
        Number of windows
    
    Returns
    -------
    atom_series : dict
        Time series of each atom
    """
    atom_series = {name: np.zeros(n_windows) for name in ATOM_NAMES}
    
    # Need extra samples for Takens embedding
    min_window = window_samples + 3 * tau_embed
    
    for i in range(n_windows):
        start = i * step_samples
        end = start + min_window
        
        if end > len(signal):
            break
        
        segment = signal[start:end]
        
        atoms = takens_phiid_direct(segment, tau_embed)
        
        if atoms is not None:
            for name in ATOM_NAMES:
                atom_series[name][i] = atoms.get(name, 0.0)
    
    return atom_series


def compute_dynamics_metrics(atom_series):
    """Compute the 6 information dynamics metrics from atoms."""
    n_windows = len(atom_series['rtr'])
    metrics = {}
    
    for metric_name, atom_list in DYNAMICS_GROUPS.items():
        metric_values = np.zeros(n_windows)
        for atom_name in atom_list:
            metric_values += atom_series[atom_name]
        metrics[metric_name] = metric_values
    
    return metrics


def compute_normalized_divergence(metrics_by_tau, tau_values):
    """
    Compute divergence normalized by each tau's typical values.
    
    Parameters
    ----------
    metrics_by_tau : dict
        {tau: {metric_name: timeseries}}
    tau_values : list
        List of tau values used
    
    Returns
    -------
    results : dict
        For each metric: divergence, TII, raw values, correlation matrix
    """
    results = {}
    
    for metric_name in DYNAMICS_GROUPS.keys():
        # Stack all tau timeseries
        raw_matrix = np.array([metrics_by_tau[tau][metric_name] for tau in tau_values])
        
        # Z-normalize within each tau (removes amplitude differences)
        z_matrix = np.zeros_like(raw_matrix)
        for i in range(len(tau_values)):
            row = raw_matrix[i]
            if np.std(row) > 0:
                z_matrix[i] = (row - np.mean(row)) / np.std(row)
            else:
                z_matrix[i] = 0
        
        # Divergence = std across taus at each timepoint
        divergence = np.std(z_matrix, axis=0)
        
        # TII = 1 / (1 + divergence)
        tii = 1 / (1 + divergence)
        
        # Cross-lag correlation matrix
        corr_matrix = np.corrcoef(raw_matrix)
        
        # Mean cross-lag correlation (off-diagonal)
        n_taus = len(tau_values)
        off_diag = corr_matrix[np.triu_indices(n_taus, k=1)]
        mean_cross_corr = np.nanmean(off_diag)
        
        results[metric_name] = {
            'raw_by_tau': {tau: raw_matrix[i] for i, tau in enumerate(tau_values)},
            'z_matrix': z_matrix,
            'normalized_divergence': divergence,
            'tii': tii,
            'corr_matrix': corr_matrix,
            'mean_cross_tau_corr': mean_cross_corr,
        }
    
    return results


def compute_cross_scale_prediction(results, tau_values, max_offset=10):
    """Compute lagged correlation between shortest and longest tau timeseries."""
    cross_scale = {}
    
    short_tau = min(tau_values)
    long_tau = max(tau_values)
    
    for metric_name in DYNAMICS_GROUPS.keys():
        short_series = results[metric_name]['raw_by_tau'][short_tau]
        long_series = results[metric_name]['raw_by_tau'][long_tau]
        
        # Normalize
        short_norm = (short_series - np.mean(short_series)) / (np.std(short_series) + 1e-10)
        long_norm = (long_series - np.mean(long_series)) / (np.std(long_series) + 1e-10)
        
        correlations = []
        offsets = range(-max_offset, max_offset + 1)
        
        for offset in offsets:
            if offset < 0:
                s = short_norm[-offset:]
                l = long_norm[:offset]
            elif offset > 0:
                s = short_norm[:-offset]
                l = long_norm[offset:]
            else:
                s = short_norm
                l = long_norm
            
            if len(s) > 10:
                corr = np.corrcoef(s, l)[0, 1]
                correlations.append(corr if np.isfinite(corr) else 0)
            else:
                correlations.append(0)
        
        correlations = np.array(correlations)
        best_idx = np.argmax(np.abs(correlations))
        
        cross_scale[metric_name] = {
            'offsets': list(offsets),
            'correlations': correlations,
            'best_offset': list(offsets)[best_idx],
            'best_correlation': correlations[best_idx],
        }
    
    return cross_scale


def compute_granger_causality(results, tau_values, max_order=5):
    """Compute Granger causality between short and long tau dynamics."""
    granger_results = {}
    
    short_tau = min(tau_values)
    long_tau = max(tau_values)
    
    for metric_name in DYNAMICS_GROUPS.keys():
        short_series = results[metric_name]['raw_by_tau'][short_tau]
        long_series = results[metric_name]['raw_by_tau'][long_tau]
        
        gc_short_to_long = []
        gc_long_to_short = []
        
        for order in range(1, max_order + 1):
            n = len(short_series) - order
            if n < 20:
                continue
            
            Y_long = long_series[order:]
            Y_short = short_series[order:]
            
            # Test short -> long
            try:
                X_long_only = np.column_stack([long_series[order-i-1:n+order-i-1] 
                                               for i in range(order)])
                X_short_only = np.column_stack([short_series[order-i-1:n+order-i-1] 
                                                for i in range(order)])
                X_both = np.column_stack([X_long_only, X_short_only])
                
                beta_r, _, _, _ = lstsq(X_long_only, Y_long)
                rss_restricted = np.sum((Y_long - X_long_only @ beta_r)**2)
                
                beta_u, _, _, _ = lstsq(X_both, Y_long)
                rss_unrestricted = np.sum((Y_long - X_both @ beta_u)**2)
                
                df1 = order
                df2 = len(Y_long) - 2*order
                if rss_unrestricted > 0 and df2 > 0:
                    F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                    gc_short_to_long.append(F)
                else:
                    gc_short_to_long.append(0)
            except:
                gc_short_to_long.append(0)
            
            # Test long -> short
            try:
                X_short_only_s = np.column_stack([short_series[order-i-1:n+order-i-1] 
                                                  for i in range(order)])
                X_long_only_s = np.column_stack([long_series[order-i-1:n+order-i-1] 
                                                 for i in range(order)])
                X_both_s = np.column_stack([X_short_only_s, X_long_only_s])
                
                beta_r, _, _, _ = lstsq(X_short_only_s, Y_short)
                rss_restricted = np.sum((Y_short - X_short_only_s @ beta_r)**2)
                
                beta_u, _, _, _ = lstsq(X_both_s, Y_short)
                rss_unrestricted = np.sum((Y_short - X_both_s @ beta_u)**2)
                
                df2 = len(Y_short) - 2*order
                if rss_unrestricted > 0 and df2 > 0:
                    F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                    gc_long_to_short.append(F)
                else:
                    gc_long_to_short.append(0)
            except:
                gc_long_to_short.append(0)
        
        max_gc_s2l = max(gc_short_to_long) if gc_short_to_long else 0
        max_gc_l2s = max(gc_long_to_short) if gc_long_to_short else 0
        
        if max_gc_s2l > max_gc_l2s and max_gc_s2l > 2:
            direction = 'short_leads'
        elif max_gc_l2s > max_gc_s2l and max_gc_l2s > 2:
            direction = 'long_leads'
        else:
            direction = 'bidirectional'
        
        granger_results[metric_name] = {
            'short_to_long': max_gc_s2l,
            'long_to_short': max_gc_l2s,
            'direction': direction,
        }
    
    return granger_results


def compute_phase_coupling(results, tau_values, fs):
    """Compute phase locking between short and long tau dynamics."""
    phase_results = {}
    
    short_tau = min(tau_values)
    long_tau = max(tau_values)
    
    for metric_name in DYNAMICS_GROUPS.keys():
        short_series = results[metric_name]['raw_by_tau'][short_tau]
        long_series = results[metric_name]['raw_by_tau'][long_tau]
        
        try:
            # Hilbert transform for phase
            short_analytic = hilbert(short_series - np.mean(short_series))
            long_analytic = hilbert(long_series - np.mean(long_series))
            
            short_phase = np.angle(short_analytic)
            long_phase = np.angle(long_analytic)
            
            # Phase difference
            phase_diff = short_phase - long_phase
            
            # PLV
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            mean_phase_diff = np.angle(np.mean(np.exp(1j * phase_diff)))
            
            if mean_phase_diff > 0.1:
                phase_direction = 'short_leads'
            elif mean_phase_diff < -0.1:
                phase_direction = 'long_leads'
            else:
                phase_direction = 'in_phase'
            
            phase_results[metric_name] = {
                'plv': plv,
                'mean_phase_diff': mean_phase_diff,
                'mean_phase_diff_deg': np.degrees(mean_phase_diff),
                'phase_direction': phase_direction,
            }
            
        except:
            phase_results[metric_name] = {
                'plv': 0, 'mean_phase_diff': 0, 'mean_phase_diff_deg': 0,
                'phase_direction': 'unknown'
            }
    
    return phase_results


def create_surrogate(signal, method='phase_shuffle'):
    """Create surrogate data by phase shuffling."""
    if method == 'phase_shuffle':
        fft = np.fft.fft(signal)
        phases = np.angle(fft)
        magnitudes = np.abs(fft)
        
        n = len(signal)
        random_phases = np.random.uniform(-np.pi, np.pi, n//2)
        new_phases = np.zeros(n)
        new_phases[1:n//2] = random_phases[1:]
        new_phases[n//2+1:] = -random_phases[1:][::-1]
        
        new_fft = magnitudes * np.exp(1j * new_phases)
        return np.real(np.fft.ifft(new_fft))
    
    return np.random.permutation(signal)


def analyze_channel(signal, fs, tau_values, window_samples, step_samples, n_surrogates=3):
    """Complete analysis for one channel."""
    n = len(signal)
    max_tau = max(tau_values)
    n_windows = (n - window_samples - 3 * max_tau) // step_samples
    
    # Real data
    print("    Computing real data with Takens embedding...")
    metrics_by_tau = {}
    for tau in tau_values:
        atoms = compute_phiid_for_tau(signal, tau, window_samples, step_samples, n_windows)
        metrics_by_tau[tau] = compute_dynamics_metrics(atoms)
    
    real_results = compute_normalized_divergence(metrics_by_tau, tau_values)
    cross_scale = compute_cross_scale_prediction(real_results, tau_values)
    granger = compute_granger_causality(real_results, tau_values)
    phase = compute_phase_coupling(real_results, tau_values, fs)
    
    # Surrogate data
    print(f"    Computing {n_surrogates} surrogates...")
    surrogate_divergences = {metric: [] for metric in DYNAMICS_GROUPS.keys()}
    
    for s in range(n_surrogates):
        surr_signal = create_surrogate(signal, method='phase_shuffle')
        surr_metrics_by_tau = {}
        for tau in tau_values:
            atoms = compute_phiid_for_tau(surr_signal, tau, window_samples, step_samples, n_windows)
            surr_metrics_by_tau[tau] = compute_dynamics_metrics(atoms)
        
        surr_results = compute_normalized_divergence(surr_metrics_by_tau, tau_values)
        
        for metric_name in DYNAMICS_GROUPS.keys():
            surrogate_divergences[metric_name].append(surr_results[metric_name]['normalized_divergence'])
    
    # Add surrogate stats
    for metric_name in DYNAMICS_GROUPS.keys():
        surr_divs = np.array(surrogate_divergences[metric_name])
        surr_mean = np.mean(surr_divs, axis=0)
        surr_std = np.std(surr_divs, axis=0)
        surr_std = np.maximum(surr_std, 0.001)
        
        real_div = real_results[metric_name]['normalized_divergence']
        zscore_div = (real_div - surr_mean) / surr_std
        
        real_results[metric_name]['surrogate_mean'] = surr_mean
        real_results[metric_name]['surrogate_std'] = surr_std
        real_results[metric_name]['divergence_zscore'] = zscore_div
        real_results[metric_name]['significant_integration'] = zscore_div < -2
        real_results[metric_name]['significant_fragmentation'] = zscore_div > 2
    
    return real_results, cross_scale, granger, phase, n_windows


def plot_analysis(results, cross_scale, granger, phase, time_sec, tau_values, fs, channel, save_dir):
    """Create visualization for analysis results."""
    
    for metric_name in DYNAMICS_GROUPS.keys():
        m = results[metric_name]
        pred = cross_scale[metric_name]
        gc = granger[metric_name]
        pc = phase[metric_name]
        
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle(f'Temporal Integration v3 (Takens): {channel} - {metric_name}', 
                    fontsize=14, fontweight='bold')
        
        # 1. TII timeseries with states
        ax1 = fig.add_subplot(3, 4, 1)
        tii = m['tii']
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        states = kmeans.fit_predict(tii.reshape(-1, 1))
        state_order = np.argsort(kmeans.cluster_centers_.flatten())
        state_map = {old: new for new, old in enumerate(state_order)}
        states = np.array([state_map[s] for s in states])
        
        colors_state = ['#ff6b6b', '#ffd93d', '#6bcf6b']
        for s in range(3):
            mask = states == s
            ax1.fill_between(time_sec, 0, 1, where=mask, alpha=0.3, 
                            color=colors_state[s], label=f'State {s}')
        ax1.plot(time_sec, tii, 'k-', linewidth=1.5, label='TII')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('TII')
        ax1.set_title('Temporal Integration Index')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_ylim(0, 1)
        ax1.grid(alpha=0.3)
        
        # 2. TII Distribution
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.hist(tii, bins=40, edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(tii), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(tii):.3f}')
        ax2.set_xlabel('TII')
        ax2.set_ylabel('Count')
        ax2.set_title('TII Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Divergence Z-score
        ax3 = fig.add_subplot(3, 4, 3)
        zscore_div = m['divergence_zscore']
        colors = np.where(zscore_div < -2, 'green',
                         np.where(zscore_div > 2, 'red', 'gray'))
        ax3.scatter(time_sec, zscore_div, c=colors, s=10, alpha=0.6)
        ax3.axhline(-2, color='green', linestyle='--', label='Integrated (z<-2)')
        ax3.axhline(2, color='red', linestyle='--', label='Fragmented (z>2)')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Z-score')
        ax3.set_title('Divergence vs Surrogate')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        
        # 4. Granger Causality
        ax4 = fig.add_subplot(3, 4, 4)
        bar_labels = ['Short→Long', 'Long→Short']
        bar_values = [gc['short_to_long'], gc['long_to_short']]
        bar_colors = ['#3498db', '#e74c3c']
        bars = ax4.bar(bar_labels, bar_values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax4.axhline(3.84, color='black', linestyle='--', label='F critical ≈ 3.84')
        ax4.set_ylabel('F-statistic')
        ax4.set_title(f'Granger Causality\nDirection: {gc["direction"]}')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3, axis='y')
        
        # 5. Cross-Tau Coupling matrix
        ax5 = fig.add_subplot(3, 4, 5)
        corr_matrix = m['corr_matrix']
        im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(tau_values)))
        ax5.set_yticks(range(len(tau_values)))
        ax5.set_xticklabels([f'{t}' for t in tau_values])
        ax5.set_yticklabels([f'{t}' for t in tau_values])
        ax5.set_xlabel('τ (samples)')
        ax5.set_ylabel('τ (samples)')
        ax5.set_title(f'Cross-τ Coupling (mean r={m["mean_cross_tau_corr"]:.3f})')
        plt.colorbar(im, ax=ax5)
        
        # 6. Phase Coupling
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.bar(['PLV'], [pc['plv']], color='purple', alpha=0.7)
        ax6.axhline(0.5, color='red', linestyle='--', label='Threshold')
        ax6.set_ylabel('PLV')
        ax6.set_title(f'Phase Locking\nPLV={pc["plv"]:.3f}, Dir: {pc["phase_direction"]}')
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3, axis='y')
        
        # 7. Cross-Scale Prediction
        ax7 = fig.add_subplot(3, 4, 7)
        ax7.bar(pred['offsets'], pred['correlations'], alpha=0.7)
        ax7.axvline(pred['best_offset'], color='red', linestyle='--', 
                   label=f'Best: {pred["best_offset"]}')
        ax7.set_xlabel('Temporal Offset (windows)')
        ax7.set_ylabel('Correlation')
        ax7.set_title('Cross-Scale Prediction')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Phase Difference
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.bar(['Phase Diff'], [abs(pc['mean_phase_diff_deg'])], color='purple', alpha=0.7)
        ax8.set_ylabel('Phase Difference (degrees)')
        ax8.set_title(f'Mean Phase Diff: {pc["mean_phase_diff_deg"]:.1f}°')
        ax8.axhline(45, color='orange', linestyle='--', label='45°')
        ax8.axhline(90, color='red', linestyle='--', label='90°')
        ax8.set_ylim(0, 180)
        ax8.legend(fontsize=8)
        ax8.grid(alpha=0.3, axis='y')
        
        # 9. Raw metric by tau
        ax9 = fig.add_subplot(3, 4, 9)
        colors_tau = plt.cm.viridis(np.linspace(0, 1, len(tau_values)))
        for i, tau in enumerate(tau_values):
            ax9.plot(time_sec, m['raw_by_tau'][tau], color=colors_tau[i], 
                    alpha=0.7, label=f'τ={tau}')
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel(f'{metric_name} (bits)')
        ax9.set_title(f'Raw {metric_name} by τ')
        ax9.legend(fontsize=7, title='τ (samples)')
        ax9.grid(alpha=0.3)
        
        # 10. Gradient Distribution
        ax10 = fig.add_subplot(3, 4, 10)
        raw_matrix = np.array([m['raw_by_tau'][tau] for tau in tau_values])
        gradients = []
        for t in range(raw_matrix.shape[1]):
            if np.std(raw_matrix[:, t]) > 0:
                slope = np.polyfit(range(len(tau_values)), raw_matrix[:, t], 1)[0]
                gradients.append(slope)
            else:
                gradients.append(0)
        gradients = np.array(gradients)
        ax10.hist(gradients, bins=40, edgecolor='black', alpha=0.7)
        ax10.axvline(0, color='red', linestyle='--')
        ax10.set_xlabel('τ Gradient')
        ax10.set_ylabel('Count')
        ax10.set_title('Gradient Distribution (+: increases with τ)')
        ax10.grid(alpha=0.3)
        
        # 11. Real vs Surrogate
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.hist(m['normalized_divergence'], bins=30, alpha=0.5, 
                label='Real', density=True, color='blue')
        ax11.hist(m['surrogate_mean'], bins=30, alpha=0.5, 
                label='Surrogate', density=True, color='gray')
        ax11.set_xlabel('Normalized Divergence')
        ax11.set_ylabel('Density')
        ax11.set_title('Real vs Surrogate')
        ax11.legend()
        ax11.grid(alpha=0.3)
        
        # 12. Summary
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.axis('off')
        
        pct_integrated = 100 * np.mean(m['significant_integration'])
        pct_fragmented = 100 * np.mean(m['significant_fragmentation'])
        
        summary_text = f"""
TEMPORAL INTEGRATION v3 SUMMARY
Channel: {channel}
Metric: {metric_name.upper()}
{'='*40}

APPROACH: Takens Delay Embedding
  τ values: {tau_values} samples
  Timeline: t, t+τ, t+2τ, t+3τ (REGULAR)
  Single lag parameter ✓

INTEGRATION INDEX (TII)
  Mean TII:        {np.mean(tii):.4f}
  TII Std:         {np.std(tii):.4f}

MULTI-SCALE DYNAMICS
  Mean Coupling:   {m['mean_cross_tau_corr']:.4f}
  Mean Gradient:   {np.mean(gradients):.4f}

GRANGER CAUSALITY
  Short→Long F:    {gc['short_to_long']:.3f}
  Long→Short F:    {gc['long_to_short']:.3f}
  Direction:       {gc['direction']}

PHASE COUPLING
  PLV:             {pc['plv']:.3f}
  Phase Diff:      {pc['mean_phase_diff_deg']:.1f}°

SURROGATE COMPARISON
  % Integrated:    {pct_integrated:.1f}%
  % Fragmented:    {pct_fragmented:.1f}%
        """
        ax12.text(0.02, 0.98, summary_text, transform=ax12.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / f"takens_{metric_name}_{channel}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # Summary figure for all metrics
    n_metrics = len(DYNAMICS_GROUPS)
    fig, axes = plt.subplots(2, n_metrics, figsize=(4*n_metrics, 8))
    
    for col, metric_name in enumerate(DYNAMICS_GROUPS.keys()):
        m = results[metric_name]
        
        # Row 1: TII timeseries
        ax = axes[0, col]
        ax.plot(time_sec, m['tii'], linewidth=1)
        ax.fill_between(time_sec, 0, 1, where=m['significant_integration'],
                       alpha=0.3, color='green')
        ax.set_ylabel('TII')
        ax.set_title(metric_name.replace('_', ' '))
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        # Row 2: TII Distribution
        ax = axes[1, col]
        ax.hist(m['tii'], bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(m['tii']), color='red', linestyle='--')
        ax.set_xlabel('TII')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'All Metrics Comparison (Takens v3): {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"takens_all_metrics_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print("TEMPORAL INTEGRATION ANALYSIS v3 (Takens Embedding)")
    print("Using direct 4-vector construction - TRUE single-lag approach")
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
    MAX_DURATION = 30
    
    # Tau values for Takens embedding (in samples)
    # These are the ONLY temporal parameters - no extra_lag!
    # Wide range covering multiple timescales:
    TAU_VALUES = [
        2,    # ~7 ms   - Fast gamma (>40 Hz dynamics)
        4,    # ~13 ms  - Gamma band (~30-40 Hz)
        8,    # ~27 ms  - Beta band (~20-30 Hz)
        15,   # ~50 ms  - Alpha/low beta (~15-20 Hz)
        30,   # ~100 ms - Alpha band (~10 Hz)
        45,   # ~150 ms - Theta band (~7 Hz)
        75,   # ~250 ms - Low theta (~4 Hz)
        120,  # ~400 ms - Delta band (~2.5 Hz)
        180,  # ~600 ms - Slow delta (~1.7 Hz)
    ]
    # Note: Need 3*τ samples for embedding, so τ=180 requires 540 samples
    
    WINDOW_SAMPLES = int(2.0 * FS)  # 2 seconds (600 samples) - enough for largest τ
    STEP_SAMPLES = int(0.1 * FS)    # 100ms step
    N_SURROGATES = 3
    
    CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    
    # Load data
    eeg_data, fs, all_channels = load_eeg_data(eeg_file, FS, MAX_DURATION)
    channels = [c for c in CHANNELS if c in all_channels]
    
    if not channels:
        channels = all_channels[:4]
    
    print(f"\nParameters:")
    print(f"  τ values (samples): {TAU_VALUES}")
    print(f"  τ values (ms): {[int(t/FS*1000) for t in TAU_VALUES]}")
    print(f"  Window: {WINDOW_SAMPLES/FS*1000:.0f}ms, Step: {STEP_SAMPLES/FS*1000:.0f}ms")
    print(f"  Surrogates: {N_SURROGATES}")
    print(f"  Channels: {channels}")
    print(f"\nTemporal structure: t, t+τ, t+2τ, t+3τ (perfectly regular)")
    
    all_summaries = []
    
    for channel in channels:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {channel}")
        print("=" * 50)
        
        signal = zscore(eeg_data[channel].values)
        
        # Estimate optimal tau
        tau_opt = estimate_optimal_tau(signal, FS)
        print(f"  Estimated optimal τ: {tau_opt} samples ({tau_opt/FS*1000:.1f}ms)")
        
        # Run analysis
        results, cross_scale, granger, phase, n_windows = analyze_channel(
            signal, FS, TAU_VALUES, WINDOW_SAMPLES, STEP_SAMPLES, N_SURROGATES
        )
        
        time_sec = np.arange(n_windows) * STEP_SAMPLES / FS
        
        # Generate plots
        print("  Generating visualization...")
        plot_analysis(
            results, cross_scale, granger, phase, 
            time_sec, TAU_VALUES, FS, channel, RESULTS_DIR
        )
        
        # Collect summary
        summary = {'channel': channel, 'optimal_tau': tau_opt}
        for metric_name in DYNAMICS_GROUPS.keys():
            m = results[metric_name]
            gc = granger[metric_name]
            pc = phase[metric_name]
            summary[f'{metric_name}_mean_tii'] = np.mean(m['tii'])
            summary[f'{metric_name}_pct_integrated'] = 100 * np.mean(m['significant_integration'])
            summary[f'{metric_name}_mean_coupling'] = m['mean_cross_tau_corr']
            summary[f'{metric_name}_granger_dir'] = gc['direction']
            summary[f'{metric_name}_plv'] = pc['plv']
        
        all_summaries.append(summary)
    
    # Save summary
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(RESULTS_DIR / "integration_summary_v3.csv", index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
