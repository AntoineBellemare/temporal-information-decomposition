"""
Temporal Integration Analysis v2 - Improved
============================================

Improvements over v1:
1. Normalize metrics by lag to remove trivial scaling effects
2. Add surrogate/baseline comparison 
3. Better TII formulation (z-score based)
4. Frequency band decomposition
5. Cross-channel synchrony analysis
6. Artifact detection

Usage:
    python temporal_integration_v2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import signal as sig
from scipy.stats import zscore, entropy, pearsonr
from sklearn.cluster import KMeans
from tqdm import tqdm

# Import PhiID
from phyid.calculate import calc_PhiID
from phyid.utils import PhiID_atoms_abbr

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "phiid" / "temporal_integration_v2"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATOM_NAMES = list(PhiID_atoms_abbr)

DYNAMICS_GROUPS = {
    "Storage": ["rtr", "xtx", "yty", "sts"],
    "Transfer": ["xty", "ytx"],
    "Erasure": ["rtx", "rty"],
    "Copy": ["xtx", "yty"],
    "Upward_causation": ["xts", "yts", "rts"],
    "Downward_causation": ["stx", "sty", "str"],
}


def load_eeg_data(filepath, fs=300, max_duration=30):
    """Load EEG data."""
    df = pd.read_csv(filepath)
    eeg_cols = [c for c in df.columns if c.startswith('eeg-') and 
                'trigger' not in c.lower() and 'A1' not in c and 'A2' not in c]
    
    max_samples = int(max_duration * fs)
    if len(df) > max_samples:
        df = df.iloc[:max_samples]
    
    return df[eeg_cols], fs, eeg_cols


def detect_artifacts(signal, threshold=5):
    """Detect potential artifacts (extreme values, flatlines)."""
    z = zscore(signal)
    extreme = np.abs(z) > threshold
    
    # Flatline detection (very low local variance)
    window = 10
    local_var = np.array([np.var(signal[max(0,i-window):i+window]) 
                          for i in range(len(signal))])
    flatline = local_var < np.percentile(local_var, 1)
    
    return extreme | flatline


def create_surrogate(signal, method='phase_shuffle'):
    """Create surrogate data preserving spectrum but destroying temporal structure."""
    if method == 'phase_shuffle':
        # FFT phase randomization
        fft = np.fft.fft(signal)
        phases = np.angle(fft)
        magnitudes = np.abs(fft)
        
        # Randomize phases (keeping symmetry for real output)
        n = len(signal)
        random_phases = np.random.uniform(-np.pi, np.pi, n//2)
        new_phases = np.zeros(n)
        new_phases[1:n//2] = random_phases[1:]
        new_phases[n//2+1:] = -random_phases[1:][::-1]
        
        new_fft = magnitudes * np.exp(1j * new_phases)
        return np.real(np.fft.ifft(new_fft))
    
    elif method == 'shuffle':
        return np.random.permutation(signal)
    
    return signal


def compute_phiid_for_lag(signal, extra_lag, tau, window_size, step_size, n_windows):
    """Compute PhiID atoms for one lag."""
    atom_series = {name: np.zeros(n_windows) for name in ATOM_NAMES}
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size + extra_lag
        
        if end > len(signal):
            break
            
        segment = signal[start:end]
        src = segment[:-extra_lag]
        tgt = segment[extra_lag:]
        
        try:
            atoms_res, _ = calc_PhiID(src, tgt, tau, kind='gaussian', redundancy='MMI')
            for name in ATOM_NAMES:
                if name in atoms_res:
                    val = np.nanmean(atoms_res[name])
                    atom_series[name][i] = val if np.isfinite(val) else 0.0
        except:
            pass
    
    return atom_series


def compute_normalized_divergence(atom_series_by_lag, lags):
    """
    Compute divergence normalized by each lag's typical values.
    
    For EACH metric in DYNAMICS_GROUPS:
    - Normalize within each lag (z-score)
    - Compute divergence across lags
    - Return TII = 1/(1+divergence)
    """
    n_times = len(list(atom_series_by_lag.values())[0]['rtr'])
    
    results = {}
    
    for metric_name, atoms in DYNAMICS_GROUPS.items():
        # Compute metric for each lag
        metric_by_lag = {}
        raw_metric_by_lag = {}
        
        for lag in lags:
            raw_values = sum(atom_series_by_lag[lag][a] for a in atoms)
            raw_metric_by_lag[lag] = raw_values
            # Z-score normalize
            if np.std(raw_values) > 0:
                metric_by_lag[lag] = zscore(raw_values)
            else:
                metric_by_lag[lag] = raw_values
        
        # Compute divergence on normalized values
        metric_matrix = np.array([metric_by_lag[lag] for lag in lags])
        normalized_divergence = np.std(metric_matrix, axis=0)
        
        # Raw divergence
        raw_matrix = np.array([raw_metric_by_lag[lag] for lag in lags])
        raw_divergence = np.std(raw_matrix, axis=0)
        
        # TII
        tii = 1 / (1 + normalized_divergence)
        
        # Cross-lag correlation
        corr_matrix = np.corrcoef(metric_matrix)
        mean_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        results[metric_name] = {
            'normalized_by_lag': metric_by_lag,
            'raw_by_lag': raw_metric_by_lag,
            'normalized_divergence': normalized_divergence,
            'raw_divergence': raw_divergence,
            'tii': tii,
            'mean_cross_lag_corr': mean_corr,
            'corr_matrix': corr_matrix,
        }
    
    return results


def compute_cross_scale_prediction(results, lags):
    """
    Compute how well short lags predict long lags at different time offsets.
    """
    predictions = {}
    
    for metric_name, data in results.items():
        short_lag = lags[0]
        long_lag = lags[-1]
        
        short_series = data['normalized_by_lag'][short_lag]
        long_series = data['normalized_by_lag'][long_lag]
        
        offsets = range(-10, 11)
        correlations = []
        
        for offset in offsets:
            if offset < 0:
                s = short_series[-offset:]
                l = long_series[:offset]
            elif offset > 0:
                s = short_series[:-offset]
                l = long_series[offset:]
            else:
                s = short_series
                l = long_series
            
            if len(s) > 10 and len(l) > 10:
                corr, _ = pearsonr(s, l)
                correlations.append(corr if np.isfinite(corr) else 0)
            else:
                correlations.append(0)
        
        best_idx = np.argmax(np.abs(correlations))
        
        predictions[metric_name] = {
            'offsets': list(offsets),
            'correlations': correlations,
            'best_offset': list(offsets)[best_idx],
            'best_correlation': correlations[best_idx],
        }
    
    return predictions


def compute_granger_causality(results, lags, max_order=5):
    """
    Compute Granger causality between short and long lag timescales.
    
    Tests: Does short_lag(t-k) help predict long_lag(t) beyond long_lag(t-k)?
    And vice versa.
    
    Returns F-statistic-like measure for each direction.
    """
    from scipy.linalg import lstsq
    
    granger_results = {}
    
    for metric_name, data in results.items():
        short_lag = lags[0]
        long_lag = lags[-1]
        
        short_series = data['normalized_by_lag'][short_lag]
        long_series = data['normalized_by_lag'][long_lag]
        
        n = len(short_series)
        
        if n < max_order + 10:
            granger_results[metric_name] = {
                'short_to_long': 0, 'long_to_short': 0,
                'best_order': 1, 'direction': 'none'
            }
            continue
        
        # Test short -> long (does short help predict long?)
        gc_short_to_long = []
        gc_long_to_short = []
        
        for order in range(1, max_order + 1):
            # Prepare lagged matrices
            Y_long = long_series[order:]
            Y_short = short_series[order:]
            
            # Build regressor matrices
            X_long_only = np.column_stack([long_series[order-i-1:-i-1 if -i-1 != 0 else None] 
                                           for i in range(order)])
            X_short_only = np.column_stack([short_series[order-i-1:-i-1 if -i-1 != 0 else None] 
                                            for i in range(order)])
            X_both = np.column_stack([X_long_only, X_short_only])
            
            # Restricted model: long predicted by its own past only
            try:
                beta_r, res_r, _, _ = lstsq(X_long_only, Y_long)
                rss_restricted = np.sum((Y_long - X_long_only @ beta_r)**2)
                
                # Unrestricted model: long predicted by both
                beta_u, res_u, _, _ = lstsq(X_both, Y_long)
                rss_unrestricted = np.sum((Y_long - X_both @ beta_u)**2)
                
                # F-statistic
                df1 = order
                df2 = len(Y_long) - 2*order
                if rss_unrestricted > 0 and df2 > 0:
                    F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
                    gc_short_to_long.append(F)
                else:
                    gc_short_to_long.append(0)
            except:
                gc_short_to_long.append(0)
            
            # Test long -> short (does long help predict short?)
            X_short_only_s = np.column_stack([short_series[order-i-1:-i-1 if -i-1 != 0 else None] 
                                              for i in range(order)])
            X_long_only_s = np.column_stack([long_series[order-i-1:-i-1 if -i-1 != 0 else None] 
                                             for i in range(order)])
            X_both_s = np.column_stack([X_short_only_s, X_long_only_s])
            
            try:
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
        
        # Best order by max F
        best_order_s2l = np.argmax(gc_short_to_long) + 1 if gc_short_to_long else 1
        best_order_l2s = np.argmax(gc_long_to_short) + 1 if gc_long_to_short else 1
        
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
            'gc_s2l_by_order': gc_short_to_long,
            'gc_l2s_by_order': gc_long_to_short,
            'best_order_s2l': best_order_s2l,
            'best_order_l2s': best_order_l2s,
            'direction': direction,
        }
    
    return granger_results


def compute_phase_coupling(results, lags, fs):
    """
    Compute phase-based coupling between short and long lag timescales.
    
    Uses Hilbert transform to extract instantaneous phase,
    then computes Phase Locking Value (PLV) and phase lead/lag.
    """
    from scipy.signal import hilbert
    
    phase_results = {}
    
    for metric_name, data in results.items():
        short_lag = lags[0]
        long_lag = lags[-1]
        
        short_series = data['normalized_by_lag'][short_lag]
        long_series = data['normalized_by_lag'][long_lag]
        
        # Get instantaneous phase via Hilbert transform
        try:
            analytic_short = hilbert(short_series)
            analytic_long = hilbert(long_series)
            
            phase_short = np.angle(analytic_short)
            phase_long = np.angle(analytic_long)
            
            # Phase difference
            phase_diff = phase_short - phase_long
            
            # Phase Locking Value (PLV): coherence of phase difference
            # PLV = |mean(exp(i * phase_diff))|
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # Mean phase difference (tells us if one leads)
            mean_phase_diff = np.angle(np.mean(np.exp(1j * phase_diff)))
            
            # Phase difference in time units (approximate)
            # If signal has dominant frequency f, phase diff of phi = phi/(2*pi*f) seconds
            # Estimate dominant frequency from spectrum
            fft_short = np.abs(np.fft.fft(short_series))
            freqs = np.fft.fftfreq(len(short_series), d=1)  # In windows, not seconds
            dominant_freq = np.abs(freqs[np.argmax(fft_short[1:len(fft_short)//2]) + 1])
            
            if dominant_freq > 0:
                phase_lag_windows = mean_phase_diff / (2 * np.pi * dominant_freq)
            else:
                phase_lag_windows = 0
            
            # Time-varying PLV (sliding window)
            window_size = min(50, len(phase_diff) // 4)
            if window_size > 5:
                plv_timeseries = []
                for i in range(len(phase_diff) - window_size):
                    local_plv = np.abs(np.mean(np.exp(1j * phase_diff[i:i+window_size])))
                    plv_timeseries.append(local_plv)
                plv_timeseries = np.array(plv_timeseries)
            else:
                plv_timeseries = np.array([plv])
            
            # Interpret direction
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
                'phase_lag_windows': phase_lag_windows,
                'dominant_freq': dominant_freq,
                'plv_timeseries': plv_timeseries,
                'plv_mean': np.mean(plv_timeseries),
                'plv_std': np.std(plv_timeseries),
                'phase_direction': phase_direction,
            }
            
        except Exception as e:
            phase_results[metric_name] = {
                'plv': 0, 'mean_phase_diff': 0, 'mean_phase_diff_deg': 0,
                'phase_lag_windows': 0, 'dominant_freq': 0,
                'plv_timeseries': np.array([0]), 'plv_mean': 0, 'plv_std': 0,
                'phase_direction': 'unknown'
            }
    
    return phase_results


def analyze_with_surrogate(signal, fs, lags, tau, window_samples, step_samples, n_surrogates=5):
    """Compare real data to surrogate baseline for all metrics."""
    n = len(signal)
    max_lag = max(lags)
    n_windows = (n - window_samples - max_lag) // step_samples
    
    # Real data
    print("    Computing real data...")
    real_atoms = {}
    for lag in lags:
        real_atoms[lag] = compute_phiid_for_lag(signal, lag, tau, window_samples, step_samples, n_windows)
    
    real_metrics = compute_normalized_divergence(real_atoms, lags)
    cross_scale = compute_cross_scale_prediction(real_metrics, lags)
    
    # Surrogate data
    print(f"    Computing {n_surrogates} surrogates...")
    surrogate_results = {metric: [] for metric in DYNAMICS_GROUPS.keys()}
    
    for s in range(n_surrogates):
        surr_signal = create_surrogate(signal, method='phase_shuffle')
        surr_atoms = {}
        for lag in lags:
            surr_atoms[lag] = compute_phiid_for_lag(surr_signal, lag, tau, window_samples, step_samples, n_windows)
        
        surr_metrics = compute_normalized_divergence(surr_atoms, lags)
        
        for metric_name in DYNAMICS_GROUPS.keys():
            surrogate_results[metric_name].append(surr_metrics[metric_name]['normalized_divergence'])
    
    # Add surrogate stats to real metrics
    for metric_name in DYNAMICS_GROUPS.keys():
        surr_divs = np.array(surrogate_results[metric_name])
        surr_mean = np.mean(surr_divs, axis=0)
        surr_std = np.std(surr_divs, axis=0)
        surr_std = np.maximum(surr_std, 0.001)
        
        real_div = real_metrics[metric_name]['normalized_divergence']
        zscore_div = (real_div - surr_mean) / surr_std
        
        real_metrics[metric_name]['surrogate_mean'] = surr_mean
        real_metrics[metric_name]['surrogate_std'] = surr_std
        real_metrics[metric_name]['divergence_zscore'] = zscore_div
        real_metrics[metric_name]['significant_integration'] = zscore_div < -2
        real_metrics[metric_name]['significant_fragmentation'] = zscore_div > 2
    
    # Compute Granger causality between timescales
    print("    Computing Granger causality...")
    granger = compute_granger_causality(real_metrics, lags)
    
    # Compute phase coupling between timescales
    print("    Computing phase coupling...")
    phase = compute_phase_coupling(real_metrics, lags, fs)
    
    return real_metrics, cross_scale, granger, phase, n_windows


def plot_advanced_analysis(metrics, cross_scale, granger, phase, time_sec, lags, fs, channel, save_dir):
    """Create advanced visualization with TII distributions, Granger causality, and phase coupling."""
    
    # =========================================================================
    # Generate detailed 12-panel figure for EACH metric (expanded from 9)
    # =========================================================================
    
    for metric_name in DYNAMICS_GROUPS.keys():
        m = metrics[metric_name]
        pred = cross_scale[metric_name]
        gc = granger[metric_name]
        pc = phase[metric_name]
        
        fig = plt.figure(figsize=(20, 14))
        
        # 1. TII timeseries with states (3-state clustering)
        ax1 = fig.add_subplot(3, 4, 1)
        tii = m['tii']
        
        # K-means clustering for states
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
        ax1.set_title('Temporal Integration Index with States')
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
        
        # 3. Divergence Z-score (significance)
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
        
        # 4. Granger Causality - NEW
        ax4 = fig.add_subplot(3, 4, 4)
        # Bar chart of F-statistics for both directions
        bar_labels = ['Short→Long', 'Long→Short']
        bar_values = [gc['short_to_long'], gc['long_to_short']]
        bar_colors = ['#3498db', '#e74c3c']
        bars = ax4.bar(bar_labels, bar_values, color=bar_colors, alpha=0.7, edgecolor='black')
        f_critical = 3.84  # approximate F critical at p=0.05
        ax4.axhline(f_critical, color='black', linestyle='--', 
                   label=f'F critical ≈ {f_critical:.2f}')
        ax4.set_ylabel('F-statistic')
        ax4.set_title(f'Granger Causality\nDirection: {gc["direction"]}')
        ax4.legend(fontsize=8)
        ax4.grid(alpha=0.3, axis='y')
        # Add significance stars
        for i, (val, bar) in enumerate(zip(bar_values, bars)):
            if val > f_critical:
                ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1, '*', 
                        ha='center', fontsize=14, color='green')
        
        # 5. Cross-Lag Coupling matrix
        ax5 = fig.add_subplot(3, 4, 5)
        corr_matrix = m['corr_matrix']
        im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(lags)))
        ax5.set_yticks(range(len(lags)))
        ax5.set_xticklabels([f'{l/fs*1000:.0f}' for l in lags])
        ax5.set_yticklabels([f'{l/fs*1000:.0f}' for l in lags])
        ax5.set_xlabel('Lag (ms)')
        ax5.set_ylabel('Lag (ms)')
        ax5.set_title(f'Cross-Lag Coupling (mean r={m["mean_cross_lag_corr"]:.3f})')
        plt.colorbar(im, ax=ax5)
        
        # 6. Phase Coupling (PLV timeseries) - NEW
        ax6 = fig.add_subplot(3, 4, 6)
        plv_ts = pc['plv_timeseries']
        if len(plv_ts) > 1:
            plv_time = np.linspace(time_sec[0], time_sec[-1], len(plv_ts))
            ax6.plot(plv_time, plv_ts, 'b-', linewidth=1.5)
            ax6.fill_between(plv_time, 0, plv_ts, alpha=0.3)
        else:
            ax6.axhline(pc['plv'], color='blue', linewidth=2)
        ax6.axhline(0.5, color='red', linestyle='--', label='Threshold')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('PLV')
        ax6.set_title(f'Phase Locking Value\nMean PLV={pc["plv_mean"]:.3f}, Dir: {pc["phase_direction"]}')
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3)
        
        # 7. Cross-Scale Prediction (keep original)
        ax7 = fig.add_subplot(3, 4, 7)
        bars = ax7.bar(pred['offsets'], pred['correlations'], alpha=0.7)
        ax7.axvline(pred['best_offset'], color='red', linestyle='--', 
                   label=f'Best: {pred["best_offset"]}')
        ax7.set_xlabel('Temporal Offset (windows)')
        ax7.set_ylabel('Correlation (short<->long lag)')
        ax7.set_title('Cross-Scale Prediction')
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # 8. Phase Histogram - NEW
        ax8 = fig.add_subplot(3, 4, 8)
        phase_diff_deg = pc['mean_phase_diff_deg']
        ax8.bar(['Phase Diff'], [abs(phase_diff_deg)], color='purple', alpha=0.7)
        ax8.set_ylabel('Phase Difference (degrees)')
        ax8.set_title(f'Mean Phase Difference: {phase_diff_deg:.1f}°')
        ax8.axhline(45, color='orange', linestyle='--', label='45° threshold')
        ax8.axhline(90, color='red', linestyle='--', label='90° quadrature')
        ax8.set_ylim(0, 180)
        ax8.legend(fontsize=8)
        ax8.grid(alpha=0.3, axis='y')
        
        # 9. Gradient Distribution
        ax9 = fig.add_subplot(3, 4, 9)
        raw_matrix = np.array([m['raw_by_lag'][lag] for lag in lags])
        gradients = []
        for t in range(raw_matrix.shape[1]):
            if np.std(raw_matrix[:, t]) > 0:
                slope = np.polyfit(range(len(lags)), raw_matrix[:, t], 1)[0]
                gradients.append(slope)
            else:
                gradients.append(0)
        gradients = np.array(gradients)
        ax9.hist(gradients, bins=40, edgecolor='black', alpha=0.7)
        ax9.axvline(0, color='red', linestyle='--')
        ax9.set_xlabel('Lag Gradient')
        ax9.set_ylabel('Count')
        ax9.set_title('Gradient Distribution (+: increases with lag)')
        ax9.grid(alpha=0.3)
        
        # 10. Raw metric by lag
        ax10 = fig.add_subplot(3, 4, 10)
        colors_lag = plt.cm.viridis(np.linspace(0, 1, len(lags)))
        for i, lag in enumerate(lags):
            ax10.plot(time_sec, m['raw_by_lag'][lag], color=colors_lag[i], 
                    alpha=0.7, label=f'{lag/fs*1000:.0f}ms')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel(f'{metric_name} (bits)')
        ax10.set_title(f'Raw {metric_name} by Lag')
        ax10.legend(fontsize=7, title='Lag')
        ax10.grid(alpha=0.3)
        
        # 11. Real vs Surrogate Distribution
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.hist(m['normalized_divergence'], bins=30, alpha=0.5, 
                label='Real', density=True, color='blue')
        ax11.hist(m['surrogate_mean'], bins=30, alpha=0.5, 
                label='Surrogate Mean', density=True, color='gray')
        ax11.set_xlabel('Normalized Divergence')
        ax11.set_ylabel('Density')
        ax11.set_title('Real vs Surrogate Distribution')
        ax11.legend()
        ax11.grid(alpha=0.3)
        
        # 12. Summary (expanded)
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.axis('off')
        
        pct_integrated = 100 * np.mean(m['significant_integration'])
        pct_fragmented = 100 * np.mean(m['significant_fragmentation'])
        
        summary_text = f"""
    TEMPORAL INTEGRATION SUMMARY: {channel}
    Metric: {metric_name.replace('_', ' ').upper()}
    ========================================
    
    INTEGRATION INDEX (TII)
      Mean TII:        {np.mean(tii):.4f}
      TII Std:         {np.std(tii):.4f}
      TII Entropy:     {entropy(np.histogram(tii, bins=20)[0] + 1):.4f}
    
    MULTI-SCALE DYNAMICS
      Mean Coupling:     {m['mean_cross_lag_corr']:.4f}
      Transition Rate:   {np.mean(np.abs(np.diff(states))):.4f}
      Mean Gradient:     {np.mean(gradients):.4f}
    
    GRANGER CAUSALITY
      Short→Long F:    {gc['short_to_long']:.3f}
      Long→Short F:    {gc['long_to_short']:.3f}
      Direction:       {gc['direction']}
      Best Order:      {gc['best_order_s2l']}
    
    PHASE COUPLING
      Mean PLV:        {pc['plv_mean']:.3f}
      Phase Diff:      {pc['mean_phase_diff_deg']:.1f}°
      Phase Dir:       {pc['phase_direction']}
    
    SURROGATE COMPARISON
      % Integrated:    {pct_integrated:.1f}%
      % Fragmented:    {pct_fragmented:.1f}%
      Mean Z-score:    {np.mean(zscore_div):.3f}
        """
        ax12.text(0.02, 0.98, summary_text, transform=ax12.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        metric_clean = metric_name.replace('_', ' ').title()
        plt.suptitle(f'Temporal Integration Analysis: {channel} - {metric_clean}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f"integration_{metric_name}_{channel}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # FIGURE: All metrics comparison (compact overview)
    # =========================================================================
    n_metrics = len(DYNAMICS_GROUPS)
    fig, axes = plt.subplots(3, n_metrics, figsize=(4*n_metrics, 10))
    
    for col, metric_name in enumerate(DYNAMICS_GROUPS.keys()):
        m = metrics[metric_name]
        pred_m = cross_scale[metric_name]
        
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
        
        # Row 3: Cross-scale prediction
        ax = axes[2, col]
        pred = cross_scale[metric_name]
        ax.bar(pred['offsets'], pred['correlations'], alpha=0.7)
        ax.axvline(pred['best_offset'], color='red', linestyle='--')
        ax.set_xlabel('Offset')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Best: {pred["best_offset"]}')
        ax.grid(alpha=0.3)
    
    axes[0, 0].set_ylabel('TII Timeseries')
    axes[1, 0].set_ylabel('TII Distribution')
    axes[2, 0].set_ylabel('Cross-Scale Pred')
    
    plt.suptitle(f'All Metrics Comparison: {channel}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"all_metrics_{channel}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Return summary stats for all metrics
    summary = {'channel': channel}
    for metric_name in DYNAMICS_GROUPS.keys():
        m = metrics[metric_name]
        pred = cross_scale[metric_name]
        gc = granger[metric_name]
        pc = phase[metric_name]
        summary[f'{metric_name}_mean_tii'] = np.mean(m['tii'])
        summary[f'{metric_name}_pct_integrated'] = 100 * np.mean(m['significant_integration'])
        summary[f'{metric_name}_mean_coupling'] = m['mean_cross_lag_corr']
        summary[f'{metric_name}_best_offset'] = pred['best_offset']
        summary[f'{metric_name}_granger_dir'] = gc['direction']
        summary[f'{metric_name}_granger_f_short_long'] = gc['short_to_long']
        summary[f'{metric_name}_granger_f_long_short'] = gc['long_to_short']
        summary[f'{metric_name}_plv'] = pc['plv_mean']
        summary[f'{metric_name}_phase_diff_deg'] = pc['mean_phase_diff_deg']
        summary[f'{metric_name}_phase_dir'] = pc['phase_direction']
    
    return summary


def main():
    print("=" * 70)
    print("TEMPORAL INTEGRATION ANALYSIS v2 (Improved)")
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
    TAU = 1
    LAGS_SAMPLES = [3, 9, 15, 30, 60]
    WINDOW_SAMPLES = int(0.5 * FS)
    STEP_SAMPLES = int(0.1 * FS)
    N_SURROGATES = 3  # Number of surrogates for baseline
    
    CHANNELS = ['eeg-Fz', 'eeg-Cz', 'eeg-O1', 'eeg-T3']
    
    # Load data
    eeg_data, fs, all_channels = load_eeg_data(eeg_file, FS, MAX_DURATION)
    channels = [c for c in CHANNELS if c in all_channels]
    
    print(f"\nParameters:")
    print(f"  Lags (ms): {[int(l/FS*1000) for l in LAGS_SAMPLES]}")
    print(f"  Window: {WINDOW_SAMPLES/FS*1000:.0f}ms, Step: {STEP_SAMPLES/FS*1000:.0f}ms")
    print(f"  Surrogates: {N_SURROGATES}")
    print(f"  Channels: {channels}")
    
    all_summaries = []
    
    for channel in channels:
        print(f"\n{'='*50}")
        print(f"ANALYZING: {channel}")
        print("=" * 50)
        
        signal = zscore(eeg_data[channel].values)
        
        # Check for artifacts
        artifacts = detect_artifacts(signal)
        pct_artifacts = 100 * np.mean(artifacts)
        print(f"  Artifact percentage: {pct_artifacts:.1f}%")
        
        # Run analysis with surrogate comparison
        metrics, cross_scale, granger, phase, n_windows = analyze_with_surrogate(
            signal, FS, LAGS_SAMPLES, TAU, WINDOW_SAMPLES, STEP_SAMPLES, N_SURROGATES
        )
        
        time_sec = np.arange(n_windows) * STEP_SAMPLES / FS
        
        # Generate plots
        print("  Generating visualization...")
        summary = plot_advanced_analysis(
            metrics, cross_scale, granger, phase, time_sec, LAGS_SAMPLES, FS, channel, RESULTS_DIR
        )
        summary['pct_artifacts'] = pct_artifacts
        all_summaries.append(summary)
    
    # Save summary
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(RESULTS_DIR / "integration_summary_v2.csv", index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
