"""
Kuramoto Oscillators: Temporal PID Analysis
============================================

This script investigates how coupling strength affects the TEMPORAL structure
of a single Kuramoto oscillator's signal using time-delay embedding PID.

KEY DISTINCTION: We analyze time-delay embeddings of ONE signal, NOT 
cross-oscillator relationships. The question is:
    "How does coupling change the temporal fingerprint of an oscillator?"

PREDICTIONS:
-----------
1. LOW COUPLING (K << Kc):
   - Oscillators are nearly independent
   - Each oscillator's signal is quasi-periodic at its natural frequency ωᵢ
   - Temporal PID: HIGH REDUNDANCY at lags matching the period (and multiples)
   - Low synergy (predictable, linear dynamics)

2. NEAR CRITICAL COUPLING (K ≈ Kc):
   - Partial synchronization, intermittent dynamics
   - Temporal structure becomes more complex
   - Temporal PID: Possible ENHANCED SYNERGY (nonlinear, history-dependent)
   - Redundancy may decrease (less periodic)

3. HIGH COUPLING (K >> Kc):
   - Full synchronization, all oscillators lock to mean frequency
   - Signal becomes very periodic at collective frequency
   - Temporal PID: HIGH REDUNDANCY at synchronized period
   - Low synergy (predictable again)

4. MEAN FIELD (order parameter R):
   - Low K: R fluctuates (incoherent) → complex temporal structure
   - High K: R ≈ 1 constant (coherent) → trivial temporal structure
   - Near critical: R shows critical fluctuations → rich dynamics

Usage:
    python kuramoto_temporal_pid.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
from datetime import datetime

import dit
from dit.pid import PID_MMI
from dit import Distribution

# Output directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent  # Go up from scripts/pid/ to project root
OUTPUT_DIR = PROJECT_DIR / "results" / "pid" / "kuramoto"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# KURAMOTO MODEL
# =============================================================================

def simulate_kuramoto(N=50, K=1.0, T=200, dt=0.01, natural_freq_std=1.0, seed=42):
    """
    Simulate Kuramoto oscillators.
    
    dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ - θᵢ)
    
    Parameters
    ----------
    N : int
        Number of oscillators
    K : float
        Coupling strength (critical Kc ≈ 2*std(ω) for Lorentzian)
    T : float
        Total simulation time
    dt : float
        Time step
    natural_freq_std : float
        Standard deviation of natural frequencies (drawn from Gaussian)
    seed : int
        Random seed
    
    Returns
    -------
    t : array
        Time points
    theta : array (n_steps, N)
        Phase of each oscillator over time
    omega : array (N,)
        Natural frequencies
    R : array (n_steps,)
        Order parameter (synchronization measure)
    """
    np.random.seed(seed)
    
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    
    # Natural frequencies (Gaussian distribution)
    omega = np.random.normal(0, natural_freq_std, N)
    
    # Initial phases (uniform random)
    theta = np.zeros((n_steps, N))
    theta[0] = np.random.uniform(0, 2*np.pi, N)
    
    # Euler integration
    for i in range(1, n_steps):
        # Compute coupling term for each oscillator
        phase_diff = theta[i-1, :, np.newaxis] - theta[i-1, np.newaxis, :]
        coupling = (K / N) * np.sum(np.sin(-phase_diff), axis=1)
        
        # Update phases
        theta[i] = theta[i-1] + dt * (omega + coupling)
    
    # Compute order parameter R(t) = |1/N Σ exp(i*θⱼ)|
    z = np.exp(1j * theta)
    R = np.abs(np.mean(z, axis=1))
    
    return t, theta, omega, R


def get_oscillator_signal(theta, oscillator_idx=0, signal_type='sin'):
    """
    Extract a 1D signal from an oscillator's phase.
    
    Parameters
    ----------
    theta : array (n_steps, N)
        Phase trajectories
    oscillator_idx : int
        Which oscillator to use
    signal_type : str
        'sin': sin(θ), 'cos': cos(θ), 'phase': θ (unwrapped)
    """
    phase = theta[:, oscillator_idx]
    
    if signal_type == 'sin':
        return np.sin(phase)
    elif signal_type == 'cos':
        return np.cos(phase)
    elif signal_type == 'phase':
        return np.unwrap(phase)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


def get_mean_field_signal(theta, signal_type='R'):
    """
    Extract mean field signal.
    
    Parameters
    ----------
    theta : array (n_steps, N)
        Phase trajectories
    signal_type : str
        'R': order parameter magnitude
        'Psi': mean phase (collective phase)
        'sin_Psi': sin of collective phase
    """
    z = np.exp(1j * theta)
    mean_z = np.mean(z, axis=1)
    
    if signal_type == 'R':
        return np.abs(mean_z)
    elif signal_type == 'Psi':
        return np.unwrap(np.angle(mean_z))
    elif signal_type == 'sin_Psi':
        return np.sin(np.angle(mean_z))
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")


# =============================================================================
# PID FUNCTIONS (from temporal_pid_analysis.py)
# =============================================================================

def discretize_timeseries(x, n_bins=4, method='quantile'):
    """Discretize continuous time series into symbols."""
    if method == 'quantile':
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x, percentiles)
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
    
    outcomes = []
    for t in range(max_lag, n):
        outcome = tuple(x[t - lag] for lag in lags) + (x[t],)
        outcomes.append(''.join(map(str, outcome)))
    
    counts = Counter(outcomes)
    total = sum(counts.values())
    
    outcomes_list = list(counts.keys())
    probs = [counts[o] / total for o in outcomes_list]
    
    return Distribution(outcomes_list, probs)


def compute_pid_summary(dist, pid_class=PID_MMI):
    """Extract key PID values from a distribution."""
    pid = pid_class(dist)
    
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


def compute_lag_sweep(x_discrete, max_lag=10):
    """Compute PID for all lag pairs up to max_lag."""
    results = []
    
    for lag1 in range(1, max_lag):
        for lag2 in range(lag1 + 1, max_lag + 1):
            dist = build_embedding_distribution(x_discrete, lags=[lag1, lag2])
            pid = compute_pid_summary(dist)
            pid['lag1'] = lag1
            pid['lag2'] = lag2
            results.append(pid)
    
    return pd.DataFrame(results)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_coupling(K, N=50, T=200, dt=0.01, max_lag=15, 
                            downsample=10, n_bins=4, seed=42):
    """
    Analyze temporal PID for a single coupling strength.
    
    Parameters
    ----------
    K : float
        Coupling strength
    downsample : int
        Take every Nth sample (reduces autocorrelation, speeds up)
    """
    # Simulate
    t, theta, omega, R = simulate_kuramoto(N=N, K=K, T=T, dt=dt, seed=seed)
    
    # Downsample
    t = t[::downsample]
    theta = theta[::downsample]
    R = R[::downsample]
    
    # Get signals
    # Use oscillator 0 (arbitrary choice)
    x_osc = get_oscillator_signal(theta, oscillator_idx=0, signal_type='sin')
    x_osc_discrete = discretize_timeseries(x_osc, n_bins=n_bins)
    
    # Mean field
    x_R = R  # Order parameter
    x_R_discrete = discretize_timeseries(x_R, n_bins=n_bins)
    
    x_Psi = get_mean_field_signal(theta, signal_type='sin_Psi')
    x_Psi_discrete = discretize_timeseries(x_Psi, n_bins=n_bins)
    
    # Compute lag sweeps
    df_osc = compute_lag_sweep(x_osc_discrete, max_lag=max_lag)
    df_osc['signal'] = 'oscillator'
    df_osc['K'] = K
    df_osc['mean_R'] = np.mean(R)
    
    df_R = compute_lag_sweep(x_R_discrete, max_lag=max_lag)
    df_R['signal'] = 'order_param_R'
    df_R['K'] = K
    df_R['mean_R'] = np.mean(R)
    
    df_Psi = compute_lag_sweep(x_Psi_discrete, max_lag=max_lag)
    df_Psi['signal'] = 'mean_phase'
    df_Psi['K'] = K
    df_Psi['mean_R'] = np.mean(R)
    
    # Store time series for plotting
    signals = {
        'oscillator': x_osc,
        'order_param_R': R,
        'mean_phase': x_Psi,
        't': t,
    }
    
    return pd.concat([df_osc, df_R, df_Psi], ignore_index=True), signals


def run_coupling_sweep(K_values, **kwargs):
    """Run analysis for multiple coupling strengths."""
    all_results = []
    all_signals = {}
    
    for K in K_values:
        print(f"  K = {K:.2f}...", end=" ")
        df, signals = analyze_single_coupling(K, **kwargs)
        all_results.append(df)
        all_signals[K] = signals
        print(f"mean R = {df['mean_R'].iloc[0]:.3f}")
    
    return pd.concat(all_results, ignore_index=True), all_signals


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_time_series_examples(all_signals, K_values, filename):
    """Plot example time series for different coupling strengths."""
    n_K = len(K_values)
    fig, axes = plt.subplots(n_K, 3, figsize=(14, 3*n_K), sharex=True)
    
    if n_K == 1:
        axes = axes.reshape(1, -1)
    
    for i, K in enumerate(K_values):
        signals = all_signals[K]
        t = signals['t']
        
        # Only plot first 500 points for clarity
        n_plot = min(500, len(t))
        t_plot = t[:n_plot]
        
        axes[i, 0].plot(t_plot, signals['oscillator'][:n_plot], 'b-', linewidth=0.5)
        axes[i, 0].set_ylabel(f'K={K:.1f}')
        if i == 0:
            axes[i, 0].set_title('Single Oscillator sin(θ)')
        
        axes[i, 1].plot(t_plot, signals['order_param_R'][:n_plot], 'r-', linewidth=0.5)
        if i == 0:
            axes[i, 1].set_title('Order Parameter R')
        axes[i, 1].set_ylim(0, 1.05)
        
        axes[i, 2].plot(t_plot, signals['mean_phase'][:n_plot], 'g-', linewidth=0.5)
        if i == 0:
            axes[i, 2].set_title('Mean Phase sin(Ψ)')
    
    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')
    axes[-1, 2].set_xlabel('Time')
    
    plt.suptitle('Kuramoto Oscillators: Time Series at Different Coupling Strengths', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_pid_vs_coupling(df, signal_type, filename):
    """Plot how PID components change with coupling strength."""
    df_signal = df[df['signal'] == signal_type].copy()
    
    # Aggregate: mean across all lag pairs for each K
    df_agg = df_signal.groupby('K').agg({
        'redundancy': ['mean', 'max'],
        'synergy': ['mean', 'max'],
        'unique_0': 'mean',
        'unique_1': 'mean',
        'mean_R': 'first'
    }).reset_index()
    df_agg.columns = ['K', 'red_mean', 'red_max', 'syn_mean', 'syn_max', 
                      'uniq0_mean', 'uniq1_mean', 'mean_R']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: PID components vs K
    ax = axes[0]
    ax.plot(df_agg['K'], df_agg['red_mean'], 'o-', label='Redundancy (mean)', 
            color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df_agg['K'], df_agg['syn_mean'], 's-', label='Synergy (mean)', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax.plot(df_agg['K'], df_agg['uniq0_mean'], '^-', label='Unique (mean)', 
            color='#3498db', linewidth=2, markersize=6)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'{signal_type}: Mean PID vs Coupling')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Max values (peaks across lag pairs)
    ax = axes[1]
    ax.plot(df_agg['K'], df_agg['red_max'], 'o-', label='Redundancy (max)', 
            color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df_agg['K'], df_agg['syn_max'], 's-', label='Synergy (max)', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Information (bits)')
    ax.set_title(f'{signal_type}: Peak PID vs Coupling')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Order parameter vs K
    ax = axes[2]
    ax.plot(df_agg['K'], df_agg['mean_R'], 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Mean Order Parameter R')
    ax.set_title('Synchronization Level')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Partial sync')
    ax.legend()
    
    plt.suptitle(f'Temporal PID vs Coupling: {signal_type}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_lag_heatmaps_by_coupling(df, signal_type, K_values, filename):
    """Plot heatmaps of PID across lags for different coupling strengths."""
    n_K = len(K_values)
    fig, axes = plt.subplots(n_K, 4, figsize=(16, 4*n_K))
    
    if n_K == 1:
        axes = axes.reshape(1, -1)
    
    components = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Redundancy', 'Synergy', 'Unique (lag1)', 'Unique (lag2)']
    cmaps = ['Greens', 'Reds', 'Blues', 'Purples']
    
    df_signal = df[df['signal'] == signal_type]
    max_lag = int(df_signal['lag2'].max())
    
    for i, K in enumerate(K_values):
        df_K = df_signal[df_signal['K'] == K]
        mean_R = df_K['mean_R'].iloc[0]
        
        for j, (comp, title, cmap) in enumerate(zip(components, titles, cmaps)):
            ax = axes[i, j]
            
            # Create matrix
            matrix = np.full((max_lag, max_lag), np.nan)
            for _, row in df_K.iterrows():
                l1, l2 = int(row['lag1']), int(row['lag2'])
                matrix[l1-1, l2-1] = row[comp]
            
            sns.heatmap(matrix, ax=ax, cmap=cmap, annot=False,
                        xticklabels=range(1, max_lag+1),
                        yticklabels=range(1, max_lag+1),
                        cbar_kws={'label': 'bits'})
            
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f'K={K:.1f} (R={mean_R:.2f})\nLag 1')
            else:
                ax.set_ylabel('Lag 1')
            ax.set_xlabel('Lag 2')
    
    plt.suptitle(f'Temporal PID Heatmaps: {signal_type}\nLag structure at different coupling strengths', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_synergy_redundancy_ratio(df, filename):
    """Plot synergy/redundancy ratio vs coupling."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    signal_types = ['oscillator', 'order_param_R', 'mean_phase']
    titles = ['Single Oscillator', 'Order Parameter R', 'Mean Phase']
    colors = ['blue', 'red', 'green']
    
    for ax, signal_type, title, color in zip(axes, signal_types, titles, colors):
        df_signal = df[df['signal'] == signal_type]
        
        # Compute ratio for each lag pair
        df_signal = df_signal.copy()
        df_signal['syn_red_ratio'] = df_signal['synergy'] / (df_signal['redundancy'] + 1e-10)
        
        # Aggregate by K
        df_agg = df_signal.groupby('K').agg({
            'syn_red_ratio': ['mean', 'max'],
            'mean_R': 'first'
        }).reset_index()
        df_agg.columns = ['K', 'ratio_mean', 'ratio_max', 'mean_R']
        
        ax.plot(df_agg['K'], df_agg['ratio_mean'], 'o-', color=color, 
                linewidth=2, markersize=8, label='Mean ratio')
        ax.plot(df_agg['K'], df_agg['ratio_max'], 's--', color=color, 
                linewidth=1, markersize=6, alpha=0.7, label='Max ratio')
        ax.set_xlabel('Coupling Strength K')
        ax.set_ylabel('Synergy / Redundancy')
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.axhline(1, color='gray', linestyle=':', alpha=0.5)
    
    plt.suptitle('Synergy-Redundancy Balance vs Coupling', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def plot_predictions_summary(df, filename):
    """Create summary plot with predictions annotated."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Aggregate data
    df_osc = df[df['signal'] == 'oscillator'].groupby('K').agg({
        'redundancy': 'mean', 'synergy': 'mean', 'mean_R': 'first'
    }).reset_index()
    
    df_R = df[df['signal'] == 'order_param_R'].groupby('K').agg({
        'redundancy': 'mean', 'synergy': 'mean', 'mean_R': 'first'
    }).reset_index()
    
    # Panel 1: Oscillator signal
    ax = axes[0, 0]
    ax.plot(df_osc['K'], df_osc['redundancy'], 'o-', label='Redundancy', 
            color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df_osc['K'], df_osc['synergy'], 's-', label='Synergy', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Mean Information (bits)')
    ax.set_title('Single Oscillator sin(θ)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Order parameter R
    ax = axes[0, 1]
    ax.plot(df_R['K'], df_R['redundancy'], 'o-', label='Redundancy', 
            color='#2ecc71', linewidth=2, markersize=8)
    ax.plot(df_R['K'], df_R['synergy'], 's-', label='Synergy', 
            color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Mean Information (bits)')
    ax.set_title('Order Parameter R')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Synchronization curve
    ax = axes[1, 0]
    ax.plot(df_osc['K'], df_osc['mean_R'], 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Coupling Strength K')
    ax.set_ylabel('Mean Order Parameter R')
    ax.set_title('Synchronization Transition')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    
    # Add critical coupling annotation
    # For Gaussian ω with std=1, Kc ≈ 2*sqrt(2/π) ≈ 1.6
    Kc_approx = 2 * np.sqrt(2/np.pi)
    ax.axvline(Kc_approx, color='red', linestyle='--', alpha=0.7, label=f'Kc ≈ {Kc_approx:.2f}')
    ax.legend()
    
    # Panel 4: Predictions text
    ax = axes[1, 1]
    ax.axis('off')
    
    predictions_text = """
    PREDICTIONS FOR TEMPORAL PID OF KURAMOTO OSCILLATORS
    =====================================================
    
    LOW COUPLING (K << Kc):
    • Oscillators are nearly independent
    • Each oscillator is quasi-periodic at its natural frequency
    • PREDICTION: High REDUNDANCY at period-matching lags
                  (past predicts future via simple periodicity)
    
    NEAR CRITICAL (K ≈ Kc ≈ 1.6):
    • Partial synchronization, intermittent dynamics
    • Complex temporal correlations emerge
    • PREDICTION: Enhanced SYNERGY possible
                  (nonlinear interactions create history-dependence)
    
    HIGH COUPLING (K >> Kc):
    • Full synchronization to collective frequency
    • Signal becomes very periodic again
    • PREDICTION: High REDUNDANCY at synchronized period
    
    ORDER PARAMETER R:
    • Low K: R fluctuates → complex temporal structure
    • High K: R ≈ 1 constant → low information content
    • Near Kc: R shows critical fluctuations → richest dynamics
    """
    ax.text(0.05, 0.95, predictions_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Kuramoto Oscillators: Temporal PID Summary', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete Kuramoto temporal PID analysis."""
    print("="*70)
    print("KURAMOTO OSCILLATORS: TEMPORAL PID ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Parameters
    N = 50  # Number of oscillators
    T = 300  # Simulation time
    dt = 0.01  # Time step
    max_lag = 15  # Maximum lag to analyze
    downsample = 10  # Downsample factor
    n_bins = 4  # Discretization bins
    
    # Coupling strengths to test
    # Critical Kc ≈ 2*sqrt(2/π) ≈ 1.6 for Gaussian frequencies with std=1
    K_values = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    print(f"\nParameters:")
    print(f"  N = {N} oscillators")
    print(f"  T = {T} time units")
    print(f"  dt = {dt}")
    print(f"  max_lag = {max_lag}")
    print(f"  downsample = {downsample}")
    print(f"  n_bins = {n_bins}")
    print(f"  K values = {K_values}")
    print(f"  Expected Kc ≈ {2*np.sqrt(2/np.pi):.2f}")
    
    # Run analysis
    print("\nRunning coupling sweep...")
    df, all_signals = run_coupling_sweep(
        K_values,
        N=N, T=T, dt=dt, max_lag=max_lag,
        downsample=downsample, n_bins=n_bins
    )
    
    # Save results
    print("\nSaving results...")
    df.to_csv(OUTPUT_DIR / "kuramoto_temporal_pid.csv", index=False)
    
    # Create figures
    print("\nGenerating figures...")
    
    # 1. Time series examples
    K_examples = [0.5, 1.5, 3.0, 5.0]
    K_examples = [k for k in K_examples if k in all_signals]
    plot_time_series_examples(all_signals, K_examples, "time_series_examples.png")
    
    # 2. PID vs coupling for each signal type
    plot_pid_vs_coupling(df, 'oscillator', "pid_vs_coupling_oscillator.png")
    plot_pid_vs_coupling(df, 'order_param_R', "pid_vs_coupling_order_param.png")
    plot_pid_vs_coupling(df, 'mean_phase', "pid_vs_coupling_mean_phase.png")
    
    # 3. Lag heatmaps for key coupling values
    K_heatmap = [0.5, 1.5, 3.0, 5.0]
    K_heatmap = [k for k in K_heatmap if k in df['K'].values]
    plot_lag_heatmaps_by_coupling(df, 'oscillator', K_heatmap, "lag_heatmaps_oscillator.png")
    plot_lag_heatmaps_by_coupling(df, 'order_param_R', K_heatmap, "lag_heatmaps_order_param.png")
    
    # 4. Synergy/redundancy ratio
    plot_synergy_redundancy_ratio(df, "synergy_redundancy_ratio.png")
    
    # 5. Summary with predictions
    plot_predictions_summary(df, "predictions_summary.png")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for signal_type in ['oscillator', 'order_param_R', 'mean_phase']:
        print(f"\n{signal_type.upper()}:")
        df_sig = df[df['signal'] == signal_type]
        summary = df_sig.groupby('K').agg({
            'redundancy': 'mean',
            'synergy': 'mean',
            'mean_R': 'first'
        }).round(4)
        print(summary.to_string())
    
    # Files created
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
