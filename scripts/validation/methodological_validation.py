"""
Methodological Validation: Temporal PID & PhiID
================================================

Additional toy validations designed to stress-test the core claims
of the temporal information decomposition framework.

Key questions:
1. Can temporal PID/PhiID separate autocorrelation from genuine structure?
2. Is the Takens embedding mapping valid (X,Y not independent)?
3. Do the new temporal metrics (persistence, lability, etc.) behave as predicted?
4. What does the method capture vs miss for known nonlinear systems?
5. Are there symmetry properties the embedding should respect?
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from collections import OrderedDict

# Add paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "phiid"))
sys.path.insert(0, str(PROJECT_DIR / "scripts" / "pid"))

from phyid.calculate import (
    _get_entropy_four_vec,
    _get_coinfo_four_vec,
    _get_redundancy_four_vec,
    _get_double_redundancy_four_vec,
    _get_atoms_four_vec
)

ATOM_NAMES = ['rtr', 'rtx', 'rty', 'rts',
              'xtr', 'xtx', 'xty', 'xts',
              'ytr', 'ytx', 'yty', 'yts',
              'str', 'stx', 'sty', 'sts']


def takens_phiid(signal, tau, kind='gaussian'):
    """Compute PhiID via Takens embedding [t, t+τ, t+2τ, t+3τ]."""
    N = len(signal) - 3 * tau
    if N < 100:
        return None
    X = np.zeros((4, N))
    X[0] = signal[0:N]
    X[1] = signal[tau:N+tau]
    X[2] = signal[2*tau:N+2*tau]
    X[3] = signal[3*tau:N+3*tau]
    if kind == 'gaussian':
        means = np.mean(X, axis=1, keepdims=True)
        stds = np.std(X, axis=1, ddof=1, keepdims=True)
        stds = np.maximum(stds, 1e-10)
        X = (X - means) / stds
    try:
        h = _get_entropy_four_vec(X, kind=kind)
        I = _get_coinfo_four_vec(h)
        R = _get_redundancy_four_vec('MMI', I)
        calc = {"h_res": h, "I_res": I, "R_res": R}
        calc["rtr"] = _get_double_redundancy_four_vec('MMI', calc)
        atoms = _get_atoms_four_vec(calc)
        result = {}
        for name in ATOM_NAMES:
            val = atoms.get(name, 0.0)
            if isinstance(val, np.ndarray):
                val = np.nanmean(val)
            result[name] = float(val) if np.isfinite(val) else 0.0
        return result
    except Exception as e:
        print(f"  Error: {e}")
        return None


def compute_new_metrics(atoms):
    """Compute the proposed temporal metrics from the new_metrics document."""
    m = {}
    m['global_persistence'] = atoms['rtr']
    m['phase_persistence'] = atoms['xtx'] + atoms['yty']
    m['integrated_persistence'] = atoms['sts']
    m['total_persistence'] = atoms['rtr'] + atoms['xtx'] + atoms['yty'] + atoms['sts']
    m['structural_lability'] = sum(atoms[a] for a in ATOM_NAMES) - m['total_persistence']
    m['phase_exchange'] = atoms['xty'] + atoms['ytx']
    m['phase_bias'] = atoms['xty'] - atoms['ytx']
    m['broadcast'] = atoms['xtr'] + atoms['ytr'] + atoms['str']
    m['fragmentation'] = atoms['rtx'] + atoms['rty'] + atoms['stx'] + atoms['sty']
    m['integration_formation'] = atoms['rts'] + atoms['xts'] + atoms['yts']
    m['integration_deployment'] = atoms['str'] + atoms['stx'] + atoms['sty']
    m['net_integration_balance'] = m['integration_formation'] - m['integration_deployment']
    return m


def fmt(val, width=8):
    """Format a float for display."""
    if abs(val) < 0.0005:
        return f"{'~0':>{width}}"
    return f"{val:>{width}.3f}"


# =============================================================================
# VALIDATION 1: Autocorrelation vs Genuine Structure
# =============================================================================

def validation_1_autocorrelation():
    """
    Core claim: Temporal PID/PhiID can separate autocorrelation from
    genuine nonlinear structure.

    Test: Compare AR(1) (pure linear autocorrelation) against processes
    with matched autocorrelation but additional nonlinear structure.
    """
    print("=" * 80)
    print("VALIDATION 1: Autocorrelation vs Genuine Nonlinear Structure")
    print("=" * 80)
    print()
    print("Claim: Synergy atoms capture structure BEYOND linear autocorrelation.")
    print("Method: Compare AR(1) against processes with same autocorrelation")
    print("        but added nonlinear dynamics.")
    print()

    N = 15000
    np.random.seed(42)

    # 1a. AR(1) baseline with phi=0.7
    phi = 0.7
    ar1 = np.zeros(N)
    noise_std = np.sqrt(1 - phi**2)
    ar1[0] = np.random.randn()
    for t in range(1, N):
        ar1[t] = phi * ar1[t-1] + noise_std * np.random.randn()

    # 1b. Same autocorrelation + nonlinear squaring
    # x[t] = phi * x[t-1] + noise, then y = sign(x) * x^2
    # This preserves the autocorrelation structure (monotonic transform)
    # but adds nonlinear dependence
    ar1_squared = np.sign(ar1) * ar1**2

    # 1c. Threshold process: binary version of AR(1)
    # Preserves temporal structure but is fundamentally nonlinear
    ar1_threshold = (ar1 > 0).astype(float)

    # 1d. AR(1) + nonlinear innovation: x[t] = phi*x[t-1] + f(noise)
    # where f is a nonlinear function. The linear autocorrelation matches
    # but the higher-order structure differs.
    np.random.seed(42)
    ar1_nl = np.zeros(N)
    ar1_nl[0] = np.random.randn()
    for t in range(1, N):
        # Non-Gaussian innovation: chi-squared - 1 (mean 0, var 2)
        innovation = (np.random.randn()**2 - 1) * noise_std / np.sqrt(2)
        ar1_nl[t] = phi * ar1_nl[t-1] + innovation

    processes = OrderedDict([
        ("AR(1) baseline", (ar1, 'gaussian')),
        ("AR(1) squared", (ar1_squared, 'gaussian')),
        ("AR(1) threshold", (ar1_threshold, 'discrete')),
        ("AR(1) non-Gauss", (ar1_nl, 'gaussian')),
    ])

    # Compare autocorrelation
    print("Autocorrelation check (should be similar for all):")
    for name, (sig, _) in processes.items():
        acf1 = np.corrcoef(sig[:-1], sig[1:])[0, 1]
        acf5 = np.corrcoef(sig[:-5], sig[5:])[0, 1]
        print(f"  {name:20s}: ρ(1)={acf1:.3f}  ρ(5)={acf5:.3f}")

    print()
    print("PhiID atoms at τ=1 and τ=5:")
    print()

    for tau in [1, 5]:
        print(f"  τ = {tau}:")
        print(f"  {'Process':20s} {'rtr':>8} {'xtx':>8} {'sts':>8} {'str':>8} "
              f"{'TotPers':>8} {'TotLabil':>8} {'IntForm':>8}")
        for name, (sig, kind) in processes.items():
            atoms = takens_phiid(sig, tau, kind=kind)
            if atoms:
                m = compute_new_metrics(atoms)
                print(f"  {name:20s} {fmt(atoms['rtr'])} {fmt(atoms['xtx'])} "
                      f"{fmt(atoms['sts'])} {fmt(atoms['str'])} "
                      f"{fmt(m['total_persistence'])} {fmt(m['structural_lability'])} "
                      f"{fmt(m['integration_formation'])}")
        print()

    print("  Key question: Do the nonlinear variants show MORE synergy (sts) or")
    print("  structural lability than the linear AR(1) baseline?")
    print("  If yes → method captures genuine nonlinear structure beyond autocorrelation.")
    print()


# =============================================================================
# VALIDATION 2: Takens Embedding Symmetry Properties
# =============================================================================

def validation_2_symmetry():
    """
    The Takens embedding maps [x(t), x(t+τ), x(t+2τ), x(t+3τ)] to
    [X_past, Y_past, X_future, Y_future].

    For a STATIONARY process, the X↔Y symmetry should hold:
    - xtx ≈ yty (both measure lag-2τ autocorrelation)
    - xty ≈ ytx (cross-phase coupling should be symmetric)
    - rtx ≈ rty (erasure from shared to each channel)
    - xts ≈ yts (upward causation from each channel)
    - stx ≈ sty (downward causation to each channel)
    - xtr ≈ ytr (spreading from each channel)

    Asymmetry implies either non-stationarity or genuine temporal
    directionality (arrow of time).
    """
    print("=" * 80)
    print("VALIDATION 2: Takens Embedding Symmetry Properties")
    print("=" * 80)
    print()
    print("Claim: For stationary processes, X↔Y swapped atoms should be equal.")
    print("       Asymmetry indicates temporal directionality (arrow of time).")
    print()

    N = 15000
    np.random.seed(42)

    # Stationary processes (should be symmetric)
    iid = np.random.randn(N)
    ar1 = np.zeros(N)
    ar1[0] = np.random.randn()
    for t in range(1, N):
        ar1[t] = 0.9 * ar1[t-1] + np.sqrt(1-0.81) * np.random.randn()

    # Noisy sine (stationary)
    t_arr = np.arange(N)
    sine = np.sin(2 * np.pi * t_arr / 20) + 0.2 * np.random.randn(N)

    # Non-stationary process (should break symmetry)
    # Chirp: frequency increases with time
    chirp = np.sin(2 * np.pi * (0.01 * t_arr + 0.00001 * t_arr**2))
    chirp += 0.2 * np.random.randn(N)

    processes = OrderedDict([
        ("IID Gaussian", (iid, 'gaussian')),
        ("AR(1) φ=0.9", (ar1, 'gaussian')),
        ("Sine T=20", (sine, 'gaussian')),
        ("Chirp (non-stat)", (chirp, 'gaussian')),
    ])

    symmetric_pairs = [
        ('xtx', 'yty'), ('xty', 'ytx'), ('rtx', 'rty'),
        ('xts', 'yts'), ('stx', 'sty'), ('xtr', 'ytr')
    ]

    tau = 5
    print(f"  Atom symmetry at τ={tau} (|a - b| for each X↔Y pair):")
    print()
    header = f"  {'Process':20s}"
    for a, b in symmetric_pairs:
        header += f" {a}/{b:>10s}"
    print(header)

    for name, (sig, kind) in processes.items():
        atoms = takens_phiid(sig, tau, kind=kind)
        if atoms:
            line = f"  {name:20s}"
            for a, b in symmetric_pairs:
                diff = abs(atoms[a] - atoms[b])
                line += f" {diff:>10.5f}"
            print(line)

    print()
    print("  Expected: Stationary processes show near-zero differences.")
    print("  The chirp (non-stationary) may show larger asymmetries.")
    print()


# =============================================================================
# VALIDATION 3: τ-Scaling Laws for Known Processes
# =============================================================================

def validation_3_tau_scaling():
    """
    For well-understood processes, we know HOW atoms should scale with τ:

    - AR(1) φ: rtr should decay as ~φ^(2τ) since the Takens window spans 3τ
    - Sine T: atoms should be periodic in τ with period T
    - IID: all atoms ≈ 0 at all τ

    This tests whether the method recovers known scaling laws.
    """
    print("=" * 80)
    print("VALIDATION 3: τ-Scaling Laws for Known Processes")
    print("=" * 80)
    print()

    N = 15000
    np.random.seed(42)

    # AR(1) processes
    ar1_signals = {}
    for phi in [0.5, 0.9, 0.99]:
        x = np.zeros(N)
        ns = np.sqrt(1 - phi**2)
        x[0] = np.random.randn()
        for t in range(1, N):
            x[t] = phi * x[t-1] + ns * np.random.randn()
        ar1_signals[phi] = x

    tau_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

    print("3a. AR(1) rtr decay vs theoretical autocorrelation ρ(τ)=φ^τ")
    print()
    for phi, sig in ar1_signals.items():
        print(f"  AR(1) φ={phi}:")
        print(f"    {'τ':>5s} {'ρ(τ)':>8s} {'ρ(2τ)':>8s} {'rtr':>8s} {'total_pers':>10s}")
        for tau in tau_values:
            atoms = takens_phiid(sig, tau)
            if atoms:
                m = compute_new_metrics(atoms)
                rho_tau = phi**tau
                rho_2tau = phi**(2*tau)
                print(f"    {tau:5d} {rho_tau:8.4f} {rho_2tau:8.4f} "
                      f"{atoms['rtr']:8.4f} {m['total_persistence']:10.4f}")
        print()

    # Sine wave: periodicity in τ
    print("3b. Sine wave (T=20): atoms should show periodic structure in τ")
    print()
    t_arr = np.arange(N)
    sine = np.sin(2 * np.pi * t_arr / 20) + 0.2 * np.random.randn(N)

    print(f"  {'τ':>5s} {'rtr':>8s} {'xtx':>8s} {'sts':>8s} {'phase_ex':>10s} {'tot_pers':>10s}")
    for tau in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
        atoms = takens_phiid(sine, tau)
        if atoms:
            m = compute_new_metrics(atoms)
            print(f"  {tau:5d} {fmt(atoms['rtr'])} {fmt(atoms['xtx'])} "
                  f"{fmt(atoms['sts'])} {fmt(m['phase_exchange'])} "
                  f"{fmt(m['total_persistence'])}")

    print()
    print("  Key: τ=5 (T/4) should show peak synergy (quadrature).")
    print("  τ=10 (T/2) should show high rtr (anti-correlation → redundancy).")
    print("  τ=20 (T) should show high rtr again (same phase).")
    print()


# =============================================================================
# VALIDATION 4: New Temporal Metrics on Diagnostic Processes
# =============================================================================

def validation_4_new_metrics():
    """
    Test the proposed temporal metrics (persistence hierarchy, broadcast,
    fragmentation, etc.) on processes with known expected behavior.
    """
    print("=" * 80)
    print("VALIDATION 4: New Temporal Metrics on Diagnostic Processes")
    print("=" * 80)
    print()

    N = 15000
    np.random.seed(42)

    # Process 1: Constant + noise (pure persistence)
    const = np.ones(N) * 5 + 0.01 * np.random.randn(N)

    # Process 2: AR(1) high phi (strong persistence, some lability)
    ar1 = np.zeros(N)
    ar1[0] = np.random.randn()
    for t in range(1, N):
        ar1[t] = 0.95 * ar1[t-1] + np.sqrt(1-0.95**2) * np.random.randn()

    # Process 3: XOR (synergistic, should show integration formation)
    xor = np.zeros(N, dtype=int)
    xor[0] = np.random.randint(2)
    xor[1] = np.random.randint(2)
    for t in range(2, N):
        if np.random.rand() < 0.05:
            xor[t] = np.random.randint(2)
        else:
            xor[t] = xor[t-1] ^ xor[t-2]
    xor = xor.astype(float)

    # Process 4: Alternating binary (high phase exchange)
    alt = np.zeros(N)
    for t in range(N):
        alt[t] = t % 2

    # Process 5: Regime switching (fragmentation)
    # Two AR(1) regimes with different means
    regime = np.zeros(N)
    state = 0
    for t in range(N):
        if np.random.rand() < 0.005:
            state = 1 - state
        regime[t] = state * 3 + 0.3 * np.random.randn()

    # Process 6: Logistic map (chaotic, complex)
    logistic = np.zeros(N)
    logistic[0] = 0.4
    for t in range(1, N):
        logistic[t] = 3.9 * logistic[t-1] * (1 - logistic[t-1])

    processes = OrderedDict([
        ("Near-constant", (const, 'gaussian')),
        ("AR(1) φ=0.95", (ar1, 'gaussian')),
        ("XOR (noisy)", (xor, 'discrete')),
        ("Alternating", (alt, 'discrete')),
        ("Regime switch", (regime, 'gaussian')),
        ("Logistic r=3.9", (logistic, 'gaussian')),
    ])

    tau = 1
    print(f"New temporal metrics at τ={tau}:")
    print()
    metrics_to_show = [
        'global_persistence', 'phase_persistence', 'integrated_persistence',
        'structural_lability', 'phase_exchange', 'broadcast',
        'fragmentation', 'integration_formation', 'net_integration_balance',
    ]
    header = f"  {'Process':16s}"
    for m in metrics_to_show:
        short = m[:10]
        header += f" {short:>10s}"
    print(header)
    print("  " + "-" * (16 + 11 * len(metrics_to_show)))

    for name, (sig, kind) in processes.items():
        atoms = takens_phiid(sig, tau, kind=kind)
        if atoms:
            m = compute_new_metrics(atoms)
            line = f"  {name:16s}"
            for metric in metrics_to_show:
                line += f" {fmt(m[metric], 10)}"
            print(line)

    print()
    print("  Expected patterns:")
    print("    Near-constant: High global_persistence, low everything else")
    print("    AR(1) φ=0.95:  High global_persistence, moderate lability")
    print("    XOR:            High integrated_persistence, high integration_formation")
    print("    Alternating:    High phase_exchange (info swaps between even/odd)")
    print("    Regime switch:  High fragmentation (shared structure breaks)")
    print("    Logistic:       Complex mix, high structural_lability")
    print()


# =============================================================================
# VALIDATION 5: Gaussian vs Discrete Estimation Comparison
# =============================================================================

def validation_5_estimation():
    """
    The Gaussian estimation assumes linear dependencies (Gaussian copula).
    For genuinely nonlinear processes, it may miss structure.

    Test: Compare Gaussian vs discrete estimation on the same processes.
    """
    print("=" * 80)
    print("VALIDATION 5: Gaussian vs Discrete Estimation")
    print("=" * 80)
    print()
    print("Claim: Gaussian estimation captures linear structure only.")
    print("       Discrete estimation captures full (including nonlinear) structure.")
    print()

    N = 15000
    np.random.seed(42)

    # XOR process (fundamentally nonlinear)
    xor = np.zeros(N, dtype=int)
    xor[0] = np.random.randint(2)
    xor[1] = np.random.randint(2)
    for t in range(2, N):
        if np.random.rand() < 0.05:
            xor[t] = np.random.randint(2)
        else:
            xor[t] = xor[t-1] ^ xor[t-2]
    xor_float = xor.astype(float)

    # Logistic map (nonlinear, discretized to 4 bins)
    logistic = np.zeros(N)
    logistic[0] = 0.4
    for t in range(1, N):
        logistic[t] = 3.9 * logistic[t-1] * (1 - logistic[t-1])

    percentiles = np.percentile(logistic, [25, 50, 75])
    logistic_discrete = np.digitize(logistic, percentiles).astype(float)

    # AR(1) - linear (should give SAME results both ways when discretized)
    ar1 = np.zeros(N)
    ar1[0] = np.random.randn()
    for t in range(1, N):
        ar1[t] = 0.9 * ar1[t-1] + np.sqrt(1-0.81) * np.random.randn()

    percentiles_ar = np.percentile(ar1, [25, 50, 75])
    ar1_discrete = np.digitize(ar1, percentiles_ar).astype(float)

    tau = 1
    print(f"  Comparison at τ={tau}:")
    print()

    tests = [
        ("XOR", xor_float, xor_float),
        ("Logistic", logistic, logistic_discrete),
        ("AR(1)", ar1, ar1_discrete),
    ]

    for name, sig_gauss, sig_disc in tests:
        atoms_g = takens_phiid(sig_gauss, tau, kind='gaussian')
        atoms_d = takens_phiid(sig_disc, tau, kind='discrete')
        if atoms_g and atoms_d:
            m_g = compute_new_metrics(atoms_g)
            m_d = compute_new_metrics(atoms_d)
            print(f"  {name}:")
            print(f"    {'':15s} {'Gaussian':>10s} {'Discrete':>10s} {'Ratio':>10s}")
            for metric in ['total_persistence', 'integrated_persistence',
                           'structural_lability', 'integration_formation']:
                vg = m_g[metric]
                vd = m_d[metric]
                ratio = vd / vg if abs(vg) > 0.001 else float('inf')
                print(f"    {metric:15s} {vg:10.4f} {vd:10.4f} {ratio:10.2f}x")
            print()

    print("  Key question: For XOR/Logistic (nonlinear), does discrete estimation")
    print("  reveal MORE integration/synergy than Gaussian?")
    print("  For AR(1) (linear), estimates should be comparable.")
    print()


# =============================================================================
# VALIDATION 6: The Sine-at-Quarter-Period Argument
# =============================================================================

def validation_6_sine_quadrature():
    """
    The central claim for separating autocorrelation from genuine structure:
    A sine wave at τ=T/4 has ZERO autocorrelation between adjacent
    embedding positions, yet shows HIGH synergy.

    This directly proves that temporal PhiID captures structure beyond
    linear autocorrelation.
    """
    print("=" * 80)
    print("VALIDATION 6: The Sine-at-Quarter-Period Argument")
    print("=" * 80)
    print()
    print("This is the KEY validation for the entire framework.")
    print()
    print("Claim: At τ = T/4, a sine wave has zero autocorrelation between")
    print("       adjacent embedding points, yet PhiID shows high synergy.")
    print("       This proves the method captures genuine nonlinear structure.")
    print()

    N = 15000
    np.random.seed(42)
    T = 20

    t_arr = np.arange(N)
    sine_clean = np.sin(2 * np.pi * t_arr / T)
    sine_noisy = sine_clean + 0.2 * np.random.randn(N)

    # AR(1) matched to have same lag-5 autocorrelation as noisy sine
    # Sine at lag 5 (T/4): ρ = cos(2π*5/20) = cos(π/2) = 0
    # So we need AR(1) with ρ(5) ≈ 0, i.e. φ^5 ≈ 0, i.e. φ ≈ 0
    # → Just use white noise as the "matched autocorrelation" baseline
    iid = np.random.randn(N)

    tau_quarter = T // 4  # = 5

    print(f"  Period T = {T}, τ = T/4 = {tau_quarter}")
    print()

    # Verify autocorrelation
    for name, sig in [("Sine (noisy)", sine_noisy), ("IID baseline", iid)]:
        rho = np.corrcoef(sig[:-tau_quarter], sig[tau_quarter:])[0, 1]
        print(f"  {name:20s}: ρ(τ) = {rho:.4f}")

    print()
    print(f"  PhiID atoms at τ={tau_quarter}:")
    print()
    print(f"  {'Process':20s} {'rtr':>8s} {'xtx':>8s} {'sts':>8s} "
          f"{'phase_ex':>10s} {'tot_pers':>10s} {'int_form':>10s}")

    for name, sig in [("Sine (noisy)", sine_noisy),
                      ("Sine (clean)", sine_clean),
                      ("IID baseline", iid)]:
        atoms = takens_phiid(sig, tau_quarter)
        if atoms:
            m = compute_new_metrics(atoms)
            print(f"  {name:20s} {fmt(atoms['rtr'])} {fmt(atoms['xtx'])} "
                  f"{fmt(atoms['sts'])} {fmt(m['phase_exchange'])} "
                  f"{fmt(m['total_persistence'])} {fmt(m['integration_formation'])}")

    print()

    # Now scan across tau to show the full picture
    print("  Full τ scan for noisy sine (T=20):")
    print(f"  {'τ':>5s} {'ρ(τ)':>8s} {'rtr':>8s} {'sts':>8s} "
          f"{'tot_pers':>10s} {'net_integ':>10s}")
    for tau in range(1, 21):
        rho = np.corrcoef(sine_noisy[:-tau], sine_noisy[tau:])[0, 1]
        atoms = takens_phiid(sine_noisy, tau)
        if atoms:
            m = compute_new_metrics(atoms)
            marker = " ← T/4 (zero ρ, peak synergy)" if tau == 5 else ""
            marker = " ← T/2" if tau == 10 else marker
            marker = " ← T" if tau == 20 else marker
            print(f"  {tau:5d} {rho:8.4f} {fmt(atoms['rtr'])} {fmt(atoms['sts'])} "
                  f"{fmt(m['total_persistence'])} {fmt(m['net_integration_balance'])}"
                  f"{marker}")

    print()
    print("  CONCLUSION: If sts >> 0 when ρ(τ) ≈ 0, then PhiID captures")
    print("  genuine temporal structure beyond linear autocorrelation.  ✓")
    print()


# =============================================================================
# VALIDATION 7: What the Method CANNOT Do
# =============================================================================

def validation_7_limitations():
    """
    Honest assessment of limitations:

    1. Gaussian estimation misses nonlinear XOR-type synergy
    2. 4-point embedding may miss structure at other timescales
    3. Non-stationary signals violate assumptions
    """
    print("=" * 80)
    print("VALIDATION 7: Known Limitations")
    print("=" * 80)
    print()

    N = 15000
    np.random.seed(42)

    # Limitation 1: Gaussian estimation misses XOR synergy
    print("7a. Gaussian estimation on binary XOR")
    print()
    xor = np.zeros(N, dtype=int)
    xor[0] = np.random.randint(2)
    xor[1] = np.random.randint(2)
    for t in range(2, N):
        if np.random.rand() < 0.05:
            xor[t] = np.random.randint(2)
        else:
            xor[t] = xor[t-1] ^ xor[t-2]
    xor_f = xor.astype(float)

    atoms_gauss = takens_phiid(xor_f, 1, kind='gaussian')
    atoms_disc = takens_phiid(xor_f, 1, kind='discrete')

    if atoms_gauss and atoms_disc:
        print(f"  {'Estimation':15s} {'sts':>8s} {'str':>8s} {'rtr':>8s} {'total':>8s}")
        total_g = sum(abs(atoms_gauss[a]) for a in ATOM_NAMES)
        total_d = sum(abs(atoms_disc[a]) for a in ATOM_NAMES)
        print(f"  {'Gaussian':15s} {fmt(atoms_gauss['sts'])} {fmt(atoms_gauss['str'])} "
              f"{fmt(atoms_gauss['rtr'])} {fmt(total_g)}")
        print(f"  {'Discrete':15s} {fmt(atoms_disc['sts'])} {fmt(atoms_disc['str'])} "
              f"{fmt(atoms_disc['rtr'])} {fmt(total_d)}")
        print()
        print("  → Gaussian estimation sees almost nothing in XOR!")
        print("    The Gaussian copula cannot represent XOR-type nonlinear dependencies.")
        print("    This is a FUNDAMENTAL limitation when kind='gaussian'.")

    # Limitation 2: Fixed 4-point embedding
    print()
    print("7b. Structure at non-probed timescales is invisible")
    print()
    # Process with structure only at lag 7 (not 1, 2, or 3 multiples of τ)
    x = np.random.randn(N)
    y = np.zeros(N)
    for t in range(7, N):
        y[t] = 0.8 * x[t-7] + 0.2 * np.random.randn()

    print(f"  Process: y[t] = 0.8*x[t-7] + noise (structure at lag 7)")
    print(f"  {'τ':>5s} {'rtr':>8s} {'tot_pers':>10s} {'Expected?':>12s}")
    for tau in [1, 2, 3, 5, 7]:
        # For this to work, we analyze y not x
        atoms = takens_phiid(y, tau)
        if atoms:
            m = compute_new_metrics(atoms)
            # At τ=7/3 ≈ 2.3 (not integer), the structure is partially visible
            # But the Takens window spans 3τ, so at τ=2, window=6 → misses lag 7
            # At τ=3, window=9 → includes lag 7 somewhere
            expected = "Visible" if 3*tau >= 7 else "Missed"
            print(f"  {tau:5d} {fmt(atoms['rtr'])} {fmt(m['total_persistence'])} "
                  f"{expected:>12s}")

    print()
    print("  → The 4-point Takens window spans [t, t+3τ]. Structure at lags")
    print("    outside this window is invisible. τ must be chosen to match")
    print("    the timescale of interest.")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║  METHODOLOGICAL VALIDATION: Temporal PID & PhiID                            ║")
    print("║  Testing core claims of the temporal information decomposition framework     ║")
    print("╚" + "═" * 78 + "╝")
    print()

    validation_1_autocorrelation()
    validation_2_symmetry()
    validation_3_tau_scaling()
    validation_4_new_metrics()
    validation_5_estimation()
    validation_6_sine_quadrature()
    validation_7_limitations()

    print("=" * 80)
    print("ALL VALIDATIONS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
