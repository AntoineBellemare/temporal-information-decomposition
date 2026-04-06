"""
EEG Sleep Temporal PID Analysis
================================

Downloads one subject from the Bitbrain Open Access Sleep Dataset (OpenNeuro ds005555),
then computes PID across minute-scale lags to reveal temporal information structure
during sleep.

Approach:
- Discretize the raw EEG signal (quantile binning)
- For each pair of lags (1 to MAX_LAG_MIN minutes, 1-minute steps):
    Build joint distribution P(x[t-lag1], x[t-lag2], x[t]) from all valid triplets
    Compute PID (MMI) -> redundancy, synergy, unique atoms
- Track how PID evolves over the recording using sliding 1-minute windows
- Overlay sleep stage annotations to link PID structure to sleep architecture

Usage:
    python eeg_sleep.py

Requirements:
    pip install mne awscli
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from collections import Counter
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon
from scipy.signal import lfilter, butter, filtfilt, iirnotch
from itertools import combinations
import hashlib

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("MNE-Python is required. Install with: pip install mne")
    sys.exit(1)

import dit
from dit.pid import PID_MMI
from dit import Distribution

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "ds005555"
RESULTS_DIR = PROJECT_DIR / "results" / "pid" / "eeg_sleep"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "sub-1"
CHANNEL = "PSG_C3"          # Central EEG, best for sleep staging
DURATION_HOURS = 7           # 6h captures multiple NREM-REM cycles
WINDOW_SEC = 15             # 30-second windows
MAX_LAG_MIN = 10             # Max lag in minutes
N_BINS = 4                   # Discretization bins
DISCRETIZE_PER_WINDOW = True # Per-window binning avoids amplitude confound
CONTINUOUS_STAGE_FILTER = False  # If True, require ALL windows between source2
                                 # and target to be same stage (stricter).
                                 # If False, only check the 3 triplet windows.

# Derived constants for minute-to-window conversion
WINDOWS_PER_MIN = 60 // WINDOW_SEC  # windows per minute (2 for 30s)
MIN_PER_WINDOW = WINDOW_SEC / 60    # minutes per window (0.5 for 30s)
MAX_LAG_WINDOWS = MAX_LAG_MIN * WINDOWS_PER_MIN  # total lag steps at window resolution

STAGE_COLORS = {
    'Wake': '#E8A317',
    'N1':   '#87CEEB',
    'N2':   '#4169E1',
    'N3':   '#191970',
    'REM':  '#DC143C',
    '?':    '#D3D3D3',
}
STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']


def config_hash():
    """Short hash of analysis parameters for cache invalidation."""
    params = (f"{WINDOW_SEC}_{MAX_LAG_MIN}_{N_BINS}_{DISCRETIZE_PER_WINDOW}"
              f"_{CONTINUOUS_STAGE_FILTER}_{DURATION_HOURS}")
    return hashlib.md5(params.encode()).hexdigest()[:8]


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR correction. Returns array of adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    adjusted = np.empty(n)
    adjusted[order[-1]] = pvals[order[-1]]
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(pvals[order[i]] * n / (i + 1),
                                  adjusted[order[i + 1]])
    return np.clip(adjusted, 0, 1)


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_subject(subject="sub-1"):
    """Download one subject from OpenNeuro ds005555 via AWS CLI."""
    subject_dir = DATA_DIR / subject
    if subject_dir.exists() and any(subject_dir.rglob("*.edf")):
        print(f"  Data already present: {subject_dir}")
        return subject_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {subject} from OpenNeuro ds005555 ...")
    cmd = (
        f'aws s3 sync --no-sign-request '
        f's3://openneuro.org/ds005555/{subject} "{subject_dir}"'
    )
    subprocess.run(cmd, check=True, shell=True)
    print(f"  Downloaded to {subject_dir}")
    return subject_dir


def preprocess_eeg(signal, fs, l_freq=0.5, h_freq=60.0):
    """Bandpass filter (0.5-60 Hz) and remove 50/60 Hz line noise."""
    nyq = fs / 2.0
    if h_freq >= nyq:
        h_freq = nyq - 1.0
    b, a = butter(4, [l_freq / nyq, h_freq / nyq], btype='band')
    signal = filtfilt(b, a, signal.astype(np.float64))
    for f0 in [50.0, 60.0]:
        if f0 < nyq:
            b_n, a_n = iirnotch(f0, Q=30, fs=fs)
            signal = filtfilt(b_n, a_n, signal)
    return signal


# =============================================================================
# DATA LOADING
# =============================================================================

def find_channel(raw, preferred="PSG_C3"):
    """Find best matching channel name in the EDF."""
    ch_names = raw.ch_names

    # exact match
    if preferred in ch_names:
        return preferred

    # partial matches
    short = preferred.replace("PSG_", "")
    for ch in ch_names:
        if preferred in ch or short in ch:
            return ch

    # fallback: any EEG-like channel
    for pattern in ["C3", "C4", "F3", "F4", "O1"]:
        for ch in ch_names:
            if pattern in ch:
                return ch

    return ch_names[0]


def load_eeg_and_stages(subject_dir, channel="PSG_C3", duration_hours=2):
    """
    Load PSG EEG channel and sleep-stage annotations.

    Returns
    -------
    signal : 1-D ndarray
    fs : float
    stages : list[dict]   (onset, duration, stage)
    ch_name : str
    """
    # --- EDF -----------------------------------------------------------------
    edf_files = list(subject_dir.rglob("*psg_eeg.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No PSG EDF in {subject_dir}")

    edf_file = edf_files[0]
    print(f"  EDF file : {edf_file.name}")

    raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
    fs = raw.info['sfreq']
    print(f"  Fs       : {fs} Hz")
    print(f"  Channels : {raw.ch_names}")

    ch_name = find_channel(raw, channel)
    print(f"  Using    : {ch_name}")

    # Crop & pick before loading to save memory
    tmax = min(duration_hours * 3600, raw.times[-1])
    raw.crop(tmax=tmax)
    raw.pick([ch_name])
    raw.load_data(verbose=False)
    signal = raw.get_data()[0]
    print(f"  Samples  : {len(signal)}  ({len(signal)/fs/3600:.2f} h)")

    # --- Sleep stages --------------------------------------------------------
    events_files = list(subject_dir.rglob("*psg_events.tsv"))
    stages = []
    if events_files:
        events_file = events_files[0]
        print(f"  Events   : {events_file.name}")
        ev = pd.read_csv(events_file, sep='\t')

        # locate stage column
        stage_col = None
        for col in ['stage_hum', 'stage_ai', 'value', 'trial_type']:
            if col in ev.columns:
                stage_col = col
                break

        if stage_col and 'onset' in ev.columns:
            for _, row in ev.iterrows():
                try:
                    stage_val = int(float(row[stage_col]))
                except (ValueError, TypeError):
                    continue
                if stage_val not in (0, 1, 2, 3, 4):
                    continue
                dur = row.get('duration', 30)
                stages.append({
                    'onset': float(row['onset']),
                    'duration': float(dur) if pd.notna(dur) else 30.0,
                    'stage': stage_val,
                })
    if stages:
        print(f"  Stages   : {len(stages)} annotations loaded")
    else:
        print("  Stages   : none found (plots will lack stage overlay)")

    return signal, fs, stages, ch_name


def get_stage_per_window(stages, n_windows, window_sec):
    """Return the dominant sleep stage label for each 1-min window."""
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    labels = []
    for w in range(n_windows):
        t_mid = (w + 0.5) * window_sec
        label = '?'
        for s in stages:
            if s['onset'] <= t_mid < s['onset'] + s['duration']:
                label = stage_names.get(s['stage'], '?')
                break
        labels.append(label)
    return labels


# =============================================================================
# DISCRETISATION
# =============================================================================

def discretize_signal(signal, n_bins=4):
    """Quantile-based discretisation of a continuous signal (global bins)."""
    x = np.asarray(signal, dtype=float)
    valid = x[~np.isnan(x)]
    edges = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(np.nanmin(x) - 1e-10, np.nanmax(x) + 1e-10, n_bins + 1)
    else:
        edges[0] -= 1e-10
        edges[-1] += 1e-10
    return np.digitize(x, edges[1:-1]).astype(np.int8)


def discretize_signal_per_window(signal, fs, window_sec, n_bins=4):
    """
    Discretise each 1-minute window independently using its OWN quantiles.

    This removes amplitude confounds: N3 (large slow waves) and N1 (small
    amplitude) get the same bin distribution within each window, so PID
    measures only temporal *structure*, not amplitude level.
    """
    W = int(window_sec * fs)
    n_windows = len(signal) // W
    out = np.zeros(n_windows * W, dtype=np.int8)

    for i in range(n_windows):
        seg = signal[i * W : (i + 1) * W]
        seg_valid = seg[~np.isnan(seg)]
        if len(seg_valid) < n_bins:
            out[i * W : (i + 1) * W] = 0
            continue
        edges = np.percentile(seg_valid, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.linspace(np.nanmin(seg) - 1e-10,
                                np.nanmax(seg) + 1e-10, n_bins + 1)
        else:
            edges[0] -= 1e-10
            edges[-1] += 1e-10
        out[i * W : (i + 1) * W] = np.digitize(seg, edges[1:-1]).astype(np.int8)

    return out[:n_windows * W]


def detect_bad_windows(signal, fs, window_sec, flat_threshold=1e-10):
    """
    Detect windows with artifacts, disconnections, or flat-line segments.

    Returns a boolean array: True = window is good, False = skip it.
    """
    W = int(window_sec * fs)
    n_windows = len(signal) // W
    good = np.ones(n_windows, dtype=bool)

    for i in range(n_windows):
        seg = signal[i * W : (i + 1) * W]
        # flat line (disconnection)
        if np.std(seg) < flat_threshold:
            good[i] = False
        # NaN contamination
        elif np.any(np.isnan(seg)):
            good[i] = False

    n_bad = np.sum(~good)
    if n_bad > 0:
        bad_idx = np.where(~good)[0]
        print(f"  Detected {n_bad} bad windows: {bad_idx.tolist()}")
    return good


# =============================================================================
# PID COMPUTATION
# =============================================================================

def compute_pid_from_arrays(src1, src2, target, n_bins=4):
    """
    Compute PID-MMI from three aligned integer arrays.

    Uses np.bincount for fast counting (much faster than Counter on strings).
    """
    codes = src1.astype(int) * n_bins * n_bins + src2.astype(int) * n_bins + target.astype(int)
    counts = np.bincount(codes, minlength=n_bins ** 3)
    total = counts.sum()
    if total == 0:
        return dict(redundancy=np.nan, unique_0=np.nan, unique_1=np.nan, synergy=np.nan)

    outcomes, probs = [], []
    for code in range(n_bins ** 3):
        if counts[code] > 0:
            s1 = code // (n_bins * n_bins)
            s2 = (code // n_bins) % n_bins
            t  = code % n_bins
            outcomes.append(f"{s1}{s2}{t}")
            probs.append(counts[code] / total)

    try:
        dist = Distribution(outcomes, probs)
        pid = PID_MMI(dist)
    except Exception:
        return dict(redundancy=np.nan, unique_0=np.nan, unique_1=np.nan, synergy=np.nan)

    summary = dict(redundancy=0.0, unique_0=0.0, unique_1=0.0, synergy=0.0)
    for node in pid._lattice:
        try:
            val = float(pid.get_pi(node))
        except Exception:
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


# ---- global matrix ----------------------------------------------------------

def compute_global_pid_matrix(signal_disc, fs, max_lag_min, window_sec):
    """
    PID for every lag pair using ALL valid sample triplets in the recording.

    Returns one PID per (lag1, lag2) pair.
    """
    W = int(window_sec * fs)
    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60
    n_pairs = max_lag_w * (max_lag_w - 1) // 2
    results, done = [], 0

    for lag1 in range(1, max_lag_w):
        for lag2 in range(lag1 + 1, max_lag_w + 1):
            off1 = lag1 * W
            off2 = lag2 * W
            n_valid = len(signal_disc) - off2

            target = signal_disc[off2 : off2 + n_valid]
            src1   = signal_disc[off2 - off1 : off2 - off1 + n_valid]
            src2   = signal_disc[0 : n_valid]

            pid = compute_pid_from_arrays(src1, src2, target, N_BINS)
            pid['lag1_min'] = round(lag1 * min_per_w, 4)
            pid['lag2_min'] = round(lag2 * min_per_w, 4)
            pid['n_triplets'] = n_valid
            results.append(pid)

            done += 1
            if done % 50 == 0 or done == n_pairs:
                print(f"    {done}/{n_pairs} lag pairs done")

    return pd.DataFrame(results)


# ---- time-resolved ----------------------------------------------------------

def compute_timeresolved_pid(signal_disc, fs, max_lag_min, window_sec,
                             good_windows=None):
    """
    PID in sliding 1-minute windows for every lag pair.

    For window *t* (index = minute):
        target  = samples in [t·W, (t+1)·W)
        source1 = samples in [(t−lag1)·W, (t−lag1+1)·W)
        source2 = samples in [(t−lag2)·W, (t−lag2+1)·W)

    Skips windows where any of the three involved windows is marked bad.
    """
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60
    first = max_lag_w  # first valid window index
    n_pairs = max_lag_w * (max_lag_w - 1) // 2

    # determine which target windows are usable
    valid_targets = []
    for t in range(first, n_windows):
        if good_windows is not None:
            # check that target + all possible source windows are good
            needed = [t] + [t - lag_w for lag_w in range(1, max_lag_w + 1)]
            if not all(good_windows[w] for w in needed if w < len(good_windows)):
                continue
        valid_targets.append(t)

    n_valid = len(valid_targets)
    print(f"    Windows          : {n_valid}  (of {n_windows - first} possible)")
    print(f"    Lag pairs/window : {n_pairs}")
    print(f"    Total PID calls  : {n_valid * n_pairs}")

    results = []
    t0 = time.time()

    for idx, t in enumerate(valid_targets):
        tgt = signal_disc[t * W : (t + 1) * W]

        for lag1 in range(1, max_lag_w):
            for lag2 in range(lag1 + 1, max_lag_w + 1):
                w1 = t - lag1
                w2 = t - lag2
                s1 = signal_disc[w1 * W : (w1 + 1) * W]
                s2 = signal_disc[w2 * W : (w2 + 1) * W]

                pid = compute_pid_from_arrays(s1, s2, tgt, N_BINS)
                pid['window']   = t
                pid['time_min'] = round(t * min_per_w, 4)
                pid['lag1_min'] = round(lag1 * min_per_w, 4)
                pid['lag2_min'] = round(lag2 * min_per_w, 4)
                results.append(pid)

        if (idx + 1) % 5 == 0 or idx == n_valid - 1:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta  = rate * (n_valid - idx - 1)
            print(f"    window {idx+1}/{n_valid}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    return pd.DataFrame(results)


# =============================================================================
# AR(1) BASELINE + AUTOCORRELATION
# =============================================================================

def fit_ar1_per_window(signal_disc, fs, window_sec, n_bins):
    """
    Fit AR(1) model per window:  x[t] = φ·x[t-1] + ε

    Returns φ per window and the per-window bin distributions.
    """
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    phi = np.zeros(n_windows)
    for i in range(n_windows):
        seg = signal_disc[i * W : (i + 1) * W].astype(float)
        if np.std(seg) < 1e-10:
            phi[i] = 0.0
            continue
        seg_z = seg - seg.mean()
        denom = np.dot(seg_z[:-1], seg_z[:-1])
        if denom > 0:
            phi[i] = np.dot(seg_z[1:], seg_z[:-1]) / denom
        else:
            phi[i] = 0.0
    return phi


def compute_ar1_global_pid(signal_disc, fs, max_lag_min, window_sec, n_bins,
                           n_realisations=5, seed=42):
    """
    Window-level AR(1) baseline for the global PID matrix.

    Fits AR(1) at the window timescale so the baseline captures realistic
    minute-scale autocorrelation (unlike sample-level AR(1) which decays
    to zero at minute lags).  Each sample position within a window evolves
    as an independent AR(1) process across windows.

    Returns
    -------
    df_mean : DataFrame  — mean AR(1) PID across realisations
    df_std  : DataFrame  — std across realisations
    """
    rng = np.random.default_rng(seed)
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W

    # Reshape to (n_windows, W) for window-level analysis
    data = signal_disc[:n_windows * W].reshape(n_windows, W).astype(float)

    # Estimate window-level AR(1) coefficient from a subsample of positions
    n_pos = min(W, 500)
    pos_idx = np.linspace(0, W - 1, n_pos, dtype=int)
    phis = []
    for j in pos_idx:
        col = data[:, j]
        col_z = col - col.mean()
        denom = np.dot(col_z[:-1], col_z[:-1])
        if denom > 0:
            phis.append(np.dot(col_z[1:], col_z[:-1]) / denom)
    phi_w = np.mean(phis) if phis else 0.0
    sigma_w = np.std(data) * np.sqrt(max(1 - phi_w ** 2, 0.01))

    print(f"    Window-level AR(1): φ_w = {phi_w:.4f}, σ_w = {sigma_w:.4f}")

    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60
    lag_pairs = [(l1, l2) for l1 in range(1, max_lag_w)
                 for l2 in range(l1 + 1, max_lag_w + 1)]
    n_pairs = len(lag_pairs)

    all_vals = np.zeros((n_realisations, n_pairs, 4))

    for rep in range(n_realisations):
        # Generate AR(1) at window level for all positions simultaneously
        noise = rng.normal(0, max(sigma_w, 1e-6), (n_windows, W))
        noise[0, :] = rng.normal(data.mean(), np.std(data), W)
        ar_cont = lfilter([1], [1, -phi_w], noise, axis=0)
        ar_cont += data.mean(axis=0, keepdims=True)

        # Discretize per window (matching real pipeline)
        ar_disc = np.zeros((n_windows, W), dtype=np.int8)
        for i in range(n_windows):
            seg = ar_cont[i]
            edges = np.percentile(seg, np.linspace(0, 100, n_bins + 1))
            edges = np.unique(edges)
            if len(edges) < 3:
                edges = np.linspace(seg.min() - 1e-10, seg.max() + 1e-10,
                                    n_bins + 1)
            else:
                edges[0] -= 1e-10
                edges[-1] += 1e-10
            ar_disc[i] = np.digitize(seg, edges[1:-1]).astype(np.int8)

        ar_flat = ar_disc.reshape(-1)
        for lag_idx, (lag1, lag2) in enumerate(lag_pairs):
            off1 = lag1 * W
            off2 = lag2 * W
            n_valid = len(ar_flat) - off2

            target = ar_flat[off2: off2 + n_valid]
            src1   = ar_flat[off2 - off1: off2 - off1 + n_valid]
            src2   = ar_flat[0: n_valid]

            pid = compute_pid_from_arrays(src1, src2, target, n_bins)
            all_vals[rep, lag_idx] = [pid['redundancy'], pid['synergy'],
                                      pid['unique_0'], pid['unique_1']]

        print(f"    Window AR(1) realisation {rep+1}/{n_realisations}")

    means = all_vals.mean(axis=0)
    stds  = all_vals.std(axis=0)

    records_mean, records_std = [], []
    for i, (l1, l2) in enumerate(lag_pairs):
        records_mean.append(dict(lag1_min=round(l1 * min_per_w, 4),
                                 lag2_min=round(l2 * min_per_w, 4),
                                 redundancy=means[i, 0], synergy=means[i, 1],
                                 unique_0=means[i, 2], unique_1=means[i, 3]))
        records_std.append(dict(lag1_min=round(l1 * min_per_w, 4),
                                lag2_min=round(l2 * min_per_w, 4),
                                redundancy=stds[i, 0], synergy=stds[i, 1],
                                unique_0=stds[i, 2], unique_1=stds[i, 3]))

    return pd.DataFrame(records_mean), pd.DataFrame(records_std)


def compute_ar1_timeresolved_pid(signal_disc, fs, max_lag_min, window_sec,
                                 n_bins, good_windows=None,
                                 ar1_global_mean=None,
                                 n_realisations=5, seed=42):
    """
    AR(1) baseline per window — by broadcasting the global AR(1) mean.

    An AR(1) process is stationary, so the expected PID per lag pair is constant
    across time. We reuse the global AR(1) mean (computed from full-length
    AR(1) realisations) and replicate it for every valid window.

    The excess (actual − AR(1)) then isolates per-window nonlinear temporal
    structure beyond what a linear Gaussian process would produce.
    """
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60
    first = max_lag_w

    valid_targets = []
    for t in range(first, n_windows):
        if good_windows is not None:
            needed = [t] + [t - lag_w for lag_w in range(1, max_lag_w + 1)]
            if not all(good_windows[w] for w in needed if w < len(good_windows)):
                continue
        valid_targets.append(t)

    print(f"    Broadcasting global AR(1) to {len(valid_targets)} windows")

    records = []
    for t in valid_targets:
        for _, row in ar1_global_mean.iterrows():
            records.append(dict(
                window=t, time_min=round(t * min_per_w, 4),
                lag1_min=row['lag1_min'],
                lag2_min=row['lag2_min'],
                redundancy=row['redundancy'],
                synergy=row['synergy'],
                unique_0=row['unique_0'],
                unique_1=row['unique_1'],
            ))

    return pd.DataFrame(records)


def compute_autocorrelation_per_window(signal, fs, max_lag_min, window_sec):
    """
    Compute autocorrelation R(τ) per window for τ = 1, 2, ..., max_lag_windows.

    Uses the continuous (pre-discretization) signal to get the true linear
    autocorrelation.

    Returns DataFrame with columns: window, time_min, lag_min, autocorr.
    """
    W = int(window_sec * fs)
    n_windows = len(signal) // W
    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60

    records = []
    for w in range(n_windows):
        seg = signal[w * W : (w + 1) * W]
        seg_z = seg - np.mean(seg)
        var = np.dot(seg_z, seg_z)
        t_min = round(w * min_per_w, 4)
        if var < 1e-20:
            for lag_w in range(1, max_lag_w + 1):
                records.append(dict(window=w, time_min=t_min,
                                    lag_min=round(lag_w * min_per_w, 4), autocorr=0.0))
            continue
        for lag_w in range(1, max_lag_w + 1):
            w2 = w - lag_w
            if w2 < 0:
                records.append(dict(window=w, time_min=t_min,
                                    lag_min=round(lag_w * min_per_w, 4), autocorr=np.nan))
                continue
            seg2 = signal[w2 * W : (w2 + 1) * W]
            seg2_z = seg2 - np.mean(seg2)
            var2 = np.dot(seg2_z, seg2_z)
            if var2 < 1e-20:
                records.append(dict(window=w, time_min=t_min,
                                    lag_min=round(lag_w * min_per_w, 4), autocorr=0.0))
                continue
            r = np.dot(seg_z, seg2_z) / np.sqrt(var * var2)
            records.append(dict(window=w, time_min=t_min,
                                lag_min=round(lag_w * min_per_w, 4), autocorr=r))

    return pd.DataFrame(records)


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def add_stage_background(ax, stage_labels):
    """Shade axis background by sleep stage colour."""
    for i, stage in enumerate(stage_labels):
        x0 = i * MIN_PER_WINDOW
        x1 = (i + 1) * MIN_PER_WINDOW
        ax.axvspan(x0, x1, alpha=0.15, color=STAGE_COLORS.get(stage, '#D3D3D3'),
                   linewidth=0)


def stage_legend_handles():
    """Return legend handles for sleep stage colours."""
    return [Patch(facecolor=STAGE_COLORS[s], alpha=0.5, label=s) for s in STAGE_ORDER]


def common_lag_range(tr):
    """
    Find the set of (lag1_min, lag2_min) pairs that have data in ALL stages
    present in tr. Filter tr down to only those pairs so that per-window
    averages are comparable across stages.
    """
    present = [s for s in STAGE_ORDER if s in tr['stage'].unique()]
    if len(present) < 2:
        return tr
    # lag pairs with data in each stage
    sets = []
    for s in present:
        sub = tr[tr['stage'] == s]
        pairs = set(zip(sub['lag1_min'], sub['lag2_min']))
        sets.append(pairs)
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    if len(common) == 0:
        return tr
    common_df = pd.DataFrame(list(common), columns=['lag1_min', 'lag2_min'])
    return tr.merge(common_df, on=['lag1_min', 'lag2_min'], how='inner')


def lag_index(df):
    """Return sorted unique lag values and a value→index mapping."""
    all_lags = sorted(set(df['lag1_min'].unique()) | set(df['lag2_min'].unique()))
    return all_lags, {v: i for i, v in enumerate(all_lags)}


def lag_tick_labels(lag_vals):
    """Compact tick labels: drop trailing zeros for whole numbers."""
    return [f'{v:g}' for v in lag_vals]


# =============================================================================
# PLOT 1 — GLOBAL PID MATRIX
# =============================================================================

def plot_global_matrix(results, save_path=None):
    """2×2 heatmap of PID atoms as lag1 × lag2 matrices."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lag_vals, l_idx = lag_index(results)
    n_lags = len(lag_vals)
    tl = lag_tick_labels(lag_vals)

    comps  = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Redundancy', 'Synergy', 'Unique (lag 1)', 'Unique (lag 2)']
    cmaps  = ['Greens', 'Reds', 'Blues', 'Purples']

    for ax, comp, title, cmap in zip(axes.flat, comps, titles, cmaps):
        mat = np.full((n_lags, n_lags), np.nan)
        for _, r in results.iterrows():
            mat[l_idx[r['lag1_min']], l_idx[r['lag2_min']]] = r[comp]

        sns.heatmap(mat, ax=ax, cmap=cmap, mask=np.isnan(mat),
                    xticklabels=tl, yticklabels=tl,
                    cbar_kws={'label': 'bits'})
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Lag 2 (min)')
        ax.set_ylabel('Lag 1 (min)')

    plt.suptitle(f'Global Temporal PID — minute-scale lags\n'
                 f'({CHANNEL}, {DURATION_HOURS}h, {N_BINS} bins)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 2 — COMBINED: HYPNOGRAM + PID TIME SERIES + AR(1) + PERCENTILE BANDS
# =============================================================================

def plot_combined_timeseries(tr, tr_ar1, stage_labels, save_path=None):
    """
    Unified time-series figure:
      - Hypnogram on top
      - For each metric (redundancy, synergy, total unique):
        actual mean (solid), AR(1) baseline (grey fill),
        excess (dashed), and percentile bands (10-90, 25-75)
        across lag pairs.
    """
    # --- actual: mean + percentile bands per window --------------------------
    windows = sorted(tr['time_min'].unique())
    w_arr = np.array(windows)

    agg_act = tr.groupby('time_min').agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()

    # compute unique_total per row in tr for percentile calculation
    tr_u = tr.copy()
    tr_u['unique_total'] = tr_u['unique_0'] + tr_u['unique_1']

    # percentile bands
    pctiles = {}
    for col in ['redundancy', 'synergy', 'unique_total']:
        p10, p25, p75, p90 = [], [], [], []
        for ww in windows:
            vals = tr_u.loc[tr_u['time_min'] == ww, col].dropna()
            p10.append(np.percentile(vals, 10))
            p25.append(np.percentile(vals, 25))
            p75.append(np.percentile(vals, 75))
            p90.append(np.percentile(vals, 90))
        pctiles[col] = (np.array(p10), np.array(p25),
                        np.array(p75), np.array(p90))

    # --- AR(1) mean per window -----------------------------------------------
    agg_ar1 = tr_ar1.groupby('time_min').agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()

    mrg = agg_act.merge(agg_ar1, on='time_min', suffixes=('', '_ar1'))
    t = mrg['time_min'].values

    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [0.7, 1, 1, 1]},
                             sharex=True)

    # ---- hypnogram ----------------------------------------------------------
    stage_num = {'Wake': 0, 'REM': -0.5, 'N1': -1, 'N2': -2, 'N3': -3, '?': 0.5}
    hyp_y = [stage_num.get(s, 0.5) for s in stage_labels]
    hyp_x = np.arange(len(stage_labels)) * MIN_PER_WINDOW
    axes[0].step(hyp_x, hyp_y, where='post', color='black', lw=1.5)
    axes[0].set_yticks([0, -0.5, -1, -2, -3])
    axes[0].set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
    axes[0].set_ylabel('Stage')
    axes[0].set_title('Hypnogram', fontweight='bold')
    add_stage_background(axes[0], stage_labels)
    axes[0].legend(handles=stage_legend_handles(), loc='upper right',
                   ncol=5, fontsize=8, framealpha=0.8)
    axes[0].grid(axis='x', alpha=0.3)

    # ---- metric panels ------------------------------------------------------
    panels = [
        ('redundancy', 'green', 'Redundancy'),
        ('synergy',    'red',   'Synergy'),
    ]

    for ax, (col, color, label) in zip(axes[1:3], panels):
        actual_v = mrg[col].values
        ar1_v    = mrg[f'{col}_ar1'].values
        p10, p25, p75, p90 = pctiles[col]

        # percentile bands (widest first)
        ax.fill_between(w_arr, p10, p90, alpha=0.08, color=color,
                        label='10–90 %ile')
        ax.fill_between(w_arr, p25, p75, alpha=0.18, color=color,
                        label='25–75 %ile')
        # AR(1) baseline
        ax.fill_between(t, ar1_v, alpha=0.3, color='gray',
                        label='AR(1) baseline')
        # actual mean
        ax.plot(t, actual_v, color=color, lw=1.5, label='Mean', zorder=3)
        # excess
        ax.plot(t, actual_v - ar1_v, color=color, lw=1, ls='--',
                alpha=0.6, label='Excess', zorder=2)
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        ax.set_ylabel(f'{label}\n(bits)')
        ax.legend(loc='upper right', fontsize=7, ncol=2)
        add_stage_background(ax, stage_labels)
        ax.grid(axis='x', alpha=0.3)

    # unique total
    act_uniq  = mrg['unique_0'].values + mrg['unique_1'].values
    ar1_uniq  = mrg['unique_0_ar1'].values + mrg['unique_1_ar1'].values
    p10, p25, p75, p90 = pctiles['unique_total']

    axes[3].fill_between(w_arr, p10, p90, alpha=0.08, color='blue',
                         label='10–90 %ile')
    axes[3].fill_between(w_arr, p25, p75, alpha=0.18, color='blue',
                         label='25–75 %ile')
    axes[3].fill_between(t, ar1_uniq, alpha=0.3, color='gray',
                         label='AR(1) baseline')
    axes[3].plot(t, act_uniq, color='blue', lw=1.5, label='Mean', zorder=3)
    axes[3].plot(t, act_uniq - ar1_uniq, color='blue', lw=1, ls='--',
                 alpha=0.6, label='Excess', zorder=2)
    axes[3].axhline(0, color='black', lw=0.5, alpha=0.3)
    axes[3].set_ylabel('Total Unique\n(bits)')
    axes[3].legend(loc='upper right', fontsize=7, ncol=2)
    add_stage_background(axes[3], stage_labels)
    axes[3].grid(axis='x', alpha=0.3)

    axes[-1].set_xlabel('Time (min)')
    plt.suptitle(f'Temporal PID evolution during sleep\n'
                 f'({CHANNEL}, mean + spread across lag pairs, AR(1) baseline)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 3 — TIME × LAG HEATMAPS
# =============================================================================

def plot_time_lag_heatmaps(tr, stage_labels, save_path=None):
    """
    Heatmap: x = time (min), y = lag2 (with lag1 fixed at 1 min), colour = metric.
    Shows how synergy/redundancy at each timescale evolves over the night.
    """
    subset = tr[tr['lag1_min'] == tr['lag1_min'].min()].copy()
    lag1_val = tr['lag1_min'].min()
    if subset.empty:
        print("    WARNING: no data with smallest lag1, skipping time-lag heatmaps")
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    metrics = [
        ('redundancy', 'Greens',  'Redundancy'),
        ('synergy',    'Reds',    'Synergy'),
        ('unique_0',   'Blues',   'Unique (lag 1)'),
    ]

    for ax, (col, cmap, label) in zip(axes, metrics):
        piv = subset.pivot_table(index='lag2_min', columns='time_min', values=col)
        sns.heatmap(piv, ax=ax, cmap=cmap, cbar_kws={'label': 'bits'},
                    xticklabels=10, yticklabels=1)
        ax.set_ylabel('Lag 2 (min)')
        ax.set_title(label, fontweight='bold')
        ax.invert_yaxis()

    axes[-1].set_xlabel('Time (min)')

    # stage colour bar on top
    stage_ax = fig.add_axes([0.125, 0.94, 0.775, 0.015])  # [left, bottom, w, h]
    valid_times = sorted(subset['time_min'].unique())
    stage_arr = np.array([[STAGE_ORDER.index(stage_labels[int(round(t / MIN_PER_WINDOW))])
                           if stage_labels[int(round(t / MIN_PER_WINDOW))] in STAGE_ORDER else -1
                           for t in valid_times]])
    from matplotlib.colors import ListedColormap
    stage_cmap = ListedColormap([STAGE_COLORS[s] for s in STAGE_ORDER])
    stage_ax.imshow(stage_arr, aspect='auto', cmap=stage_cmap,
                    vmin=0, vmax=len(STAGE_ORDER) - 1)
    stage_ax.set_yticks([])
    stage_ax.set_xticks([])
    stage_ax.set_title('Sleep stage', fontsize=9)

    plt.suptitle(f'Time × Lag heatmap (lag1 fixed at {lag1_val:g} min)\n'
                 f'({CHANNEL}, {N_BINS} bins)',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 4 — STAGE-STRATIFIED COMPARISON
# =============================================================================

def plot_stage_comparison(tr, save_path=None):
    """Boxplots of each PID atom grouped by sleep stage, with stats."""
    # restrict to common lag range so means are comparable across stages
    tr_common = common_lag_range(tr)
    n_common = len(tr_common[['lag1_min', 'lag2_min']].drop_duplicates())
    # aggregate per window
    agg = tr_common.groupby(['time_min', 'stage']).agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()
    agg['unique_total'] = agg['unique_0'] + agg['unique_1']
    agg['ratio'] = agg['synergy'] / agg['redundancy'].replace(0, np.nan)
    agg = agg[agg['stage'].isin(STAGE_ORDER)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    palette = {s: STAGE_COLORS[s] for s in STAGE_ORDER}

    metrics = [
        ('redundancy', 'Redundancy (bits)'),
        ('synergy',    'Synergy (bits)'),
        ('unique_total', 'Total Unique (bits)'),
        ('ratio',      'Synergy / Redundancy'),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        sns.boxplot(data=agg, x='stage', y=col, order=STAGE_ORDER,
                    palette=palette, ax=ax, showfliers=False, width=0.6)
        ax.set_xlabel('Sleep stage')
        ax.set_ylabel(ylabel)
        ax.set_title(col.replace('_', ' ').title() if col != 'ratio' else 'S / R Ratio',
                     fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if col == 'ratio':
            ax.axhline(1.0, color='grey', ls='--', alpha=0.6, label='S = R')
            ax.legend(fontsize=8)

        # Kruskal-Wallis across stages
        groups = [agg.loc[agg['stage'] == s, col].dropna().values
                  for s in STAGE_ORDER if len(agg.loc[agg['stage'] == s, col].dropna()) > 0]
        if len(groups) >= 2:
            H, p_kw = kruskal(*groups)
            sig_str = '***' if p_kw < 0.001 else '**' if p_kw < 0.01 else '*' if p_kw < 0.05 else 'n.s.'
            ax.set_title(f"{ax.get_title()}\nKW: H={H:.1f}, p={p_kw:.1e} {sig_str}",
                         fontweight='bold', fontsize=10)

        # pairwise Mann-Whitney with FDR correction
        present = [s for s in STAGE_ORDER
                   if len(agg.loc[agg['stage'] == s, col].dropna()) >= 3]
        all_pairs = []
        all_pvals = []
        for s1, s2 in combinations(present, 2):
            v1 = agg.loc[agg['stage'] == s1, col].dropna().values
            v2 = agg.loc[agg['stage'] == s2, col].dropna().values
            if len(v1) >= 3 and len(v2) >= 3:
                _, p_mw = mannwhitneyu(v1, v2, alternative='two-sided')
                all_pairs.append((s1, s2, p_mw))
                all_pvals.append(p_mw)

        # apply BH-FDR and keep only significant pairs
        sig_pairs = []
        if all_pvals:
            adj_pvals = benjamini_hochberg(all_pvals)
            for (s1, s2, _), p_adj in zip(all_pairs, adj_pvals):
                if p_adj < 0.05:
                    sig_pairs.append((s1, s2, p_adj))

        # draw significance brackets (top 4 most significant)
        sig_pairs.sort(key=lambda x: x[2])
        y_max = agg[col].dropna().quantile(0.95) if len(agg[col].dropna()) > 0 else 1
        y_step = y_max * 0.08
        for rank, (s1, s2, p_val) in enumerate(sig_pairs[:4]):
            x1 = STAGE_ORDER.index(s1)
            x2 = STAGE_ORDER.index(s2)
            y_bar = y_max + y_step * (rank + 1)
            ax.plot([x1, x1, x2, x2], [y_bar - y_step*0.2, y_bar, y_bar, y_bar - y_step*0.2],
                    color='black', lw=0.8)
            stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
            ax.text((x1 + x2) / 2, y_bar, stars, ha='center', va='bottom', fontsize=8)

    plt.suptitle(f'PID by sleep stage (per-window means, {n_common} common lag pairs)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 5 — TIME × LAG-DIFF HEATMAP (averaged over absolute lags)
# =============================================================================

def plot_lagdiff_heatmap(tr, stage_labels, save_path=None):
    """
    Heatmap: x = time, y = lag_diff (lag2 − lag1), colour = metric.
    Each cell is the mean metric across all lag pairs with that difference.
    """
    tr = tr.copy()
    tr['lag_diff'] = tr['lag2_min'] - tr['lag1_min']

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    metrics = [
        ('redundancy', 'Greens', 'Redundancy'),
        ('synergy',    'Reds',   'Synergy'),
        ('unique_0',   'Blues',  'Unique (lag 1)'),
    ]

    for ax, (col, cmap, label) in zip(axes, metrics):
        piv = tr.pivot_table(index='lag_diff', columns='time_min',
                             values=col, aggfunc='mean')
        sns.heatmap(piv, ax=ax, cmap=cmap, cbar_kws={'label': 'bits'},
                    xticklabels=10, yticklabels=1)
        ax.set_ylabel('Lag diff (min)')
        ax.set_title(label, fontweight='bold')
        ax.invert_yaxis()

    axes[-1].set_xlabel('Time (min)')

    plt.suptitle('Time × Lag-difference heatmap\n'
                 f'({CHANNEL}, mean over lag pairs with same difference)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 6 — GLOBAL PID MATRIX: ACTUAL vs AR(1) vs EXCESS
# =============================================================================

def plot_global_matrix_vs_ar1(actual, ar1_mean, save_path=None):
    """3-row × 4-col grid: actual / AR(1) baseline / excess for each PID atom."""
    merged = actual.merge(ar1_mean, on=['lag1_min', 'lag2_min'],
                          suffixes=('', '_ar1'), how='inner')
    ar1_df = merged[['lag1_min', 'lag2_min'] +
                    [f'{m}_ar1' for m in ['redundancy', 'synergy', 'unique_0', 'unique_1']]].copy()
    ar1_df.columns = ['lag1_min', 'lag2_min', 'redundancy', 'synergy', 'unique_0', 'unique_1']
    actual_df = merged[['lag1_min', 'lag2_min', 'redundancy', 'synergy', 'unique_0', 'unique_1']].copy()
    excess = actual_df.copy()
    for m in ['redundancy', 'synergy', 'unique_0', 'unique_1']:
        excess[m] = actual_df[m].values - ar1_df[m].values

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))

    lag_vals, l_idx = lag_index(actual_df)
    n_lags = len(lag_vals)
    tl = lag_tick_labels(lag_vals)

    comps  = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    labels = ['Redundancy', 'Synergy', 'Unique₁', 'Unique₂']
    cmaps  = ['Greens', 'Reds', 'Blues', 'Purples']

    row_dfs    = [actual_df, ar1_df, excess]
    row_labels = ['Actual', 'AR(1) baseline', 'Excess (Actual − AR(1))']

    for ri, (df_r, row_lab) in enumerate(zip(row_dfs, row_labels)):
        for ci, (comp, lab, cmap) in enumerate(zip(comps, labels, cmaps)):
            ax = axes[ri, ci]
            mat = np.full((n_lags, n_lags), np.nan)
            for _, r in df_r.iterrows():
                i1 = l_idx.get(r['lag1_min']); i2 = l_idx.get(r['lag2_min'])
                if i1 is not None and i2 is not None:
                    mat[i1, i2] = r[comp]

            if ri < 2:
                sns.heatmap(mat, ax=ax, cmap=cmap, mask=np.isnan(mat),
                            xticklabels=tl, yticklabels=tl,
                            cbar_kws={'label': 'bits'})
            else:
                vmax = np.nanmax(np.abs(mat))
                if vmax == 0:
                    vmax = 1e-6
                sns.heatmap(mat, ax=ax, cmap='RdBu_r', mask=np.isnan(mat),
                            center=0, vmin=-vmax, vmax=vmax,
                            xticklabels=tl, yticklabels=tl,
                            cbar_kws={'label': 'bits'})

            if ri == 0:
                ax.set_title(lab, fontweight='bold')
            ax.set_xlabel('Lag 2 (min)' if ri == 2 else '')
            ax.set_ylabel(f'{row_lab}\nLag 1 (min)' if ci == 0 else '')

    plt.suptitle(f'Global PID: Actual vs AR(1) Baseline\n'
                 f'({CHANNEL}, {N_BINS} bins)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 7 — STAGE COMPARISON: ACTUAL vs AR(1) (PAIRED)
# =============================================================================

def plot_stage_comparison_vs_ar1(tr_actual, tr_ar1, save_path=None):
    """
    Side-by-side boxplots per stage: actual (filled) vs AR(1) (hatched).
    Plus excess panel.
    """
    # restrict to common lag range
    tr_actual_c = common_lag_range(tr_actual)
    tr_ar1_c = common_lag_range(tr_ar1)

    def _agg(df):
        a = df.groupby(['time_min', 'stage']).agg(
            redundancy=('redundancy', 'mean'),
            synergy=('synergy', 'mean'),
            unique_0=('unique_0', 'mean'),
            unique_1=('unique_1', 'mean'),
        ).reset_index()
        a['unique_total'] = a['unique_0'] + a['unique_1']
        return a[a['stage'].isin(STAGE_ORDER)]

    agg_act = _agg(tr_actual_c)
    agg_ar1 = _agg(tr_ar1_c)

    # merge to compute excess per window
    mrg = agg_act.merge(agg_ar1, on=['time_min', 'stage'], suffixes=('', '_ar1'))
    for m in ['redundancy', 'synergy', 'unique_total']:
        mrg[f'{m}_excess'] = mrg[m] - mrg.get(f'{m}_ar1', 0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    palette = {s: STAGE_COLORS[s] for s in STAGE_ORDER}

    metrics = ['redundancy', 'synergy', 'unique_total']
    labels  = ['Redundancy', 'Synergy', 'Total Unique']

    # top row: actual vs AR(1)
    for ax, m, lab in zip(axes[0], metrics, labels):
        a_plot = agg_act[['stage', m]].copy()
        a_plot['source'] = 'Actual'
        s_plot = agg_ar1[['stage']].copy()
        s_plot[m] = agg_ar1[m].values if m != 'unique_total' else \
            agg_ar1['unique_0'].values + agg_ar1['unique_1'].values
        s_plot['source'] = 'AR(1)'
        combined = pd.concat([a_plot, s_plot], ignore_index=True)

        sns.boxplot(data=combined, x='stage', y=m, hue='source',
                    order=STAGE_ORDER,
                    palette={'Actual': 'steelblue', 'AR(1)': 'lightgray'},
                    ax=ax, showfliers=False, width=0.7)
        ax.set_title(lab, fontweight='bold')
        ax.set_ylabel(f'{lab} (bits)')
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
        if ax != axes[0, 0]:
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=8)

    # top-row stats: Wilcoxon signed-rank actual vs AR(1) per stage
    for ax, m, lab in zip(axes[0], metrics, labels):
        y_top = ax.get_ylim()[1]
        for i, s in enumerate(STAGE_ORDER):
            a_vals = agg_act.loc[agg_act['stage'] == s, m].dropna().values
            b_vals = agg_ar1.loc[agg_ar1['stage'] == s, m].dropna().values
            if m == 'unique_total':
                b_vals = (agg_ar1.loc[agg_ar1['stage'] == s, 'unique_0'].values
                          + agg_ar1.loc[agg_ar1['stage'] == s, 'unique_1'].values)
            n_pair = min(len(a_vals), len(b_vals))
            if n_pair > 10:
                try:
                    _, p_w = wilcoxon(a_vals[:n_pair], b_vals[:n_pair])
                    star = '***' if p_w < 0.001 else '**' if p_w < 0.01 else '*' if p_w < 0.05 else 'n.s.'
                    ax.text(i, y_top * 0.97, star, ha='center', fontsize=8,
                            fontweight='bold', color='black')
                except Exception:
                    pass

    # bottom row: excess by stage
    for ax, m, lab in zip(axes[1], metrics, labels):
        col = f'{m}_excess'
        sns.boxplot(data=mrg, x='stage', y=col, order=STAGE_ORDER,
                    palette=palette, ax=ax, showfliers=False)
        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
        ax.set_ylabel(f'Excess (bits)')
        ax.set_xlabel('Sleep stage')
        ax.grid(axis='y', alpha=0.3)

        # KW on excess across stages
        groups = [mrg.loc[mrg['stage'] == s, col].dropna().values
                  for s in STAGE_ORDER if len(mrg.loc[mrg['stage'] == s, col].dropna()) > 0]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            ax.set_title(f'Excess {lab}\nKW: H={H:.1f}, p={p:.1e}', fontweight='bold')
        else:
            ax.set_title(f'Excess {lab} (nonlinear)', fontweight='bold')

    plt.suptitle(f'PID by sleep stage: Actual vs AR(1) Baseline\n'
                 f'(Excess = nonlinear temporal structure beyond linear prediction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 8 — AUTOCORRELATION: TIME × LAG HEATMAP + PID OVERLAY
# =============================================================================

def plot_autocorrelation_vs_pid(autocorr_df, tr, stage_labels, save_path=None):
    """
    Top: autocorrelation heatmap (time × lag).
    Middle: PID synergy heatmap at lag1=1 (time × lag2) — same axes.
    Bottom: scatter of autocorrelation vs PID per window, coloured by stage.
    Shows whether PID changes track autocorrelation or differ from it.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                             gridspec_kw={'height_ratios': [1, 1, 1.2]})

    # --- autocorrelation heatmap ---
    ac_piv = autocorr_df.pivot_table(index='lag_min', columns='time_min',
                                     values='autocorr')
    sns.heatmap(ac_piv, ax=axes[0], cmap='RdBu_r', center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'R(τ)'},
                xticklabels=10, yticklabels=1)
    axes[0].set_ylabel('Lag (min)')
    axes[0].set_title('Cross-window autocorrelation R(τ)', fontweight='bold')
    axes[0].invert_yaxis()

    # --- PID synergy heatmap (smallest lag1) ---
    subset = tr[tr['lag1_min'] == tr['lag1_min'].min()].copy()
    lag1_val = tr['lag1_min'].min()
    if not subset.empty:
        piv_syn = subset.pivot_table(index='lag2_min', columns='time_min',
                                     values='synergy')
        sns.heatmap(piv_syn, ax=axes[1], cmap='Reds',
                    cbar_kws={'label': 'bits'},
                    xticklabels=10, yticklabels=1)
        axes[1].set_ylabel('Lag₂ (min)')
        axes[1].set_title(f'Synergy (lag₁ = {lag1_val:g} min)', fontweight='bold')
        axes[1].invert_yaxis()

    # stage colour bar — aligned to top heatmap via inset_axes
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    valid_times = sorted(autocorr_df['time_min'].unique())
    stage_arr = np.array([[STAGE_ORDER.index(stage_labels[int(round(t / MIN_PER_WINDOW))])
                           if int(round(t / MIN_PER_WINDOW)) < len(stage_labels) and stage_labels[int(round(t / MIN_PER_WINDOW))] in STAGE_ORDER
                           else -1
                           for t in valid_times]])
    stage_cmap = ListedColormap([STAGE_COLORS[s] for s in STAGE_ORDER])
    stage_ax = inset_axes(axes[0], width='100%', height='8%',
                          loc='upper center',
                          bbox_to_anchor=(0, 1.12, 1, 1),
                          bbox_transform=axes[0].transAxes, borderpad=0)
    stage_ax.imshow(stage_arr, aspect='auto', cmap=stage_cmap,
                    vmin=0, vmax=len(STAGE_ORDER) - 1)
    stage_ax.set_yticks([])
    stage_ax.set_xticks([])
    stage_ax.set_title('Sleep stage', fontsize=9)

    # --- scatter: mean autocorr vs mean synergy per window ---
    ac_mean = autocorr_df.groupby('time_min')['autocorr'].mean().reset_index()
    ac_mean.columns = ['time_min', 'mean_autocorr']
    pid_mean = tr.groupby('time_min').agg(
        synergy=('synergy', 'mean'),
        redundancy=('redundancy', 'mean'),
    ).reset_index()
    sc = ac_mean.merge(pid_mean, on='time_min')
    sc['stage'] = sc['time_min'].map(
        lambda w: stage_labels[int(round(w / MIN_PER_WINDOW))]
                  if int(round(w / MIN_PER_WINDOW)) < len(stage_labels) else '?')
    sc = sc[sc['stage'].isin(STAGE_ORDER)]

    for s in STAGE_ORDER:
        sub = sc[sc['stage'] == s]
        axes[2].scatter(sub['mean_autocorr'], sub['synergy'],
                        c=STAGE_COLORS[s], label=s, alpha=0.6, s=20, edgecolors='none')

    # overall Spearman correlation
    valid_sc = sc.dropna(subset=['mean_autocorr', 'synergy'])
    if len(valid_sc) > 5:
        rho, p_sp = spearmanr(valid_sc['mean_autocorr'], valid_sc['synergy'])
        axes[2].set_title(f'Autocorrelation vs Synergy\nSpearman ρ={rho:.3f}, p={p_sp:.1e}',
                          fontweight='bold')
    else:
        axes[2].set_title('Autocorrelation vs Synergy per window', fontweight='bold')
    axes[2].set_xlabel('Mean autocorrelation R(τ)')
    axes[2].set_ylabel('Mean synergy (bits)')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    plt.suptitle(f'Autocorrelation structure vs PID\n({CHANNEL})',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 9 — S/R RATIO vs AUTOCORRELATION PER STAGE
# =============================================================================

def plot_sr_ratio_vs_autocorr(autocorr_df, tr, save_path=None):
    """
    Key diagnostic: if S/R ratio varies across stages even when autocorrelation
    is similar, the PID structure is genuinely different — not just a confound.

    Left: S/R ratio vs autocorrelation, coloured by stage, with regression lines.
    Right: boxplot of S/R ratio per stage (same as plot 4 but here for context).
    """
    ac_mean = autocorr_df.groupby('time_min')['autocorr'].mean().reset_index()
    ac_mean.columns = ['time_min', 'mean_autocorr']

    pid_agg = tr.groupby(['time_min', 'stage']).agg(
        synergy=('synergy', 'mean'),
        redundancy=('redundancy', 'mean'),
    ).reset_index()
    pid_agg['sr_ratio'] = pid_agg['synergy'] / pid_agg['redundancy'].replace(0, np.nan)

    sc = pid_agg.merge(ac_mean, on='time_min')
    sc = sc[sc['stage'].isin(STAGE_ORDER)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- scatter: autocorr vs S/R ---
    legend_entries = []
    for s in STAGE_ORDER:
        sub = sc[sc['stage'] == s]
        axes[0].scatter(sub['mean_autocorr'], sub['sr_ratio'],
                        c=STAGE_COLORS[s], alpha=0.5, s=20, edgecolors='none')
        # regression line + Spearman
        valid = sub.dropna(subset=['mean_autocorr', 'sr_ratio'])
        if len(valid) > 5:
            z = np.polyfit(valid['mean_autocorr'], valid['sr_ratio'], 1)
            x_line = np.linspace(valid['mean_autocorr'].min(),
                                 valid['mean_autocorr'].max(), 50)
            axes[0].plot(x_line, np.polyval(z, x_line), color=STAGE_COLORS[s],
                         lw=2, alpha=0.7)
            rho, p_sp = spearmanr(valid['mean_autocorr'], valid['sr_ratio'])
            legend_entries.append(f'{s}: ρ={rho:.2f}, p={p_sp:.1e}')
        else:
            legend_entries.append(s)

    axes[0].axhline(1.0, color='grey', ls='--', alpha=0.5)
    # custom legend with Spearman values
    for i, s in enumerate(STAGE_ORDER):
        if i < len(legend_entries):
            axes[0].plot([], [], color=STAGE_COLORS[s], lw=2,
                         label=legend_entries[i])
    axes[0].plot([], [], color='grey', ls='--', label='S = R')
    axes[0].set_xlabel('Mean autocorrelation R(τ)')
    axes[0].set_ylabel('Synergy / Redundancy')
    axes[0].set_title('S/R ratio vs Autocorrelation', fontweight='bold')
    axes[0].legend(fontsize=7, ncol=1)
    axes[0].grid(alpha=0.3)

    # --- boxplot: S/R per stage ---
    sns.boxplot(data=sc, x='stage', y='sr_ratio', order=STAGE_ORDER,
                palette=STAGE_COLORS, ax=axes[1], showfliers=False)
    axes[1].axhline(1.0, color='grey', ls='--', alpha=0.5, label='S = R')
    axes[1].set_xlabel('Sleep stage')
    axes[1].set_ylabel('Synergy / Redundancy')
    # KW test on S/R across stages
    sr_groups = [sc.loc[sc['stage'] == s, 'sr_ratio'].dropna().values
                 for s in STAGE_ORDER if len(sc.loc[sc['stage'] == s, 'sr_ratio'].dropna()) > 0]
    if len(sr_groups) >= 2:
        H, p = kruskal(*sr_groups)
        axes[1].set_title(f'S/R ratio by stage\nKW: H={H:.1f}, p={p:.1e}', fontweight='bold')
    else:
        axes[1].set_title('S/R ratio by stage', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)

    # --- boxplot: autocorrelation per stage ---
    sns.boxplot(data=sc, x='stage', y='mean_autocorr', order=STAGE_ORDER,
                palette=STAGE_COLORS, ax=axes[2], showfliers=False)
    axes[2].set_xlabel('Sleep stage')
    axes[2].set_ylabel('Mean autocorrelation R(τ)')
    axes[2].set_title('Autocorrelation by stage', fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle('Diagnostic: Is S/R ratio explained by autocorrelation?\n'
                 '(Different S/R at similar autocorrelation → genuine nonlinear structure)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 10 — PID ATOM PROFILES PER STAGE (lag1 × lag2 matrix per stage)
# =============================================================================

def plot_pid_per_stage_matrix(tr, save_path=None):
    """
    One lag1×lag2 synergy heatmap per sleep stage.
    Shows whether the timescale structure of synergy differs across stages.
    """
    present = [s for s in STAGE_ORDER if s in tr['stage'].unique()]
    n_stages = len(present)
    if n_stages == 0:
        return

    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 8))
    if n_stages == 1:
        axes = axes.reshape(2, 1)

    lag_vals, l_idx = lag_index(tr)
    n_lags = len(lag_vals)
    tl = lag_tick_labels(lag_vals)

    for ci, stage in enumerate(present):
        sub = tr[tr['stage'] == stage]
        for ri, (metric, cmap, label) in enumerate([
            ('synergy', 'Reds', 'Synergy'),
            ('redundancy', 'Greens', 'Redundancy'),
        ]):
            ax = axes[ri, ci]
            piv = sub.pivot_table(index='lag1_min', columns='lag2_min',
                                  values=metric, aggfunc='mean')
            mat = np.full((n_lags, n_lags), np.nan)
            for l1v in piv.index:
                for l2v in piv.columns:
                    val = piv.loc[l1v, l2v]
                    if not np.isnan(val) and l1v in l_idx and l2v in l_idx:
                        mat[l_idx[l1v], l_idx[l2v]] = val

            sns.heatmap(mat, ax=ax, cmap=cmap, mask=np.isnan(mat),
                        cbar_kws={'label': 'bits', 'shrink': 0.7},
                        xticklabels=tl,
                        yticklabels=tl)
            if ri == 0:
                ax.set_title(f'{stage}\n(n={len(sub["time_min"].unique())} windows)',
                             fontweight='bold', color=STAGE_COLORS.get(stage, 'black'))
            if ci == 0:
                ax.set_ylabel(f'{label}\nLag 1 (min)')
            else:
                ax.set_ylabel('')
            ax.set_xlabel('Lag 2 (min)' if ri == 1 else '')

    plt.suptitle('PID atom matrices by sleep stage',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 11 — TOTAL INFORMATION & ATOM FRACTIONS PER STAGE
# =============================================================================

def plot_atom_fractions(tr, save_path=None):
    """
    Stacked bar chart: fraction of total MI each atom represents per stage.
    Shows whether stages differ in *how* they represent information, not just
    how much.
    """
    tr_common = common_lag_range(tr)
    agg = tr_common.groupby(['time_min', 'stage']).agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()
    agg = agg[agg['stage'].isin(STAGE_ORDER)]
    agg['total'] = agg['redundancy'] + agg['synergy'] + agg['unique_0'] + agg['unique_1']
    for col in ['redundancy', 'synergy', 'unique_0', 'unique_1']:
        agg[f'{col}_frac'] = agg[col] / agg['total'].replace(0, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: total MI per stage
    sns.boxplot(data=agg, x='stage', y='total', order=STAGE_ORDER,
                palette=STAGE_COLORS, ax=axes[0], showfliers=False)
    axes[0].set_ylabel('Total MI (bits)')
    axes[0].set_xlabel('Sleep stage')
    axes[0].set_title('Total mutual information', fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # stats
    groups = [agg.loc[agg['stage'] == s, 'total'].dropna().values
              for s in STAGE_ORDER if len(agg.loc[agg['stage'] == s, 'total'].dropna()) > 0]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        axes[0].set_title(f'Total MI\nKW: H={H:.1f}, p={p:.1e}', fontweight='bold')

    # Right: stacked bar of mean fractions
    frac_means = agg.groupby('stage')[
        ['redundancy_frac', 'synergy_frac', 'unique_0_frac', 'unique_1_frac']
    ].mean()
    frac_means = frac_means.reindex(STAGE_ORDER).dropna(how='all')

    bar_colors = ['green', 'red', 'steelblue', 'mediumpurple']
    bar_labels = ['Redundancy', 'Synergy', 'Unique₁', 'Unique₂']
    bottom = np.zeros(len(frac_means))
    x = np.arange(len(frac_means))

    for col, color, label in zip(
        ['redundancy_frac', 'synergy_frac', 'unique_0_frac', 'unique_1_frac'],
        bar_colors, bar_labels
    ):
        vals = frac_means[col].values
        axes[1].bar(x, vals, bottom=bottom, color=color, label=label,
                    alpha=0.8, edgecolor='white', linewidth=0.5)
        bottom += vals

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(frac_means.index)
    axes[1].set_ylabel('Fraction of total MI')
    axes[1].set_xlabel('Sleep stage')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)

    # KW on synergy fraction across stages
    syn_frac_groups = [agg.loc[agg['stage'] == s, 'synergy_frac'].dropna().values
                       for s in STAGE_ORDER
                       if len(agg.loc[agg['stage'] == s, 'synergy_frac'].dropna()) > 0]
    if len(syn_frac_groups) >= 2:
        H, p = kruskal(*syn_frac_groups)
        axes[1].set_title(f'PID atom composition\nSynergy frac. KW: H={H:.1f}, p={p:.1e}',
                          fontweight='bold')
    else:
        axes[1].set_title('PID atom composition', fontweight='bold')
    axes[1].legend(fontsize=8, loc='upper right')

    plt.suptitle('Information quantity vs. composition across sleep stages',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 12 — NREM-REM CYCLE DYNAMICS
# =============================================================================

def plot_nrem_rem_cycles(tr, stage_labels, save_path=None):
    """
    Detect NREM→REM cycles and show how PID atoms evolve relative to
    REM onset. Aligns windows to REM-onset and averages across cycles.
    """
    # find REM onsets: first window of each REM bout
    rem_onsets = []
    in_rem = False
    for i, s in enumerate(stage_labels):
        if s == 'REM' and not in_rem:
            rem_onsets.append(i)
            in_rem = True
        elif s != 'REM':
            in_rem = False

    if len(rem_onsets) < 1:
        print("    No REM onsets found, skipping NREM-REM cycle plot")
        return

    # collect PID aligned to REM onset (±15 min window around onset)
    half_win = 15  # minutes before/after REM onset
    agg = tr.groupby('time_min').agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()
    agg['unique_total'] = agg['unique_0'] + agg['unique_1']

    aligned = []
    for onset_win in rem_onsets:
        onset_min = onset_win * MIN_PER_WINDOW
        for _, row in agg.iterrows():
            dt = row['time_min'] - onset_min
            if -half_win <= dt <= half_win:
                aligned.append({
                    'dt': dt,
                    'redundancy': row['redundancy'],
                    'synergy': row['synergy'],
                    'unique_total': row['unique_total'],
                    'cycle': rem_onsets.index(onset_win),
                })

    if not aligned:
        return

    df_al = pd.DataFrame(aligned)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    metrics = [('redundancy', 'green', 'Redundancy'),
               ('synergy', 'red', 'Synergy'),
               ('unique_total', 'blue', 'Total Unique')]

    for ax, (col, color, label) in zip(axes, metrics):
        means = df_al.groupby('dt')[col].agg(['mean', 'sem']).reset_index()
        ax.plot(means['dt'], means['mean'], color=color, lw=2)
        ax.fill_between(means['dt'],
                         means['mean'] - means['sem'],
                         means['mean'] + means['sem'],
                         color=color, alpha=0.2)
        ax.axvline(0, color='red', ls='--', lw=1.5, alpha=0.7, label='REM onset')
        ax.set_ylabel(f'{label}\n(bits)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Time relative to REM onset (min)')
    plt.suptitle(f'PID dynamics around NREM→REM transitions\n'
                 f'(n={len(rem_onsets)} cycles, mean ± SEM)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 13 — EFFECT SIZE: COHEN'S d BETWEEN STAGE PAIRS
# =============================================================================

def plot_effect_sizes(tr, save_path=None):
    """
    Heatmap of Cohen's d between all stage pairs for each PID atom.
    Highlights which stage contrasts have the largest information differences.
    """
    tr_common = common_lag_range(tr)
    agg = tr_common.groupby(['time_min', 'stage']).agg(
        redundancy=('redundancy', 'mean'),
        synergy=('synergy', 'mean'),
        unique_0=('unique_0', 'mean'),
        unique_1=('unique_1', 'mean'),
    ).reset_index()
    agg['unique_total'] = agg['unique_0'] + agg['unique_1']
    agg['ratio'] = agg['synergy'] / agg['redundancy'].replace(0, np.nan)
    agg = agg[agg['stage'].isin(STAGE_ORDER)]

    present = [s for s in STAGE_ORDER if s in agg['stage'].unique()]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = ['redundancy', 'synergy', 'unique_total', 'ratio']
    labels = ['Redundancy', 'Synergy', 'Total Unique', 'S/R Ratio']

    for ax, metric, label in zip(axes, metrics, labels):
        n = len(present)
        d_mat = np.full((n, n), np.nan)
        p_mat = np.full((n, n), np.nan)

        for i, s1 in enumerate(present):
            for j, s2 in enumerate(present):
                if i >= j:
                    continue
                v1 = agg.loc[agg['stage'] == s1, metric].dropna().values
                v2 = agg.loc[agg['stage'] == s2, metric].dropna().values
                if len(v1) < 3 or len(v2) < 3:
                    continue
                pooled_std = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1) +
                                      (len(v2)-1)*np.var(v2, ddof=1)) /
                                     (len(v1) + len(v2) - 2))
                if pooled_std > 0:
                    d_mat[i, j] = (np.mean(v1) - np.mean(v2)) / pooled_std
                    d_mat[j, i] = -d_mat[i, j]
                _, p_val = mannwhitneyu(v1, v2, alternative='two-sided')
                p_mat[i, j] = p_val
                p_mat[j, i] = p_val

        # FDR correction within this metric
        upper_p = []
        upper_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isnan(p_mat[i, j]):
                    upper_p.append(p_mat[i, j])
                    upper_idx.append((i, j))
        if upper_p:
            adj_p = benjamini_hochberg(upper_p)
            p_mat_fdr = np.full((n, n), np.nan)
            for (i, j), ap in zip(upper_idx, adj_p):
                p_mat_fdr[i, j] = ap
                p_mat_fdr[j, i] = ap
        else:
            p_mat_fdr = p_mat

        vmax = np.nanmax(np.abs(d_mat)) if not np.all(np.isnan(d_mat)) else 1
        sns.heatmap(d_mat, ax=ax, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=present, yticklabels=present,
                    annot=True, fmt='.2f', annot_kws={'fontsize': 8},
                    cbar_kws={'label': "Cohen's d", 'shrink': 0.8},
                    mask=np.isnan(d_mat))
        # mark FDR-significant cells
        for i in range(n):
            for j in range(n):
                if not np.isnan(p_mat_fdr[i, j]) and p_mat_fdr[i, j] < 0.05:
                    ax.text(j + 0.5, i + 0.8, '*', ha='center', va='center',
                            fontsize=10, fontweight='bold')
        ax.set_title(label, fontweight='bold')

    plt.suptitle("Effect sizes (Cohen's d) between sleep stages\n"
                 "(* = Mann-Whitney FDR-corrected p < 0.05)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# PLOT 14 — BLOCK-PERMUTATION SIGNIFICANCE TEST
# =============================================================================

def compute_block_permutation_null(signal_disc, fs, max_lag_min, window_sec,
                                   n_bins, n_permutations=100, seed=42):
    """
    Block-permutation null for the global PID matrix.

    Shuffles which windows map to which time positions (preserving
    within-window sample structure) to destroy inter-window temporal order.
    The resulting null distribution quantifies the PID expected by chance.
    """
    rng = np.random.default_rng(seed)
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    data = signal_disc[:n_windows * W].reshape(n_windows, W)

    max_lag_w = max_lag_min * (60 // window_sec)
    lag_pairs = [(l1, l2) for l1 in range(1, max_lag_w)
                 for l2 in range(l1 + 1, max_lag_w + 1)]
    n_pairs = len(lag_pairs)
    null_vals = np.zeros((n_permutations, n_pairs, 4))

    for perm in range(n_permutations):
        perm_order = rng.permutation(n_windows)
        perm_flat = data[perm_order].reshape(-1)

        for li, (lag1, lag2) in enumerate(lag_pairs):
            off1 = lag1 * W
            off2 = lag2 * W
            n_valid = len(perm_flat) - off2

            target = perm_flat[off2: off2 + n_valid]
            src1   = perm_flat[off2 - off1: off2 - off1 + n_valid]
            src2   = perm_flat[0: n_valid]

            pid = compute_pid_from_arrays(src1, src2, target, n_bins)
            null_vals[perm, li] = [pid['redundancy'], pid['synergy'],
                                   pid['unique_0'], pid['unique_1']]

        if (perm + 1) % 20 == 0:
            print(f"    Permutation {perm+1}/{n_permutations}")

    return null_vals


def plot_block_permutation(global_results, null_vals, save_path=None):
    """
    Significance map: -log10(p) for each lag pair, with * marking p < 0.05.
    p-value = fraction of permutations >= observed (one-tailed).
    """
    lag_pairs_w = [(l1, l2) for l1 in range(1, MAX_LAG_WINDOWS)
                   for l2 in range(l1 + 1, MAX_LAG_WINDOWS + 1)]

    # Build data-driven lag index
    all_lag_vals = sorted(set(global_results['lag1_min'].unique()) |
                          set(global_results['lag2_min'].unique()))
    n_lags = len(all_lag_vals)
    l_idx = {v: i for i, v in enumerate(all_lag_vals)}
    tick_labels = [f'{v:g}' for v in all_lag_vals]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    comps = ['redundancy', 'synergy', 'unique_0', 'unique_1']
    titles = ['Redundancy', 'Synergy', 'Unique₁', 'Unique₂']
    cmaps = ['Greens', 'Reds', 'Blues', 'Purples']
    comp_idx = {c: i for i, c in enumerate(comps)}

    for ax, comp, title, cmap in zip(axes.flat, comps, titles, cmaps):
        p_mat = np.full((n_lags, n_lags), np.nan)
        ci = comp_idx[comp]

        for li, (l1w, l2w) in enumerate(lag_pairs_w):
            l1_min = round(l1w * MIN_PER_WINDOW, 4)
            l2_min = round(l2w * MIN_PER_WINDOW, 4)
            row = global_results[(global_results['lag1_min'] == l1_min) &
                                 (global_results['lag2_min'] == l2_min)]
            if row.empty:
                continue
            observed = row[comp].values[0]
            null_dist = null_vals[:, li, ci]
            # one-tailed p-value with continuity correction
            p = (np.sum(null_dist >= observed) + 1) / (len(null_dist) + 1)
            if l1_min in l_idx and l2_min in l_idx:
                p_mat[l_idx[l1_min], l_idx[l2_min]] = -np.log10(max(p, 1e-10))

        sns.heatmap(p_mat, ax=ax, cmap=cmap, mask=np.isnan(p_mat),
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    cbar_kws={'label': '-log₁₀(p)'})
        # mark significant cells
        for l1i in range(n_lags):
            for l2i in range(n_lags):
                if not np.isnan(p_mat[l1i, l2i]) and p_mat[l1i, l2i] > -np.log10(0.05):
                    ax.text(l2i + 0.5, l1i + 0.5, '*', ha='center', va='center',
                            fontsize=12, fontweight='bold', color='white')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Lag 2 (min)')
        ax.set_ylabel('Lag 1 (min)')

    plt.suptitle(f'Block-permutation significance test\n'
                 f'(n={null_vals.shape[0]} permutations, * = p < 0.05)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    

# =============================================================================
# PLOT 15 — OPTIMAL TIMESCALE PROFILES PER STAGE
# =============================================================================

def plot_optimal_timescales(tr, save_path=None):
    """
    Three-tier analysis of optimal timescales per stage.

    Row 1: Marginal lag profiles — mean PID atom vs lag1 (averaged over all
            lag2) and vs lag2 (averaged over all lag1), per stage.
    Row 2: Mean-lag profiles — PID vs (lag1+lag2)/2, collapsing the 2D lag
            space to a single absolute-timescale axis.
    Row 3: Center-of-mass — synergy-weighted and redundancy-weighted
            characteristic timescale per stage with bootstrap 95% CIs.
    """
    tr = tr.copy()
    tr['mean_lag'] = (tr['lag1_min'] + tr['lag2_min']) / 2.0
    tr = tr[tr['stage'].isin(STAGE_ORDER)]

    atoms = [('synergy', 'red', 'Synergy'),
             ('redundancy', 'green', 'Redundancy'),
             ('unique_total', 'blue', 'Total Unique')]
    tr['unique_total'] = tr['unique_0'] + tr['unique_1']

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ---- ROW 1: Marginal lag profiles (synergy, redundancy, unique) ---------
    for col_i, (atom, color, label) in enumerate(atoms):
        ax = fig.add_subplot(gs[0, col_i])
        for stage in STAGE_ORDER:
            sub = tr[tr['stage'] == stage]
            if sub.empty:
                continue
            # marginal over lag1: for each lag1 value, avg atom across all lag2
            marg1 = sub.groupby('lag1_min')[atom].mean()
            ax.plot(marg1.index, marg1.values, color=STAGE_COLORS[stage],
                    lw=2, label=f'{stage} (lag₁)')
            # marginal over lag2: for each lag2 value, avg atom across all lag1
            marg2 = sub.groupby('lag2_min')[atom].mean()
            ax.plot(marg2.index, marg2.values, color=STAGE_COLORS[stage],
                    lw=2, ls='--', alpha=0.7)
        ax.set_xlabel('Lag (min)')
        ax.set_ylabel(f'{label} (bits)')
        ax.set_title(f'{label}\n(solid=lag₁ marginal, dashed=lag₂ marginal)',
                     fontweight='bold', fontsize=10)
        ax.grid(alpha=0.3)
        if col_i == 0:
            ax.legend(fontsize=7, ncol=1)

    # ---- ROW 2: Mean-lag profiles -------------------------------------------
    for col_i, (atom, color, label) in enumerate(atoms):
        ax = fig.add_subplot(gs[1, col_i])
        for stage in STAGE_ORDER:
            sub = tr[tr['stage'] == stage]
            if sub.empty:
                continue
            grp = sub.groupby('mean_lag')[atom].agg(['mean', 'sem']).reset_index()
            ax.plot(grp['mean_lag'], grp['mean'], color=STAGE_COLORS[stage],
                    lw=2, label=stage)
            ax.fill_between(grp['mean_lag'],
                            grp['mean'] - grp['sem'],
                            grp['mean'] + grp['sem'],
                            color=STAGE_COLORS[stage], alpha=0.12)
        ax.set_xlabel('Mean lag  (lag₁+lag₂)/2  (min)')
        ax.set_ylabel(f'{label} (bits)')
        ax.set_title(f'{label} vs absolute timescale', fontweight='bold',
                     fontsize=10)
        ax.grid(alpha=0.3)
        if col_i == 0:
            ax.legend(fontsize=7)

    # ---- ROW 3: Center-of-mass bar chart ------------------------------------
    # compute per-window CoM, then bootstrap 95% CI across windows
    rng = np.random.default_rng(42)
    n_boot = 1000
    com_atoms = [('synergy', 'red', 'Synergy'), ('redundancy', 'green', 'Redundancy')]

    for col_i, (atom, color, label) in enumerate(com_atoms):
        ax = fig.add_subplot(gs[2, col_i])
        means, ci_lo, ci_hi = [], [], []
        present = []
        for stage in STAGE_ORDER:
            sub = tr[tr['stage'] == stage]
            if sub.empty:
                continue
            present.append(stage)
            # aggregate per window → one CoM per window
            window_coms = []
            for w, wdf in sub.groupby('time_min'):
                weights = wdf[atom].values
                total = weights.sum()
                if total > 0:
                    com = np.sum(weights * wdf['mean_lag'].values) / total
                    window_coms.append(com)
            window_coms = np.array(window_coms)
            if len(window_coms) == 0:
                means.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                continue
            means.append(np.mean(window_coms))
            # bootstrap CI
            boot_means = np.array([
                np.mean(rng.choice(window_coms, size=len(window_coms),
                                   replace=True))
                for _ in range(n_boot)
            ])
            ci_lo.append(np.percentile(boot_means, 2.5))
            ci_hi.append(np.percentile(boot_means, 97.5))

        x = np.arange(len(present))
        colors_bar = [STAGE_COLORS[s] for s in present]
        errs = np.array([[m - lo, hi - m]
                         for m, lo, hi in zip(means, ci_lo, ci_hi)]).T
        ax.bar(x, means, color=colors_bar, edgecolor='white', linewidth=0.5,
               alpha=0.85)
        ax.errorbar(x, means, yerr=errs, fmt='none', ecolor='black',
                    capsize=4, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(present)
        ax.set_ylabel('Characteristic timescale (min)')
        ax.set_xlabel('Sleep stage')
        ax.grid(axis='y', alpha=0.3)

        # KW on per-window CoM across stages
        com_groups = {}
        for stage in STAGE_ORDER:
            sub = tr[tr['stage'] == stage]
            if sub.empty:
                continue
            wcoms = []
            for w, wdf in sub.groupby('time_min'):
                wts = wdf[atom].values
                tot = wts.sum()
                if tot > 0:
                    wcoms.append(np.sum(wts * wdf['mean_lag'].values) / tot)
            if wcoms:
                com_groups[stage] = np.array(wcoms)
        grp_list = [v for v in com_groups.values() if len(v) > 0]
        if len(grp_list) >= 2:
            H, p = kruskal(*grp_list)
            ax.set_title(f'{label} center-of-mass\nKW: H={H:.1f}, p={p:.1e}',
                         fontweight='bold', fontsize=10)
        else:
            ax.set_title(f'{label} center-of-mass\n(bootstrap 95% CI)',
                         fontweight='bold', fontsize=10)

    # rightmost panel: CoM table with numeric values
    ax_tab = fig.add_subplot(gs[2, 2])
    ax_tab.axis('off')
    table_data = []
    for atom, _, label in com_atoms:
        for stage in STAGE_ORDER:
            sub = tr[tr['stage'] == stage]
            if sub.empty:
                continue
            window_coms = []
            for w, wdf in sub.groupby('time_min'):
                weights = wdf[atom].values
                total = weights.sum()
                if total > 0:
                    com = np.sum(weights * wdf['mean_lag'].values) / total
                    window_coms.append(com)
            if window_coms:
                m = np.mean(window_coms)
                s = np.std(window_coms) / np.sqrt(len(window_coms))
                table_data.append([label, stage, f'{m:.2f}', f'±{s:.2f}'])
    if table_data:
        tbl = ax_tab.table(cellText=table_data,
                           colLabels=['Atom', 'Stage', 'CoM (min)', 'SEM'],
                           loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.3)
    ax_tab.set_title('Characteristic timescales', fontweight='bold',
                     fontsize=10)

    plt.suptitle('Optimal timescale analysis per sleep stage\n'
                 '(marginal profiles → mean-lag curves → center-of-mass)',
                 fontsize=14, fontweight='bold')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EEG SLEEP — TEMPORAL PID AT MINUTE-SCALE LAGS")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ---- 1. download --------------------------------------------------------
    print(f"\n{'='*70}\nSTEP 1 — DOWNLOAD DATA\n{'='*70}")
    subject_dir = download_subject(SUBJECT)

    # ---- 2. load data -------------------------------------------------------
    print(f"\n{'='*70}\nSTEP 2 — LOAD EEG & SLEEP STAGES\n{'='*70}")
    signal, fs, stages, ch_name = load_eeg_and_stages(
        subject_dir, channel=CHANNEL, duration_hours=DURATION_HOURS
    )

    # ---- 2b. preprocess EEG -------------------------------------------------
    print(f"\n{'='*70}\nSTEP 2b — PREPROCESS (bandpass 0.5-60 Hz, notch 50/60 Hz)\n{'='*70}")
    signal = preprocess_eeg(signal, fs)
    print(f"  Filtered: bandpass 0.5-60 Hz, notch 50 & 60 Hz")

    # ---- 3. discretise ------------------------------------------------------
    print(f"\n{'='*70}\nSTEP 3 — DISCRETISE SIGNAL\n{'='*70}")

    W = int(WINDOW_SEC * fs)
    n_windows = len(signal) // W

    # Detect bad windows (flat-line disconnections, NaN)
    good_windows = detect_bad_windows(signal, fs, WINDOW_SEC)

    if DISCRETIZE_PER_WINDOW:
        print("  Mode: per-window quantile binning (amplitude-independent)")
        signal_disc = discretize_signal_per_window(signal, fs, WINDOW_SEC,
                                                    n_bins=N_BINS)
    else:
        print("  Mode: global quantile binning")
        signal_disc = discretize_signal(signal, n_bins=N_BINS)

    print(f"  {len(signal)} samples  →  {N_BINS} bins  ({n_windows} windows)")
    print(f"  Good windows: {good_windows.sum()}/{n_windows}")

    # ---- 4. stage labels per window -----------------------------------------
    stage_labels = get_stage_per_window(stages, n_windows, WINDOW_SEC)
    scounts = pd.Series(stage_labels).value_counts()
    print(f"\n  Stage distribution:")
    for s, c in scounts.items():
        print(f"    {s:5s}: {c:3d} windows ({c/n_windows*100:.1f}%)")

    # ---- 5. global PID matrix -----------------------------------------------
    print(f"\n{'='*70}\nSTEP 4 — GLOBAL PID MATRIX\n{'='*70}")
    chash = config_hash()
    global_csv = RESULTS_DIR / f"global_pid_matrix_{chash}.csv"
    if global_csv.exists():
        print("  Loading cached results …")
        global_results = pd.read_csv(global_csv)
    else:
        t0 = time.time()
        global_results = compute_global_pid_matrix(signal_disc, fs,
                                                   MAX_LAG_MIN, WINDOW_SEC)
        print(f"  Done in {time.time()-t0:.1f}s")
        global_results.to_csv(global_csv, index=False)
        print(f"  Saved → {global_csv.name}")

    # ---- 6. time-resolved PID -----------------------------------------------
    print(f"\n{'='*70}\nSTEP 5 — TIME-RESOLVED PID\n{'='*70}")
    tr_csv = RESULTS_DIR / f"timeresolved_pid_{chash}.csv"
    if tr_csv.exists():
        print("  Loading cached results …")
        tr_results = pd.read_csv(tr_csv)
    else:
        t0 = time.time()
        tr_results = compute_timeresolved_pid(signal_disc, fs,
                                              MAX_LAG_MIN, WINDOW_SEC,
                                              good_windows=good_windows)
        print(f"  Done in {time.time()-t0:.1f}s")
        tr_results.to_csv(tr_csv, index=False)
        print(f"  Saved → {tr_csv.name}")

    # attach stage label to each row (target window stage)
    tr_results['stage'] = tr_results['window'].map(
        lambda w: stage_labels[int(w)] if int(w) < len(stage_labels) else '?'
    )

    # ---- same-stage filter: keep only triplets where target + both sources
    #      share the same sleep stage
    n_before = len(tr_results)
    stages_arr = np.array(stage_labels)
    w_t  = tr_results['window'].values.astype(int)
    w_s1 = w_t - np.round(tr_results['lag1_min'].values * WINDOWS_PER_MIN).astype(int)
    w_s2 = w_t - np.round(tr_results['lag2_min'].values * WINDOWS_PER_MIN).astype(int)
    in_bounds = (w_s1 >= 0) & (w_s2 >= 0) & (w_s1 < len(stages_arr)) & (w_s2 < len(stages_arr))
    stage_t  = np.where(in_bounds, stages_arr[np.clip(w_t, 0, len(stages_arr)-1)], '')
    stage_s1 = np.where(in_bounds, stages_arr[np.clip(w_s1, 0, len(stages_arr)-1)], '?')
    stage_s2 = np.where(in_bounds, stages_arr[np.clip(w_s2, 0, len(stages_arr)-1)], '??')
    same_stage = in_bounds & (stage_t == stage_s1) & (stage_t == stage_s2)
    if CONTINUOUS_STAGE_FILTER:
        # stricter: every window from source2 to target must be same stage
        for i in np.where(same_stage)[0]:
            span = stages_arr[w_s2[i] : w_t[i] + 1]
            if not np.all(span == stage_t[i]):
                same_stage[i] = False
    tr_results = tr_results[same_stage].copy()
    filter_name = 'continuous-bout' if CONTINUOUS_STAGE_FILTER else '3-point'
    print(f"  Same-stage filter ({filter_name}): {n_before} → {len(tr_results)} rows "
          f"({len(tr_results)/max(n_before,1)*100:.1f}% kept)")

    # ---- 7. AR(1) baseline ----------------------------------------------------
    print(f"\n{'='*70}\nSTEP 6 — AR(1) BASELINE\n{'='*70}")

    N_AR_REALISATIONS = 5

    ar1_global_csv = RESULTS_DIR / f"ar1_global_pid_mean_{chash}.csv"
    ar1_global_std_csv = RESULTS_DIR / f"ar1_global_pid_std_{chash}.csv"
    if ar1_global_csv.exists():
        print("  Loading cached global AR(1) baseline …")
        ar1_global_mean = pd.read_csv(ar1_global_csv)
        ar1_global_std  = pd.read_csv(ar1_global_std_csv)
    else:
        print("  Computing global AR(1) baseline PID …")
        t0 = time.time()
        ar1_global_mean, ar1_global_std = compute_ar1_global_pid(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS,
            n_realisations=N_AR_REALISATIONS)
        print(f"  Done in {time.time()-t0:.1f}s")
        ar1_global_mean.to_csv(ar1_global_csv, index=False)
        ar1_global_std.to_csv(ar1_global_std_csv, index=False)

    ar1_tr_csv = RESULTS_DIR / f"ar1_timeresolved_pid_{chash}.csv"
    if ar1_tr_csv.exists():
        print("  Loading cached time-resolved AR(1) baseline …")
        ar1_tr = pd.read_csv(ar1_tr_csv)
    else:
        print("  Computing time-resolved AR(1) baseline …")
        t0 = time.time()
        ar1_tr = compute_ar1_timeresolved_pid(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS,
            good_windows=good_windows,
            ar1_global_mean=ar1_global_mean)
        print(f"  Done in {time.time()-t0:.1f}s")
        ar1_tr.to_csv(ar1_tr_csv, index=False)

    # attach stage labels + same-stage filter for AR(1) too
    ar1_tr['stage'] = ar1_tr['window'].map(
        lambda w: stage_labels[int(w)] if int(w) < len(stage_labels) else '?'
    )
    # apply same-stage filter (same mask logic)
    n_before_ar1 = len(ar1_tr)
    w_t_ar1  = ar1_tr['window'].values.astype(int)
    w_s1_ar1 = w_t_ar1 - np.round(ar1_tr['lag1_min'].values * WINDOWS_PER_MIN).astype(int)
    w_s2_ar1 = w_t_ar1 - np.round(ar1_tr['lag2_min'].values * WINDOWS_PER_MIN).astype(int)
    in_b = (w_s1_ar1 >= 0) & (w_s2_ar1 >= 0) & (w_s1_ar1 < len(stages_arr)) & (w_s2_ar1 < len(stages_arr))
    st_t  = np.where(in_b, stages_arr[np.clip(w_t_ar1, 0, len(stages_arr)-1)], '')
    st_s1 = np.where(in_b, stages_arr[np.clip(w_s1_ar1, 0, len(stages_arr)-1)], '?')
    st_s2 = np.where(in_b, stages_arr[np.clip(w_s2_ar1, 0, len(stages_arr)-1)], '??')
    same_ar1 = in_b & (st_t == st_s1) & (st_t == st_s2)
    if CONTINUOUS_STAGE_FILTER:
        for i in np.where(same_ar1)[0]:
            span = stages_arr[w_s2_ar1[i] : w_t_ar1[i] + 1]
            if not np.all(span == st_t[i]):
                same_ar1[i] = False
    ar1_tr = ar1_tr[same_ar1].copy()
    print(f"  AR(1) same-stage filter ({filter_name}): {n_before_ar1} → {len(ar1_tr)} rows")

    # ---- 8. autocorrelation -------------------------------------------------
    print(f"\n{'='*70}\nSTEP 7 — AUTOCORRELATION\n{'='*70}")

    autocorr_csv = RESULTS_DIR / f"autocorrelation_{chash}.csv"
    if autocorr_csv.exists():
        print("  Loading cached autocorrelation …")
        autocorr_df = pd.read_csv(autocorr_csv)
    else:
        print("  Computing per-window cross-window autocorrelation …")
        t0 = time.time()
        autocorr_df = compute_autocorrelation_per_window(
            signal, fs, MAX_LAG_MIN, WINDOW_SEC)
        print(f"  Done in {time.time()-t0:.1f}s")
        autocorr_df.to_csv(autocorr_csv, index=False)

    # ---- 9. block-permutation test ------------------------------------------
    print(f"\n{'='*70}\nSTEP 8b — BLOCK-PERMUTATION NULL\n{'='*70}")

    N_PERMUTATIONS = 100
    perm_npz = RESULTS_DIR / f"block_perm_null_{chash}.npz"
    if perm_npz.exists():
        print("  Loading cached permutation null …")
        perm_data = np.load(perm_npz)
        null_vals = perm_data['null_vals']
    else:
        print(f"  Computing block-permutation null ({N_PERMUTATIONS} perms) …")
        t0 = time.time()
        null_vals = compute_block_permutation_null(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS,
            n_permutations=N_PERMUTATIONS)
        print(f"  Done in {time.time()-t0:.1f}s")
        np.savez(perm_npz, null_vals=null_vals)
        print(f"  Saved → {perm_npz.name}")

    # ---- 10. plots -----------------------------------------------------------
    print(f"\n{'='*70}\nSTEP 9 — GENERATING PLOTS\n{'='*70}")

    print("   1/15  Global PID matrix")
    plot_global_matrix(global_results,
                       save_path=RESULTS_DIR / "global_pid_matrix.png")

    print("   2/15  Combined time series (hypnogram + PID + AR(1) + bands)")
    plot_combined_timeseries(tr_results, ar1_tr, stage_labels,
                             save_path=RESULTS_DIR / "combined_timeseries.png")

    print("   3/15  Time × Lag heatmaps (lag1 = 1 min)")
    plot_time_lag_heatmaps(tr_results, stage_labels,
                           save_path=RESULTS_DIR / "time_lag_heatmaps.png")

    print("   4/15  Stage comparison")
    plot_stage_comparison(tr_results,
                          save_path=RESULTS_DIR / "stage_comparison.png")

    print("   5/15  Lag-difference heatmap")
    plot_lagdiff_heatmap(tr_results, stage_labels,
                         save_path=RESULTS_DIR / "lagdiff_heatmap.png")

    print("   6/15  Global PID: Actual vs AR(1)")
    plot_global_matrix_vs_ar1(global_results, ar1_global_mean,
                              save_path=RESULTS_DIR / "global_pid_vs_ar1.png")

    print("   7/15  Stage comparison vs AR(1)")
    plot_stage_comparison_vs_ar1(tr_results, ar1_tr,
                                 save_path=RESULTS_DIR / "stage_comparison_vs_ar1.png")

    print("   8/15  Autocorrelation vs PID")
    plot_autocorrelation_vs_pid(autocorr_df, tr_results, stage_labels,
                                save_path=RESULTS_DIR / "autocorrelation_vs_pid.png")

    print("   9/15  S/R ratio vs Autocorrelation diagnostic")
    plot_sr_ratio_vs_autocorr(autocorr_df, tr_results,
                              save_path=RESULTS_DIR / "sr_ratio_vs_autocorr.png")

    print("  10/15  PID atom matrices per stage")
    plot_pid_per_stage_matrix(tr_results,
                              save_path=RESULTS_DIR / "pid_per_stage_matrix.png")

    print("  11/15  Atom fractions per stage")
    plot_atom_fractions(tr_results,
                        save_path=RESULTS_DIR / "atom_fractions.png")

    print("  12/15  NREM→REM cycle dynamics")
    plot_nrem_rem_cycles(tr_results, stage_labels,
                         save_path=RESULTS_DIR / "nrem_rem_cycles.png")

    print("  13/15  Effect sizes (Cohen's d)")
    plot_effect_sizes(tr_results,
                      save_path=RESULTS_DIR / "effect_sizes.png")

    print("  14/15  Block-permutation significance")
    plot_block_permutation(global_results, null_vals,
                           save_path=RESULTS_DIR / "block_permutation.png")

    print("  15/15  Optimal timescale profiles")
    plot_optimal_timescales(tr_results,
                            save_path=RESULTS_DIR / "optimal_timescales.png")

    # ---- summary ------------------------------------------------------------
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  Subject     : {SUBJECT}")
    print(f"  Channel     : {ch_name}")
    print(f"  Duration    : {DURATION_HOURS} h  ({n_windows} windows × {WINDOW_SEC}s)")
    print(f"  Window      : {WINDOW_SEC}s")
    print(f"  Lag range   : 1–{MAX_LAG_MIN} min")
    print(f"  Same-stage  : {'continuous-bout' if CONTINUOUS_STAGE_FILTER else '3-point'} filter")
    print(f"  Bins        : {N_BINS}")
    print(f"  AR(1) reals : {N_AR_REALISATIONS}")

    print(f"\n  Global PID — Actual vs AR(1) (mean across all lag pairs):")
    for m in ['redundancy', 'synergy', 'unique_0', 'unique_1']:
        act_val = global_results[m].mean()
        ar1_val = ar1_global_mean[m].mean()
        excess  = act_val - ar1_val
        print(f"    {m:12s}: actual={act_val:.4f}  AR(1)={ar1_val:.4f}  "
              f"excess={excess:+.4f}")

    print(f"\n  Mean PID by sleep stage (actual):")
    for m in ['synergy', 'redundancy']:
        act_stage = tr_results.groupby('stage')[m].mean()
        ar1_stage = ar1_tr.groupby('stage')[m].mean()
        print(f"    {m}:")
        for s in STAGE_ORDER:
            if s in act_stage.index:
                ar1_v = ar1_stage[s] if s in ar1_stage.index else 0
                print(f"      {s:5s}: actual={act_stage[s]:.4f}  "
                      f"AR(1)={ar1_v:.4f}  excess={act_stage[s]-ar1_v:+.4f}")

    print(f"\n  Mean S/R ratio by sleep stage:")
    sr = tr_results.groupby(['time_min', 'stage']).agg(
        synergy=('synergy', 'mean'),
        redundancy=('redundancy', 'mean'),
    ).reset_index()
    sr['ratio'] = sr['synergy'] / sr['redundancy'].replace(0, np.nan)
    for s in STAGE_ORDER:
        sub = sr[sr['stage'] == s]['ratio'].dropna()
        if len(sub) > 0:
            print(f"    {s:5s}: median={sub.median():.2f}  "
                  f"IQR=[{sub.quantile(0.25):.2f}, {sub.quantile(0.75):.2f}]")

    print(f"\n  Results saved to {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*")):
        print(f"    {f.name}")

    print(f"\nFinished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
