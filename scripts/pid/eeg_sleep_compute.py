"""
EEG Sleep Temporal PID — Compute CSVs across electrodes
========================================================

Runs the PID computation pipeline for all 6 PSG EEG electrodes
(F3, F4, C3, C4, O1, O2) and saves per-electrode CSV/NPZ results.

No plotting — see eeg_sleep_plot.py for visualisation.

Usage:
    python eeg_sleep_compute.py                   # all 6 electrodes
    python eeg_sleep_compute.py --channels PSG_C3  # single channel
    python eeg_sleep_compute.py --channels PSG_F3 PSG_O1  # subset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import time
import warnings
import argparse
import hashlib
import json

warnings.filterwarnings('ignore')

from scipy.signal import lfilter, butter, filtfilt, iirnotch

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

SUBJECT = "sub-1"
ALL_EEG_CHANNELS = ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"]
DURATION_HOURS = 5
WINDOW_SEC = 30
MAX_LAG_MIN = 10
N_BINS = 6
DISCRETIZE_PER_WINDOW = True
CONTINUOUS_STAGE_FILTER = False

# Band-filtered PID: set to None for broadband, or a dict of {name: (lo, hi)} in Hz
# Example: {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (11, 16), "beta": (16, 30)}
BANDS = None
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (11, 16), "beta": (16, 30)}

WINDOWS_PER_MIN = 60 // WINDOW_SEC
MIN_PER_WINDOW = WINDOW_SEC / 60
MAX_LAG_WINDOWS = MAX_LAG_MIN * WINDOWS_PER_MIN

STAGE_ORDER = ['Wake', 'N1', 'N2', 'N3', 'REM']


def config_hash():
    """Short hash of analysis parameters for cache invalidation."""
    bands_str = "_".join(f"{k}{lo}-{hi}" for k, (lo, hi) in sorted(BANDS.items())) if BANDS else "broadband"
    params = (f"{WINDOW_SEC}_{MAX_LAG_MIN}_{N_BINS}_{DISCRETIZE_PER_WINDOW}"
              f"_{CONTINUOUS_STAGE_FILTER}_{DURATION_HOURS}_{bands_str}")
    return hashlib.md5(params.encode()).hexdigest()[:8]


def save_params(chash):
    """Write analysis parameters to a JSON file so the plot script can load them."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    params_file = RESULTS_DIR / f"params_{chash}.json"
    if params_file.exists():
        return
    params = {
        "WINDOW_SEC": WINDOW_SEC,
        "MAX_LAG_MIN": MAX_LAG_MIN,
        "N_BINS": N_BINS,
        "DISCRETIZE_PER_WINDOW": DISCRETIZE_PER_WINDOW,
        "CONTINUOUS_STAGE_FILTER": CONTINUOUS_STAGE_FILTER,
        "DURATION_HOURS": DURATION_HOURS,
        "BANDS": BANDS,
    }
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved params → {params_file}")


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


# =============================================================================
# PREPROCESSING
# =============================================================================

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


def bandpass_filter(signal, fs, lo, hi):
    """Zero-phase bandpass filter for a single frequency band."""
    nyq = fs / 2.0
    hi = min(hi, nyq - 1.0)
    b, a = butter(4, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, signal.astype(np.float64))


# =============================================================================
# DATA LOADING
# =============================================================================

def find_channel(raw, preferred="PSG_C3"):
    """Find best matching channel name in the EDF."""
    ch_names = raw.ch_names
    if preferred in ch_names:
        return preferred
    short = preferred.replace("PSG_", "")
    for ch in ch_names:
        if preferred in ch or short in ch:
            return ch
    for pattern in ["C3", "C4", "F3", "F4", "O1"]:
        for ch in ch_names:
            if pattern in ch:
                return ch
    return ch_names[0]


def load_eeg_and_stages(subject_dir, channel="PSG_C3", duration_hours=2):
    """
    Load PSG EEG channel and sleep-stage annotations.

    Returns: signal, fs, stages, ch_name
    """
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

    tmax = min(duration_hours * 3600, raw.times[-1])
    raw.crop(tmax=tmax)
    raw.pick([ch_name])
    raw.load_data(verbose=False)
    signal = raw.get_data()[0]
    print(f"  Samples  : {len(signal)}  ({len(signal)/fs/3600:.2f} h)")

    # Sleep stages
    events_files = list(subject_dir.rglob("*psg_events.tsv"))
    stages = []
    if events_files:
        events_file = events_files[0]
        print(f"  Events   : {events_file.name}")
        ev = pd.read_csv(events_file, sep='\t')
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
        print("  Stages   : none found")

    return signal, fs, stages, ch_name


def get_stage_per_window(stages, n_windows, window_sec):
    """Return the dominant sleep stage label for each window."""
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

def discretize_signal_per_window(signal, fs, window_sec, n_bins=4):
    """Discretise each window independently using its own quantiles."""
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


def discretize_signal(signal, n_bins=4):
    """Quantile-based discretisation (global bins)."""
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


def detect_bad_windows(signal, fs, window_sec, flat_threshold=1e-10):
    """Detect flat-line or NaN windows. Returns boolean mask (True = good)."""
    W = int(window_sec * fs)
    n_windows = len(signal) // W
    good = np.ones(n_windows, dtype=bool)
    for i in range(n_windows):
        seg = signal[i * W : (i + 1) * W]
        if np.std(seg) < flat_threshold:
            good[i] = False
        elif np.any(np.isnan(seg)):
            good[i] = False
    n_bad = np.sum(~good)
    if n_bad > 0:
        print(f"  Detected {n_bad} bad windows")
    return good


# =============================================================================
# PID COMPUTATION
# =============================================================================

def compute_pid_from_arrays(src1, src2, target, n_bins=4):
    """Compute PID-MMI from three aligned integer arrays."""
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


def compute_global_pid_matrix(signal_disc, fs, max_lag_min, window_sec):
    """PID for every lag pair using ALL valid sample triplets."""
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


def compute_timeresolved_pid(signal_disc, fs, max_lag_min, window_sec,
                             good_windows=None):
    """PID in sliding windows for every lag pair."""
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    max_lag_w = max_lag_min * (60 // window_sec)
    min_per_w = window_sec / 60
    first = max_lag_w
    n_pairs = max_lag_w * (max_lag_w - 1) // 2

    valid_targets = []
    for t in range(first, n_windows):
        if good_windows is not None:
            needed = [t] + [t - lag_w for lag_w in range(1, max_lag_w + 1)]
            if not all(good_windows[w] for w in needed if w < len(good_windows)):
                continue
        valid_targets.append(t)

    print(f"    Windows          : {len(valid_targets)}  (of {n_windows - first} possible)")
    print(f"    Lag pairs/window : {n_pairs}")
    print(f"    Total PID calls  : {len(valid_targets) * n_pairs}")

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
        if (idx + 1) % 5 == 0 or idx == len(valid_targets) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (idx + 1)
            eta  = rate * (len(valid_targets) - idx - 1)
            print(f"    window {idx+1}/{len(valid_targets)}  "
                  f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")
    return pd.DataFrame(results)


# =============================================================================
# AR(1) BASELINE + AUTOCORRELATION
# =============================================================================

def compute_ar1_global_pid(signal_disc, fs, max_lag_min, window_sec, n_bins,
                           n_realisations=5, seed=42):
    """Window-level AR(1) baseline for the global PID matrix."""
    rng = np.random.default_rng(seed)
    W = int(window_sec * fs)
    n_windows = len(signal_disc) // W
    data = signal_disc[:n_windows * W].reshape(n_windows, W).astype(float)

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
        noise = rng.normal(0, max(sigma_w, 1e-6), (n_windows, W))
        noise[0, :] = rng.normal(data.mean(), np.std(data), W)
        ar_cont = lfilter([1], [1, -phi_w], noise, axis=0)
        ar_cont += data.mean(axis=0, keepdims=True)

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
                                 ar1_global_mean=None):
    """AR(1) baseline per window — broadcasts global AR(1) mean."""
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
    """Compute autocorrelation R(τ) per window for τ = 1..max_lag_windows."""
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


def compute_block_permutation_null(signal_disc, fs, max_lag_min, window_sec,
                                   n_bins, n_permutations=100, seed=42):
    """Block-permutation null for the global PID matrix."""
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


# =============================================================================
# SAME-STAGE FILTER
# =============================================================================

def apply_same_stage_filter(df, stage_labels, continuous=False):
    """Keep only rows where target + both source windows share the same stage."""
    stages_arr = np.array(stage_labels)
    n_before = len(df)

    w_t  = df['window'].values.astype(int)
    w_s1 = w_t - np.round(df['lag1_min'].values * WINDOWS_PER_MIN).astype(int)
    w_s2 = w_t - np.round(df['lag2_min'].values * WINDOWS_PER_MIN).astype(int)

    in_bounds = ((w_s1 >= 0) & (w_s2 >= 0) &
                 (w_s1 < len(stages_arr)) & (w_s2 < len(stages_arr)))
    stage_t  = np.where(in_bounds, stages_arr[np.clip(w_t, 0, len(stages_arr)-1)], '')
    stage_s1 = np.where(in_bounds, stages_arr[np.clip(w_s1, 0, len(stages_arr)-1)], '?')
    stage_s2 = np.where(in_bounds, stages_arr[np.clip(w_s2, 0, len(stages_arr)-1)], '??')
    same_stage = in_bounds & (stage_t == stage_s1) & (stage_t == stage_s2)

    if continuous:
        for i in np.where(same_stage)[0]:
            span = stages_arr[w_s2[i] : w_t[i] + 1]
            if not np.all(span == stage_t[i]):
                same_stage[i] = False

    result = df[same_stage].copy()
    filter_name = 'continuous-bout' if continuous else '3-point'
    print(f"  Same-stage filter ({filter_name}): {n_before} → {len(result)} rows "
          f"({len(result)/max(n_before,1)*100:.1f}% kept)")
    return result


# =============================================================================
# PER-CHANNEL PIPELINE
# =============================================================================

def run_channel(subject_dir, channel, chash):
    """Run full compute pipeline for one electrode. Returns output dir."""
    ch_short = channel.replace("PSG_", "")
    ch_dir = RESULTS_DIR / ch_short
    ch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  CHANNEL: {channel}")
    print(f"  Output : {ch_dir}")
    print(f"{'='*70}")

    # ---- load ---------------------------------------------------------------
    print(f"\n  Loading EEG …")
    signal, fs, stages, ch_name = load_eeg_and_stages(
        subject_dir, channel=channel, duration_hours=DURATION_HOURS)

    # ---- preprocess ---------------------------------------------------------
    print(f"  Preprocessing (bandpass 0.5-60 Hz, notch 50/60 Hz) …")
    signal = preprocess_eeg(signal, fs)

    # ---- common setup -------------------------------------------------------
    W = int(WINDOW_SEC * fs)
    n_windows = len(signal) // W
    good_windows = detect_bad_windows(signal, fs, WINDOW_SEC)
    stage_labels = get_stage_per_window(stages, n_windows, WINDOW_SEC)

    if BANDS is None:
        # ---- broadband mode ------------------------------------------------
        if DISCRETIZE_PER_WINDOW:
            signal_disc = discretize_signal_per_window(signal, fs, WINDOW_SEC, N_BINS)
        else:
            signal_disc = discretize_signal(signal, N_BINS)
        print(f"  {n_windows} windows, {good_windows.sum()} good")

        _run_pipeline(signal, signal_disc, fs, good_windows, stage_labels,
                      n_windows, ch_dir, chash, label="broadband")
    else:
        # ---- band-filtered mode ---------------------------------------------
        print(f"  {n_windows} windows, {good_windows.sum()} good")
        print(f"  Running {len(BANDS)} bands: {list(BANDS.keys())}")
        for band_name, (lo, hi) in BANDS.items():
            print(f"\n  --- Band: {band_name} ({lo}–{hi} Hz) ---")
            sig_band = bandpass_filter(signal, fs, lo, hi)
            if DISCRETIZE_PER_WINDOW:
                disc_band = discretize_signal_per_window(sig_band, fs, WINDOW_SEC, N_BINS)
            else:
                disc_band = discretize_signal(sig_band, N_BINS)
            band_dir = ch_dir / band_name
            band_dir.mkdir(parents=True, exist_ok=True)
            _run_pipeline(sig_band, disc_band, fs, good_windows, stage_labels,
                          n_windows, band_dir, chash, label=band_name)

    print(f"\n  ✓ {channel} complete → {ch_dir}")
    return ch_dir


def _run_pipeline(signal, signal_disc, fs, good_windows, stage_labels,
                  n_windows, ch_dir, chash, label="broadband"):
    """Core compute pipeline — called once for broadband, or once per band."""
    # ---- stage labels -------------------------------------------------------
    stage_csv = ch_dir / f"stage_labels_{chash}.csv"
    if not stage_csv.exists():
        pd.DataFrame({'window': range(n_windows), 'stage': stage_labels}).to_csv(
            stage_csv, index=False)

    # ---- global PID ---------------------------------------------------------
    print(f"\n  [{label}] Global PID matrix …")
    global_csv = ch_dir / f"global_pid_matrix_{chash}.csv"
    if global_csv.exists():
        print("    Cached")
        global_results = pd.read_csv(global_csv)
    else:
        t0 = time.time()
        global_results = compute_global_pid_matrix(signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC)
        print(f"    Done in {time.time()-t0:.1f}s")
        global_results.to_csv(global_csv, index=False)

    # ---- time-resolved PID --------------------------------------------------
    print(f"\n  [{label}] Time-resolved PID …")
    tr_csv = ch_dir / f"timeresolved_pid_{chash}.csv"
    if tr_csv.exists():
        print("    Cached")
        tr_results = pd.read_csv(tr_csv)
    else:
        t0 = time.time()
        tr_results = compute_timeresolved_pid(signal_disc, fs, MAX_LAG_MIN,
                                              WINDOW_SEC, good_windows=good_windows)
        print(f"    Done in {time.time()-t0:.1f}s")
        tr_results.to_csv(tr_csv, index=False)

    # attach stage + same-stage filter
    tr_results['stage'] = tr_results['window'].map(
        lambda w: stage_labels[int(w)] if int(w) < len(stage_labels) else '?')
    tr_results = apply_same_stage_filter(tr_results, stage_labels,
                                         CONTINUOUS_STAGE_FILTER)
    # save filtered version
    tr_filt_csv = ch_dir / f"timeresolved_pid_filtered_{chash}.csv"
    tr_results.to_csv(tr_filt_csv, index=False)

    # ---- AR(1) baseline -----------------------------------------------------
    print(f"\n  [{label}] AR(1) baseline …")
    N_AR = 5
    ar1_mean_csv = ch_dir / f"ar1_global_pid_mean_{chash}.csv"
    ar1_std_csv  = ch_dir / f"ar1_global_pid_std_{chash}.csv"
    if ar1_mean_csv.exists():
        print("    Cached")
        ar1_global_mean = pd.read_csv(ar1_mean_csv)
    else:
        t0 = time.time()
        ar1_global_mean, ar1_global_std = compute_ar1_global_pid(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS, n_realisations=N_AR)
        print(f"    Done in {time.time()-t0:.1f}s")
        ar1_global_mean.to_csv(ar1_mean_csv, index=False)
        ar1_global_std.to_csv(ar1_std_csv, index=False)

    ar1_tr_csv = ch_dir / f"ar1_timeresolved_pid_{chash}.csv"
    if ar1_tr_csv.exists():
        print("    Cached (time-resolved)")
        ar1_tr = pd.read_csv(ar1_tr_csv)
    else:
        ar1_tr = compute_ar1_timeresolved_pid(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS,
            good_windows=good_windows, ar1_global_mean=ar1_global_mean)
        ar1_tr.to_csv(ar1_tr_csv, index=False)

    # same-stage filter for AR(1)
    ar1_tr['stage'] = ar1_tr['window'].map(
        lambda w: stage_labels[int(w)] if int(w) < len(stage_labels) else '?')
    ar1_tr = apply_same_stage_filter(ar1_tr, stage_labels, CONTINUOUS_STAGE_FILTER)
    ar1_filt_csv = ch_dir / f"ar1_timeresolved_pid_filtered_{chash}.csv"
    ar1_tr.to_csv(ar1_filt_csv, index=False)

    # ---- autocorrelation ----------------------------------------------------
    print(f"\n  [{label}] Autocorrelation …")
    autocorr_csv = ch_dir / f"autocorrelation_{chash}.csv"
    if autocorr_csv.exists():
        print("    Cached")
    else:
        t0 = time.time()
        autocorr_df = compute_autocorrelation_per_window(
            signal, fs, MAX_LAG_MIN, WINDOW_SEC)
        print(f"    Done in {time.time()-t0:.1f}s")
        autocorr_df.to_csv(autocorr_csv, index=False)

    # ---- block permutation --------------------------------------------------
    print(f"\n  [{label}] Block-permutation null …")
    N_PERM = 100
    perm_npz = ch_dir / f"block_perm_null_{chash}.npz"
    if perm_npz.exists():
        print("    Cached")
    else:
        t0 = time.time()
        null_vals = compute_block_permutation_null(
            signal_disc, fs, MAX_LAG_MIN, WINDOW_SEC, N_BINS,
            n_permutations=N_PERM)
        print(f"    Done in {time.time()-t0:.1f}s")
        np.savez(perm_npz, null_vals=null_vals)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute PID CSVs across EEG electrodes")
    parser.add_argument('--channels', nargs='+', default=ALL_EEG_CHANNELS,
                        help=f"Channels to process (default: all 6)")
    args = parser.parse_args()

    channels = args.channels
    chash = config_hash()
    save_params(chash)

    print("=" * 70)
    print("EEG SLEEP — MULTI-ELECTRODE PID COMPUTATION")
    print(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Channels : {channels}")
    print(f"Config   : W={WINDOW_SEC}s, lags=1-{MAX_LAG_MIN}min, "
          f"bins={N_BINS}, hash={chash}")
    if BANDS:
        print(f"Bands    : {', '.join(f'{k} ({lo}-{hi} Hz)' for k,(lo,hi) in BANDS.items())}")
    else:
        print(f"Bands    : broadband")
    print("=" * 70)

    # Download data once
    print(f"\nSTEP 1 — DOWNLOAD DATA")
    subject_dir = download_subject(SUBJECT)

    # Process each channel
    for i, ch in enumerate(channels):
        print(f"\n{'#'*70}")
        print(f"  ELECTRODE {i+1}/{len(channels)}: {ch}")
        print(f"{'#'*70}")
        run_channel(subject_dir, ch, chash)

    print(f"\n{'='*70}")
    print(f"ALL DONE — {len(channels)} electrodes processed")
    print(f"Results in {RESULTS_DIR}/")
    for ch in channels:
        ch_short = ch.replace("PSG_", "")
        ch_dir = RESULTS_DIR / ch_short
        n_files = len(list(ch_dir.glob("*")))
        print(f"  {ch_short}/  ({n_files} files)")
    print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
