"""
Per-subject plot driver for the multi-subject 1-s pass.

The main eeg_sleep_plot.py assumes a flat layout under results/pid/eeg_sleep/;
this wrapper iterates PID-10-subjects-1sec/sub-X/ and runs the per-channel
plotting on each subject's nested CSVs, producing the same 16+1 figures per
channel that sub-1 has.

Usage:
    python scripts/pid/eeg_sleep_plot_all_subjects.py
    python scripts/pid/eeg_sleep_plot_all_subjects.py --subjects sub-2 sub-3
    python scripts/pid/eeg_sleep_plot_all_subjects.py --force
"""
import argparse
import sys
from pathlib import Path

# Force UTF-8 stdout.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
BASE = PROJECT_DIR / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
CHASH = "ba79f4ef"

# Patch the main plot script's globals BEFORE importing it.
# eeg_sleep_plot.py auto-discovers params at import. Point its RESULTS_DIR at
# the multi-subject parent so it finds the shared params_<hash>.json.
sys.path.insert(0, str(SCRIPT_DIR))
import eeg_sleep_plot as P

P.RESULTS_DIR = BASE
# Re-run discovery / apply params from the new directory.
P._ALL_CONFIGS = P._discover_all_params()
if P._ALL_CONFIGS:
    P._apply_params(P._ALL_CONFIGS[0][0])


def _per_subject_data(subject: str, ch_short: str):
    """Load all CSVs/NPZ for one (subject, channel)."""
    import pandas as pd
    import numpy as np
    ch_dir = BASE / subject / ch_short
    if not ch_dir.exists():
        return None
    h = CHASH
    label = f"{subject}/{ch_short}"
    data = {"channel": label, "ch_dir": ch_dir,
            "subject": subject, "ch_short": ch_short}

    f = ch_dir / f"global_pid_matrix_{h}.csv"
    data["global"] = pd.read_csv(f) if f.exists() else None
    f = ch_dir / f"timeresolved_pid_filtered_{h}.csv"
    data["tr"] = pd.read_csv(f) if f.exists() else None
    f = ch_dir / f"ar1_global_pid_mean_{h}.csv"
    data["ar1_global"] = pd.read_csv(f) if f.exists() else None
    f = ch_dir / f"ar1_timeresolved_pid_filtered_{h}.csv"
    data["ar1_tr"] = pd.read_csv(f) if f.exists() else None
    f = ch_dir / f"autocorrelation_{h}.csv"
    data["autocorr"] = pd.read_csv(f) if f.exists() else None
    f = ch_dir / f"block_perm_null_{h}.npz"
    data["null_vals"] = np.load(f)["null_vals"] if f.exists() else None
    f = ch_dir / f"stage_labels_{h}.csv"
    data["stage_labels"] = pd.read_csv(f)["stage"].tolist() if f.exists() else None
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    default=[f"sub-{i}" for i in range(1, 11)])
    ap.add_argument("--channels", nargs="+",
                    default=["F3", "F4", "C3", "C4", "O1", "O2"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Per-subject plot driver (hash={CHASH})")
    print(f"Subjects : {args.subjects}")
    print(f"Channels : {args.channels}")
    print(f"Output   : {BASE}/<subject>/<channel>/*.png")
    print("=" * 70)

    n_done = n_skipped = 0
    for subject in args.subjects:
        sub_dir = BASE / subject
        if not sub_dir.exists():
            print(f"  {subject}: directory missing — skipping")
            continue
        for ch in args.channels:
            data = _per_subject_data(subject, ch)
            if data is None or data["tr"] is None:
                print(f"  {subject}/{ch}: no data — skipping")
                n_skipped += 1
                continue
            P.generate_plots_for_channel(data, CHASH, force=args.force)
            n_done += 1

    print(f"\nDone — plotted {n_done} (subject, channel), skipped {n_skipped}")


if __name__ == "__main__":
    main()
