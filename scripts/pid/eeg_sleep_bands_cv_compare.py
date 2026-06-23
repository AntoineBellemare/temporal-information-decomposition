"""
Band-by-band CV-by-timescale comparison.

Loads any (subject, channel, band) CSVs that exist under
  results/pid/eeg_sleep/PID-10-subjects-1sec/<sub>/<ch>/<band>/
and produces a 5-row × 4-col panel:
  rows    = frequency band (delta, theta, alpha, sigma, beta)
  columns = atom (S, R, U_total, S/R)
Each cell overlays the 5 stages.

Y-axis: within-recording CV (mean across available (sub, ch) units per
lag pair) so the figure is directly comparable to the group broadband
CV figure.

Usage:
    python scripts/pid/eeg_sleep_bands_cv_compare.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
BASE = PROJECT_DIR / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
OUT_DIR = BASE / "group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hash for the band-resolved config (different from the broadband hash).
BAND_HASH = "e63716fc"
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
BAND_FREQS = {"delta": "0.5–4 Hz", "theta": "4–8 Hz", "alpha": "8–13 Hz",
              "sigma": "11–16 Hz", "beta": "16–30 Hz"}
CHANNELS = ["C3", "C4", "F3", "F4", "O1", "O2"]
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {"Wake": "#E8A317", "N1": "#87CEEB", "N2": "#4169E1",
                "N3": "#191970", "REM": "#DC143C"}


def discover():
    """Return list of (subject, channel, band) tuples with a filtered CSV on disk."""
    out = []
    if not BASE.exists():
        return out
    for sub_dir in sorted(BASE.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue
        for ch in CHANNELS:
            ch_dir = sub_dir / ch
            if not ch_dir.exists():
                continue
            for band in BANDS:
                f = ch_dir / band / f"timeresolved_pid_filtered_{BAND_HASH}.csv"
                if f.exists():
                    out.append((sub_dir.name, ch, band))
    return out


def _smooth(arr, win=7):
    arr = np.asarray(arr, dtype=float)
    if win < 2 or len(arr) <= win:
        return arr.copy()
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(win) / win, mode="valid")[:len(arr)]


def load_tr(subject, channel, band):
    p = BASE / subject / channel / band / f"timeresolved_pid_filtered_{BAND_HASH}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["unique_total"] = df["unique_0"] + df["unique_1"]
    df["sr_ratio"] = np.where(df["redundancy"] > 0,
                              df["synergy"] / df["redundancy"], np.nan)
    return df


def main():
    inventory = discover()
    if not inventory:
        print("No band-resolved data found yet.")
        return
    by_band = {}
    for sub, ch, band in inventory:
        by_band.setdefault(band, []).append((sub, ch))
    print(f"Inventory:")
    for band in BANDS:
        n = len(by_band.get(band, []))
        print(f"  {band:6s} ({BAND_FREQS[band]:>9s}) : {n} (subject, channel) units")

    metrics = [
        ("synergy",      "Synergy"),
        ("redundancy",   "Redundancy"),
        ("unique_total", "Total Unique"),
        ("sr_ratio",     "S / R Ratio"),
    ]

    fig, axes = plt.subplots(len(BANDS), len(metrics),
                             figsize=(5 * len(metrics), 3.0 * len(BANDS)),
                             constrained_layout=True, sharex=True)

    for bi, band in enumerate(BANDS):
        units = by_band.get(band, [])
        if not units:
            for ci in range(len(metrics)):
                ax = axes[bi, ci]
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                if ci == 0:
                    ax.set_ylabel(f"{band}\n({BAND_FREQS[band]})",
                                  fontsize=11, fontweight="bold")
            continue

        # Concatenate available units for this band.
        rows = []
        for sub, ch in units:
            df = load_tr(sub, ch, band)
            if df is None: continue
            df = df.assign(subject=sub, ch_short=ch)
            rows.append(df)
        big = pd.concat(rows, ignore_index=True)
        big = big[big["stage"].isin(STAGE_ORDER)]

        for ci, (metric, label) in enumerate(metrics):
            ax = axes[bi, ci]
            present = [s for s in STAGE_ORDER if s in big["stage"].unique()]
            for stage in present:
                sub = big[(big["stage"] == stage) & big[metric].notna()]
                if sub.empty:
                    continue
                per_unit = (sub.groupby(["subject", "ch_short",
                                         "lag1_min", "lag2_min"])[metric]
                              .agg(["mean", "std"]).reset_index())
                per_unit["cv"] = np.where(per_unit["mean"] > 0,
                                          per_unit["std"] / per_unit["mean"],
                                          np.nan)
                lag_cv = (per_unit.dropna(subset=["cv"])
                                  .groupby(["lag1_min", "lag2_min"])["cv"]
                                  .mean().reset_index())
                lag_cv["mean_lag"] = (lag_cv["lag1_min"] + lag_cv["lag2_min"]) / 2
                grp = (lag_cv.groupby("mean_lag")["cv"].mean()
                              .reset_index().sort_values("mean_lag"))
                x, y = grp["mean_lag"].values, grp["cv"].values
                ax.plot(x, y, color=STAGE_COLORS[stage],
                        lw=0.9, alpha=0.25, zorder=1)
                ax.plot(x, _smooth(y, win=7),
                        color=STAGE_COLORS[stage], lw=2.2,
                        alpha=0.95, zorder=3,
                        label=stage if (bi == 0 and ci == 0) else None)

            if bi == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{band}\n({BAND_FREQS[band]})",
                              fontsize=11, fontweight="bold")
            if bi == len(BANDS) - 1:
                ax.set_xlabel("Mean lag (min)", fontsize=10)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=9)

    # Subject / channel / unit count annotation
    n_units_total = len(inventory)
    subs = sorted(set(t[0] for t in inventory))
    fig.suptitle(
        f"Within-recording CV by timescale, per band\n"
        f"({len(subs)} subject(s): {', '.join(subs)} ·  "
        f"{n_units_total} (channel × band) units total)",
        fontsize=13, fontweight="bold")

    # Single legend at the bottom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=len(labels), frameon=False, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))

    out = OUT_DIR / f"bands_cv_by_timescale_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
