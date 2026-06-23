"""
Band-resolved exploration of PID stage signatures.

Each figure answers ONE question. Reruns are idempotent — pick up new
(subject, channel, band) units as they arrive on disk.

Figures (all under PID-10-subjects-1sec/group/):

  bands_spectral_fingerprint_<hash>.png
      One panel per atom; x = frequency band; lines = sleep stages.
      Answers: which band carries each stage's PID signature?

  bands_n3_vs_wake_<hash>.png
      Per-band Cohen-d heatmap of stage contrasts (rows = atoms,
      cols = stage pairs). Answers: where is each contrast detectable?

  bands_ranking_consistency_<hash>.png
      Per-band rank-order of stages on each atom. Answers: does the
      broadband ordering survive band-restriction, or is broadband a
      mixture artefact?

  bands_total_mi_spectrum_<hash>.png
      For each stage, total predictive MI vs band. The "PID power
      spectrum" of each stage.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

BAND_HASH = "e63716fc"
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
BAND_FREQS = {"delta": "0.5–4 Hz", "theta": "4–8 Hz", "alpha": "8–13 Hz",
              "sigma": "11–16 Hz", "beta": "16–30 Hz"}
BAND_CENTRES = {"delta": 2.25, "theta": 6, "alpha": 10.5, "sigma": 13.5, "beta": 23}
CHANNELS = ["C3", "C4", "F3", "F4", "O1", "O2"]
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {"Wake": "#E8A317", "N1": "#87CEEB", "N2": "#4169E1",
                "N3": "#191970", "REM": "#DC143C"}
ATOMS = ["redundancy", "synergy", "unique_0", "unique_1"]
ATOM_LABELS = ["Redundancy", "Synergy", "Unique₁", "Unique₂"]


def discover():
    if not BASE.exists():
        return
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
                    yield sub_dir.name, ch, band


def load_tr(sub, ch, band):
    p = BASE / sub / ch / band / f"timeresolved_pid_filtered_{BAND_HASH}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["unique_total"] = df["unique_0"] + df["unique_1"]
    df["total_mi"] = (df["redundancy"] + df["synergy"]
                       + df["unique_0"] + df["unique_1"])
    df["sr_ratio"] = np.where(df["redundancy"] > 0,
                              df["synergy"] / df["redundancy"], np.nan)
    return df


def collect_per_unit_per_stage(inventory, metrics):
    """Return long-form: one row per (subject, channel, band, stage, metric_dict).

    Each value is the mean of `metric` across all windows × lag pairs for
    that stage in that recording.
    """
    rows = []
    for sub, ch, band in inventory:
        df = load_tr(sub, ch, band)
        if df is None:
            continue
        df = df[df["stage"].isin(STAGE_ORDER)]
        agg = df.groupby("stage")[metrics].mean().reset_index()
        agg = agg.assign(subject=sub, ch_short=ch, band=band)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else None


# ============================================================================
# Figure 1: spectral fingerprint — per atom, x=band, lines=stage
# ============================================================================
def plot_spectral_fingerprint(inventory):
    metrics_ext = ATOMS + ["unique_total", "sr_ratio", "total_mi"]
    long = collect_per_unit_per_stage(inventory, metrics_ext)
    if long is None:
        return

    panels = [("synergy",      "Synergy (bits)"),
              ("redundancy",   "Redundancy (bits)"),
              ("unique_total", "Total Unique (bits)"),
              ("sr_ratio",     "S / R Ratio")]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(5.2 * len(panels), 5),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    x_band = np.array([BAND_CENTRES[b] for b in BANDS])

    for ax, (metric, label) in zip(axes, panels):
        for stage in STAGE_ORDER:
            ys, lo, hi = [], [], []
            for band in BANDS:
                vals = long.loc[(long["stage"] == stage)
                                & (long["band"] == band), metric].dropna().values
                if len(vals):
                    m = np.mean(vals)
                    s = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                    ys.append(m); lo.append(m - s); hi.append(m + s)
                else:
                    ys.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            ys = np.array(ys, dtype=float)
            mask = ~np.isnan(ys)
            if not mask.any():
                continue
            ax.plot(x_band[mask], ys[mask], "o-", color=STAGE_COLORS[stage],
                    lw=2.2, ms=8, label=stage)
            ax.fill_between(x_band[mask],
                            np.array(lo)[mask], np.array(hi)[mask],
                            color=STAGE_COLORS[stage], alpha=0.15)
        ax.set_xticks(x_band)
        ax.set_xticklabels([f"{b}\n{BAND_FREQS[b]}" for b in BANDS],
                            fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label.split("(")[0].strip(), fontweight="bold", fontsize=12)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"Spectral fingerprint of PID atoms across sleep stages\n"
        f"({n_subs} subject(s), available channels)",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / f"bands_spectral_fingerprint_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  spectral_fingerprint → {out}")


# ============================================================================
# Figure 2: Cohen's d for stage contrasts per band
# ============================================================================
def plot_stage_contrasts(inventory):
    long = collect_per_unit_per_stage(inventory,
                                       ATOMS + ["unique_total", "sr_ratio"])
    if long is None:
        return
    pairs = [("N3", "Wake"), ("N3", "N1"), ("N2", "Wake"),
             ("REM", "Wake"), ("REM", "N2"), ("REM", "N3")]
    metrics = [("synergy",      "Synergy"),
               ("redundancy",   "Redundancy"),
               ("unique_total", "Total Unique"),
               ("sr_ratio",     "S / R Ratio")]

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(2 + 1.4 * len(pairs), 2.4 * len(metrics)),
                             constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, metrics):
        d_mat = np.full((len(BANDS), len(pairs)), np.nan)
        for bi, band in enumerate(BANDS):
            sub = long[long["band"] == band]
            for pi, (sa, sb) in enumerate(pairs):
                va = sub.loc[sub["stage"] == sa, metric].dropna().values
                vb = sub.loc[sub["stage"] == sb, metric].dropna().values
                if len(va) < 2 or len(vb) < 2:
                    continue
                pooled = np.sqrt(((len(va) - 1) * np.var(va, ddof=1)
                                  + (len(vb) - 1) * np.var(vb, ddof=1))
                                  / (len(va) + len(vb) - 2))
                if pooled > 1e-12:
                    d_mat[bi, pi] = (np.mean(va) - np.mean(vb)) / pooled
        vmax = max(0.5, np.nanmax(np.abs(d_mat))) if np.isfinite(np.nanmax(d_mat)) else 1
        sns.heatmap(d_mat, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    xticklabels=[f"{a}\n−\n{b}" for a, b in pairs],
                    yticklabels=[f"{b}\n{BAND_FREQS[b]}" for b in BANDS],
                    annot=True, fmt=".2f", annot_kws={"fontsize": 9},
                    cbar_kws={"label": "Cohen's d", "shrink": 0.8},
                    mask=np.isnan(d_mat))
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.tick_params(labelsize=9)

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"Stage-contrast effect sizes per band\n"
        f"(Cohen's d across available (subject × channel) units, "
        f"{n_subs} subject(s))",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / f"bands_stage_contrasts_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  stage_contrasts → {out}")


# ============================================================================
# Figure 3: ranking consistency — does each band agree on the stage order?
# ============================================================================
def plot_ranking_consistency(inventory):
    long = collect_per_unit_per_stage(inventory, ATOMS + ["sr_ratio"])
    if long is None:
        return

    panels = [("synergy",     "Synergy"),
              ("redundancy",  "Redundancy"),
              ("unique_0",    "Unique₁ (shorter lag)"),
              ("unique_1",    "Unique₂ (longer lag)"),
              ("sr_ratio",    "S / R Ratio")]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.5 * len(panels), 4.8),
                             constrained_layout=True)

    for ax, (metric, label) in zip(axes, panels):
        # For each band: rank stages by mean value (1 = highest, 5 = lowest)
        present_stages = [s for s in STAGE_ORDER if s in long["stage"].unique()]
        rank_mat = np.full((len(BANDS), len(present_stages)), np.nan)
        for bi, band in enumerate(BANDS):
            sub = long[long["band"] == band]
            means = sub.groupby("stage")[metric].mean()
            # rank descending (so 1 = highest mean)
            ranks = means.rank(ascending=False)
            for si, stage in enumerate(present_stages):
                if stage in ranks.index:
                    rank_mat[bi, si] = ranks[stage]

        # plot as heatmap
        n_st = len(present_stages)
        sns.heatmap(rank_mat, ax=ax, cmap="RdYlGn_r",
                    vmin=1, vmax=n_st, annot=True, fmt=".0f",
                    annot_kws={"fontsize": 11, "fontweight": "bold"},
                    xticklabels=present_stages,
                    yticklabels=[f"{b}" for b in BANDS],
                    cbar_kws={"label": "rank (1 = highest)", "shrink": 0.7},
                    mask=np.isnan(rank_mat), linewidths=0.5,
                    linecolor="white")
        ax.set_title(label, fontweight="bold", fontsize=11)
        ax.tick_params(labelsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Band", fontsize=10)

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"Stage ranking per band — is the broadband order preserved everywhere?\n"
        f"(1 = highest, {len(STAGE_ORDER)} = lowest; "
        f"if rows agree, the order is band-invariant)",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / f"bands_ranking_consistency_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ranking_consistency → {out}")


# ============================================================================
# Figure 4: total-MI spectrum per stage
# ============================================================================
def plot_total_mi_spectrum(inventory):
    long = collect_per_unit_per_stage(inventory, ["total_mi"])
    if long is None:
        return

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x_band = np.array([BAND_CENTRES[b] for b in BANDS])
    for stage in STAGE_ORDER:
        ys, sems = [], []
        for band in BANDS:
            vals = long.loc[(long["stage"] == stage)
                            & (long["band"] == band), "total_mi"].dropna().values
            if len(vals):
                ys.append(np.mean(vals))
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                            if len(vals) > 1 else 0)
            else:
                ys.append(np.nan); sems.append(np.nan)
        ys = np.asarray(ys); sems = np.asarray(sems)
        mask = ~np.isnan(ys)
        if not mask.any():
            continue
        ax.plot(x_band[mask], ys[mask], "o-",
                 color=STAGE_COLORS[stage], lw=2.4, ms=9, label=stage)
        ax.fill_between(x_band[mask], ys[mask] - sems[mask],
                         ys[mask] + sems[mask],
                         color=STAGE_COLORS[stage], alpha=0.15)
    ax.set_xticks(x_band)
    ax.set_xticklabels([f"{b}\n{BAND_FREQS[b]}" for b in BANDS], fontsize=10)
    ax.set_xlabel("Frequency band (Hz)", fontsize=11)
    ax.set_ylabel("Total predictive MI (bits) — R + S + U₁ + U₂", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=False)
    n_subs = len(set(s for s, _, _ in inventory))
    ax.set_title(
        f"Predictive-information spectrum per sleep stage  "
        f"({n_subs} subject(s))",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / f"bands_total_mi_spectrum_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  total_mi_spectrum → {out}")


def main():
    inv = list(discover())
    by_band = {}
    for s, c, b in inv:
        by_band.setdefault(b, []).append((s, c))
    print(f"Inventory ({len(inv)} units):")
    for band in BANDS:
        n = len(by_band.get(band, []))
        print(f"  {band:6s} ({BAND_FREQS[band]:>9s}) : {n}")
    if not inv:
        return
    plot_spectral_fingerprint(inv)
    plot_stage_contrasts(inv)
    plot_ranking_consistency(inv)
    plot_total_mi_spectrum(inv)
    print("\nDone.")


if __name__ == "__main__":
    main()
