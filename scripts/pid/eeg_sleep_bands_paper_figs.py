"""
Ten band-resolved paper-ready figures for the 1-s sleep PID pass.

Each function is independent and produces ONE figure. With sub-1 data only
the n-per-stage is small; the figures still render correctly and will
acquire proper statistics as sub-2 and sub-3 come online.

Layout convention:
    Per-band rows usually = δ, θ, α, σ, β (5)
    Per-atom cols usually = R, S, U₁, U₂ (4)

Outputs (all under PID-10-subjects-1sec/group/):
    f01_band_stage_diff_n3vswake.png      — fig 1
    f02_discriminability_scoreboard.png   — fig 2
    f03_pid_phase_portrait.png            — fig 3
    f04_band_excess_ar1.png               — fig 4
    f05_cross_band_correlation.png        — fig 5
    f06_time_resolved_sr_spectrogram.png  — fig 6
    f07_optimal_timescale_fingerprint.png — fig 7
    f08_nrem_rem_transitions.png          — fig 8
    f09_band_stage_bootstrap.png          — fig 9
    f10_alpha_vs_delta_compass.png        — fig 10
"""
import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu

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
BAND_CENTRES = {"delta": 2.25, "theta": 6, "alpha": 10.5,
                "sigma": 13.5, "beta": 23}
CHANNELS = ["C3", "C4", "F3", "F4", "O1", "O2"]
STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {"Wake": "#E8A317", "N1": "#87CEEB", "N2": "#4169E1",
                "N3": "#191970", "REM": "#DC143C"}
ATOMS = ["redundancy", "synergy", "unique_0", "unique_1"]
ATOM_LABELS = ["Redundancy", "Synergy", "Unique₁", "Unique₂"]
ATOM_CMAPS = ["Greens", "Reds", "Blues", "Purples"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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


def load_global(sub, ch, band):
    p = BASE / sub / ch / band / f"global_pid_matrix_{BAND_HASH}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_ar1(sub, ch, band):
    p = BASE / sub / ch / band / f"ar1_global_pid_mean_{BAND_HASH}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_stages(sub, ch, band):
    """Return list of stage labels per window for one (sub, ch, band)."""
    p = BASE / sub / ch / band / f"stage_labels_{BAND_HASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)["stage"].tolist()


def bootstrap_ci(arr, n_boot=1000, alpha=0.05, rng=None):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(42)
    boots = np.array([np.mean(rng.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    return (np.mean(arr),
            np.percentile(boots, 100 * alpha / 2),
            np.percentile(boots, 100 * (1 - alpha / 2)))


def collect_per_unit_per_stage(inventory, metrics):
    rows = []
    for sub, ch, band in inventory:
        df = load_tr(sub, ch, band)
        if df is None:
            continue
        df = df[df["stage"].isin(STAGE_ORDER)]
        agg = df.groupby("stage")[metrics].mean().reset_index()
        rows.append(agg.assign(subject=sub, ch_short=ch, band=band))
    return pd.concat(rows, ignore_index=True) if rows else None


# ===========================================================================
# FIGURE 1: N3 vs Wake τ₁ × τ₂ difference maps, per band per atom
# ===========================================================================
def fig01_band_stage_diff(inventory, contrast=("N3", "Wake"), save_name=None):
    """5 atoms × 5 bands grid of (N3 − Wake) τ1×τ2 difference heatmaps."""
    sa, sb = contrast
    # Per (sub, ch, band, stage) get the mean τ1×τ2 matrix from the
    # time-resolved CSV (averaged across windows of that stage).
    band_stage_mats = {b: {} for b in BANDS}
    for sub, ch, band in inventory:
        df = load_tr(sub, ch, band)
        if df is None:
            continue
        for stg in (sa, sb):
            sub_stg = df[df["stage"] == stg]
            if sub_stg.empty:
                continue
            for atom in ATOMS:
                key = (stg, atom)
                grp = (sub_stg.groupby(["lag1_min", "lag2_min"])[atom]
                              .mean().reset_index())
                if key not in band_stage_mats[band]:
                    band_stage_mats[band][key] = []
                band_stage_mats[band][key].append(grp)

    fig, axes = plt.subplots(len(ATOMS), len(BANDS),
                             figsize=(3.6 * len(BANDS), 3.5 * len(ATOMS)),
                             constrained_layout=True)
    for ai, (atom, lab) in enumerate(zip(ATOMS, ATOM_LABELS)):
        # Determine shared vmax across bands for this atom
        all_diffs = []
        per_band_diffs = {}
        for band in BANDS:
            if (sa, atom) not in band_stage_mats[band] or (sb, atom) not in band_stage_mats[band]:
                continue
            ma = pd.concat(band_stage_mats[band][(sa, atom)]).groupby(
                ["lag1_min", "lag2_min"])[atom].mean().reset_index()
            mb = pd.concat(band_stage_mats[band][(sb, atom)]).groupby(
                ["lag1_min", "lag2_min"])[atom].mean().reset_index()
            mrg = ma.merge(mb, on=["lag1_min", "lag2_min"],
                           suffixes=("_a", "_b"))
            mrg["diff"] = mrg[f"{atom}_a"] - mrg[f"{atom}_b"]
            per_band_diffs[band] = mrg
            all_diffs.append(mrg["diff"].values)
        if all_diffs:
            vmax = np.nanpercentile(np.abs(np.concatenate(all_diffs)), 98)
            vmax = max(vmax, 1e-9)
        else:
            vmax = 1.0
        for bi, band in enumerate(BANDS):
            ax = axes[ai, bi]
            if band not in per_band_diffs:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                if ai == 0:
                    ax.set_title(f"{band}\n({BAND_FREQS[band]})",
                                 fontsize=11, fontweight="bold")
                if bi == 0:
                    ax.set_ylabel(lab, fontsize=11, fontweight="bold")
                continue
            mrg = per_band_diffs[band]
            lag_vals = sorted(set(mrg["lag1_min"]) | set(mrg["lag2_min"]))
            l_idx = {v: i for i, v in enumerate(lag_vals)}
            n = len(lag_vals)
            mat = np.full((n, n), np.nan)
            for _, r in mrg.iterrows():
                i1 = l_idx.get(r["lag1_min"]); i2 = l_idx.get(r["lag2_min"])
                if i1 is not None and i2 is not None:
                    mat[i1, i2] = r["diff"]
            tick_labels = [t if i % max(1, n // 6) == 0 else ""
                           for i, t in enumerate([f"{v:g}" for v in lag_vals])]
            sns.heatmap(mat, ax=ax, cmap="RdBu_r", center=0,
                        vmin=-vmax, vmax=vmax,
                        mask=np.isnan(mat),
                        xticklabels=tick_labels, yticklabels=tick_labels,
                        cbar_kws={"label": "Δ (bits)" if bi == len(BANDS) - 1 else ""})
            if ai == 0:
                ax.set_title(f"{band}\n({BAND_FREQS[band]})",
                             fontsize=11, fontweight="bold")
            if bi == 0:
                ax.set_ylabel(f"{lab}\nLag 1 (min)", fontsize=10,
                              fontweight="bold")
            if ai == len(ATOMS) - 1:
                ax.set_xlabel("Lag 2 (min)", fontsize=10)
            ax.tick_params(labelsize=8)

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"τ₁ × τ₂ difference maps:  {sa} − {sb},  per band per atom\n"
        f"({n_subs} subject(s); positive = {sa} elevated)",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / (save_name or
                     f"f01_band_stage_diff_{sa.lower()}vs{sb.lower()}_{BAND_HASH}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 1 ({sa} vs {sb}) → {out}")


# ===========================================================================
# FIGURE 2: Stage-discriminability scoreboard (KW H)
# ===========================================================================
def fig02_discriminability_scoreboard(inventory):
    metrics = [("synergy", "Synergy"), ("redundancy", "Redundancy"),
               ("unique_0", "Unique₁"), ("unique_1", "Unique₂"),
               ("sr_ratio", "S / R Ratio"), ("total_mi", "Total MI")]
    long = collect_per_unit_per_stage(inventory,
                                       ATOMS + ["sr_ratio", "total_mi"])
    if long is None: return

    H_mat = np.full((len(metrics), len(BANDS)), np.nan)
    p_mat = np.full((len(metrics), len(BANDS)), np.nan)
    for mi, (metric, _) in enumerate(metrics):
        for bi, band in enumerate(BANDS):
            sub = long[long["band"] == band]
            groups = [sub.loc[sub["stage"] == s, metric].dropna().values
                      for s in STAGE_ORDER
                      if len(sub.loc[sub["stage"] == s, metric].dropna()) >= 2]
            if len(groups) < 2: continue
            try:
                H, p = kruskal(*groups)
                H_mat[mi, bi] = H
                p_mat[mi, bi] = p
            except Exception:
                pass

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
    sns.heatmap(H_mat, ax=ax, cmap="viridis", annot=True, fmt=".1f",
                annot_kws={"fontsize": 11, "fontweight": "bold"},
                xticklabels=[f"{b}\n({BAND_FREQS[b]})" for b in BANDS],
                yticklabels=[m for _, m in metrics],
                cbar_kws={"label": "Kruskal–Wallis H statistic"},
                linewidths=0.5, linecolor="white",
                mask=np.isnan(H_mat))
    ax.set_title("Stage-discriminability scoreboard\n"
                 "(higher H = stronger stage separation)",
                 fontweight="bold", fontsize=12)
    ax.tick_params(labelsize=10)
    out = OUT_DIR / f"f02_discriminability_scoreboard_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 2 → {out}")


# ===========================================================================
# FIGURE 3: PID phase portrait (S vs R) per band — redesigned
# ===========================================================================
def fig03_phase_portrait(inventory):
    """Per-(sub, ch, stage) means (one dot per recording per stage) with
    per-stage 95% covariance ellipses. Avoids the per-window blob."""
    fig, axes = plt.subplots(1, len(BANDS),
                             figsize=(3.6 * len(BANDS), 4.0),
                             constrained_layout=True)
    for bi, band in enumerate(BANDS):
        ax = axes[bi]
        units = [(s, c) for (s, c, b) in inventory if b == band]
        if not units:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="#888")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{band}  ({BAND_FREQS[band]})",
                         fontsize=11, fontweight="bold")
            continue
        rows = []
        for sub, ch in units:
            df = load_tr(sub, ch, band)
            if df is None:
                continue
            df = df[df["stage"].isin(STAGE_ORDER)]
            agg = df.groupby("stage")[["synergy", "redundancy"]].mean().reset_index()
            agg = agg.assign(subject=sub, ch_short=ch)
            rows.append(agg)
        big = pd.concat(rows, ignore_index=True)

        # Per stage: scatter unit dots + covariance ellipse
        for stage in STAGE_ORDER:
            sub = big[big["stage"] == stage]
            if sub.empty:
                continue
            xs = sub["synergy"].values; ys = sub["redundancy"].values
            ax.scatter(xs, ys, c=STAGE_COLORS[stage], s=70,
                       alpha=0.75, edgecolors="black", linewidths=0.8,
                       zorder=3, label=stage)
            # ellipse only if we have ≥3 points
            if len(xs) >= 3:
                cov = np.cov(xs, ys)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]; vecs = vecs[:, order]
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                # 95% confidence ellipse: scale = sqrt(chi2_2dof at 0.95) ≈ 2.448
                w, h = 2 * 2.448 * np.sqrt(vals)
                ell = Ellipse((xs.mean(), ys.mean()), width=w, height=h,
                              angle=angle, facecolor=STAGE_COLORS[stage],
                              alpha=0.13, edgecolor=STAGE_COLORS[stage],
                              lw=1.5, zorder=2)
                ax.add_patch(ell)
            # stage centroid marker
            ax.scatter(xs.mean(), ys.mean(), c=STAGE_COLORS[stage],
                       s=320, alpha=1.0, marker="X", edgecolors="black",
                       linewidths=1.5, zorder=5)

        # axis range from data with a small pad
        x_lo, x_hi = big["synergy"].min(), big["synergy"].max()
        y_lo, y_hi = big["redundancy"].min(), big["redundancy"].max()
        x_pad = 0.10 * (x_hi - x_lo); y_pad = 0.10 * (y_hi - y_lo)
        ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        ax.set_xlabel("Synergy  (bits)", fontsize=10)
        if bi == 0:
            ax.set_ylabel("Redundancy  (bits)", fontsize=10)
            ax.legend(loc="best", fontsize=8, frameon=False)
        ax.set_title(f"{band}  ({BAND_FREQS[band]})",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"PID phase portrait — Synergy vs Redundancy, per band\n"
        f"({n_subs} subject(s); dots = per-(subject, channel) stage mean; "
        f"X = stage mean; ellipse = 95% covariance)",
        fontsize=12, fontweight="bold")
    out = OUT_DIR / f"f03_phase_portrait_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 3 → {out}")


# ===========================================================================
# FIGURE 4: per-band excess vs stage-conditional AR(1)
# ===========================================================================
def fig04_band_excess_ar1(inventory):
    """Per band: mean (Actual − stage-AR(1)) per stage per atom, bars."""
    rows = []
    for sub, ch, band in inventory:
        ar1 = load_ar1(sub, ch, band)
        if ar1 is None or "stage" not in ar1.columns:
            continue
        df = load_tr(sub, ch, band)
        if df is None:
            continue
        # Compute per (stage, lag pair) means of actual; subtract AR(1) per stage.
        actual = (df.groupby(["stage", "lag1_min", "lag2_min"])[ATOMS]
                    .mean().reset_index())
        ar1_long = (ar1.groupby(["stage", "lag1_min", "lag2_min"])[ATOMS]
                       .mean().reset_index())
        m = actual.merge(ar1_long, on=["stage", "lag1_min", "lag2_min"],
                         suffixes=("_a", "_ar1"))
        # Excess per (stage, lag pair) per atom
        excess = {}
        for atom in ATOMS:
            m[f"{atom}_ex"] = m[f"{atom}_a"] - m[f"{atom}_ar1"]
        # Average across lag pairs per stage
        per_stage = m.groupby("stage")[[f"{a}_ex" for a in ATOMS]].mean()
        for stage, row in per_stage.iterrows():
            for a in ATOMS:
                rows.append({"subject": sub, "ch_short": ch, "band": band,
                             "stage": stage, "atom": a, "excess": row[f"{a}_ex"]})
    if not rows:
        return
    long = pd.DataFrame(rows)

    fig, axes = plt.subplots(len(ATOMS), len(BANDS),
                             figsize=(3.6 * len(BANDS), 2.8 * len(ATOMS)),
                             constrained_layout=True, sharey="row")
    for ai, (atom, lab) in enumerate(zip(ATOMS, ATOM_LABELS)):
        for bi, band in enumerate(BANDS):
            ax = axes[ai, bi]
            sub = long[(long["band"] == band) & (long["atom"] == atom)]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                if ai == 0: ax.set_title(band, fontsize=11, fontweight="bold")
                if bi == 0: ax.set_ylabel(lab, fontsize=11, fontweight="bold")
                continue
            means = [sub.loc[sub["stage"] == s, "excess"].mean() for s in STAGE_ORDER]
            sems  = [sub.loc[sub["stage"] == s, "excess"].sem() for s in STAGE_ORDER]
            x = np.arange(len(STAGE_ORDER))
            colors = [STAGE_COLORS[s] for s in STAGE_ORDER]
            ax.bar(x, means, color=colors, edgecolor="white",
                   alpha=0.9, linewidth=1.2)
            ax.errorbar(x, means, yerr=sems, fmt="none",
                        ecolor="#222", capsize=3, lw=1)
            ax.axhline(0, color="grey", lw=0.5, ls="--")
            ax.set_xticks(x); ax.set_xticklabels(STAGE_ORDER, fontsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(axis="y", alpha=0.3)
            if ai == 0:
                ax.set_title(f"{band}\n({BAND_FREQS[band]})",
                             fontsize=10, fontweight="bold")
            if bi == 0:
                ax.set_ylabel(f"{lab}\nexcess (bits)",
                              fontsize=10, fontweight="bold")

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(f"Per-stage excess vs stage-conditional AR(1), per band\n"
                 f"({n_subs} subject(s); positive = above stage's own AR(1))",
                 fontsize=13, fontweight="bold")
    out = OUT_DIR / f"f04_band_excess_ar1_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 4 → {out}")


# ===========================================================================
# FIGURE 5: Cross-band atom correlation — redesigned
# ===========================================================================
def fig05_cross_band_correlation(inventory):
    """Across-(sub, ch) per-stage atom correlation, separated into a
    summary panel + one heatmap collapsed across stages.

    For each band pair we compute the across-window correlation of the
    per-window mean atom value, separately per stage, then average
    across (subject, channel) units. The original 5-stage 5×5 grid
    was almost-empty because per-window S/R between bands within a
    single stage is essentially uncorrelated.  Repackage as:

      Left: mean off-diagonal correlation per stage, per atom (bars).
      Right: collapsed 5×5 (band × band) correlation across all (sub, ch,
             window, stage) — the global cross-band matrix.
    """
    units = {}
    for sub, ch, band in inventory:
        units.setdefault((sub, ch), set()).add(band)
    full = [k for k, bs in units.items() if set(BANDS).issubset(bs)]
    if not full:
        print("  fig 5: no (sub, ch) has all 5 bands yet")
        return

    target_metrics = [("synergy", "Synergy"),
                      ("redundancy", "Redundancy"),
                      ("sr_ratio", "S / R Ratio")]

    # global big merged table across all units, one row per (sub, ch, window).
    big_rows = []
    for sub, ch in full:
        wm_by_band = {}
        for band in BANDS:
            df = load_tr(sub, ch, band)
            if df is None: continue
            wm = (df.groupby(["window", "stage"])[
                ["synergy", "redundancy", "sr_ratio"]].mean().reset_index())
            wm = wm.rename(columns={"synergy": f"S_{band}",
                                     "redundancy": f"R_{band}",
                                     "sr_ratio": f"SR_{band}"})
            wm_by_band[band] = wm
        if len(wm_by_band) < len(BANDS):
            continue
        merged = None
        for band, wm in wm_by_band.items():
            if merged is None:
                merged = wm
            else:
                # keep stage from first
                merged = merged.merge(wm.drop(columns="stage"),
                                       on="window", how="inner")
        if merged is None or merged.empty:
            continue
        merged["subject"] = sub; merged["ch_short"] = ch
        big_rows.append(merged)
    if not big_rows:
        return
    big = pd.concat(big_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    cols_sr = [f"SR_{b}" for b in BANDS]
    corr = big[cols_sr].corr().values
    corr_masked = corr.copy()
    np.fill_diagonal(corr_masked, np.nan)
    abs_max = np.nanmax(np.abs(corr_masked)) if np.isfinite(np.nanmax(corr_masked)) else 0.5
    vmax = max(round(abs_max * 1.2, 1), 0.2)
    sns.heatmap(corr_masked, ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                annot=True, fmt=".2f",
                annot_kws={"fontsize": 18, "fontweight": "bold"},
                xticklabels=BANDS, yticklabels=BANDS,
                cbar_kws={"label": "Pearson r", "shrink": 0.7},
                linewidths=1, linecolor="white",
                mask=np.isnan(corr_masked))
    ax.set_title("Cross-band S/R correlation  (diagonal masked)\n"
                 "values near zero → bands carry independent information",
                 fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=13)

    n_subs = len(set(s for s, _ in full))
    fig.suptitle(
        f"Cross-band independence — {n_subs} subject(s) × {len(full)} "
        f"(subject, channel) units",
        fontsize=12, fontweight="bold")
    out = OUT_DIR / f"f05_cross_band_correlation_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 5 → {out}")


# ===========================================================================
# FIGURE 6: Time-resolved S/R spectrogram (sub-1 C3, all bands)
# ===========================================================================
def fig06_sr_spectrogram(inventory, target_sub="sub-1", target_ch="C3"):
    """For one canonical channel: x=time, y=band, color=mean S/R per window."""
    bands_avail = sorted(
        {b for (s, c, b) in inventory if s == target_sub and c == target_ch})
    if not bands_avail:
        print(f"  fig 6: no data for {target_sub} {target_ch}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 4),
                             gridspec_kw={"height_ratios": [0.4, 1.2]},
                             sharex=True, constrained_layout=True)

    stage_to_y = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0, "?": -1}
    stages_band = None
    band_traces = {}
    for band in bands_avail:
        df = load_tr(target_sub, target_ch, band)
        if df is None: continue
        wm = df.groupby(["window", "time_min", "stage"])["sr_ratio"].mean()
        wm = wm.reset_index().sort_values("time_min")
        band_traces[band] = wm
        if stages_band is None:
            stages_band = wm

    # Hypnogram from stages_band
    ax_hyp = axes[0]
    if stages_band is not None:
        for _, r in stages_band.iterrows():
            y = stage_to_y.get(r["stage"], -1)
            t = r["time_min"]
            ax_hyp.add_patch(plt.Rectangle((t, y - 0.4),
                                            (3.0 / 60), 0.8,
                                            facecolor=STAGE_COLORS.get(
                                                r["stage"], "#ccc"),
                                            edgecolor="none"))
    ax_hyp.set_yticks([0, 1, 2, 3, 4])
    ax_hyp.set_yticklabels(["N3", "N2", "N1", "REM", "Wake"], fontsize=9)
    ax_hyp.set_ylim(-0.5, 4.5)
    ax_hyp.set_title(f"Hypnogram — {target_sub} {target_ch}",
                     fontsize=10)
    ax_hyp.set_xlabel("")
    ax_hyp.set_xlim(0, stages_band["time_min"].max())
    ax_hyp.grid(False)
    for spine in ax_hyp.spines.values():
        spine.set_visible(False)

    # Spectrogram-style heatmap
    ax = axes[1]
    full_bands = BANDS  # canonical order
    times = sorted(stages_band["time_min"].unique())
    sr_mat = np.full((len(full_bands), len(times)), np.nan)
    t_idx = {t: i for i, t in enumerate(times)}
    for bi, band in enumerate(full_bands):
        if band not in band_traces: continue
        wm = band_traces[band].set_index("time_min")
        for t, row in wm.iterrows():
            if t in t_idx:
                sr_mat[bi, t_idx[t]] = row["sr_ratio"]
    vmin = np.nanpercentile(sr_mat, 5) if np.isfinite(np.nanmin(sr_mat)) else 0
    vmax = np.nanpercentile(sr_mat, 95) if np.isfinite(np.nanmax(sr_mat)) else 5
    # Lightly smooth each band's time series so the visualisation isn't
    # dominated by single-window jitter — preserves stage-scale features.
    smoothed = sr_mat.copy()
    for bi in range(smoothed.shape[0]):
        row = smoothed[bi]
        good = ~np.isnan(row)
        if good.sum() > 7:
            pad = 3
            r = np.pad(row[good], pad, mode="edge")
            sm = np.convolve(r, np.ones(7) / 7, mode="valid")[:good.sum()]
            tmp = row.copy(); tmp[good] = sm
            smoothed[bi] = tmp
    im = ax.imshow(smoothed, aspect="auto", origin="lower",
                   extent=[times[0], times[-1], -0.5, len(full_bands) - 0.5],
                   cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(full_bands)))
    ax.set_yticklabels([f"{b}\n({BAND_FREQS[b]})" for b in full_bands],
                        fontsize=9)
    ax.set_xlabel("Time (min)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("S / R ratio", fontsize=10)
    ax.set_title(f"S/R ratio per band over time — {target_sub} {target_ch}",
                 fontsize=11, fontweight="bold")
    out = OUT_DIR / f"f06_sr_spectrogram_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 6 → {out}")


# ===========================================================================
# FIGURE 7: Lag-resolved (band × stage) atom maps — redesigned
# ===========================================================================
def fig07_lag_profile(inventory):
    """One panel per atom; in each panel a heatmap with rows = (band, stage)
    and cols = mean lag, colour = z-scored atom value across rows so the
    timescale structure is comparable across rows of very different scale.

    This collapses what used to be 20 line panels into 4 compact heatmaps
    where stage-vs-band differences in *where on the lag axis* the atom
    peaks are read off at a glance.
    """
    # gather mean atom value per (band, stage, mean_lag) averaged over units
    fig, axes = plt.subplots(1, len(ATOMS),
                             figsize=(4.6 * len(ATOMS), 6.2),
                             constrained_layout=True)
    row_labels = []
    row_band = []  # for separators
    for band in BANDS:
        for stage in STAGE_ORDER:
            row_labels.append(f"{band[:3]}·{stage}")
            row_band.append(band)

    for ai, (atom, lab) in enumerate(zip(ATOMS, ATOM_LABELS)):
        ax = axes[ai]
        rows = []
        lag_vals = None
        for band in BANDS:
            units = [(s, c) for (s, c, b) in inventory if b == band]
            for stage in STAGE_ORDER:
                if not units:
                    rows.append(None); continue
                vals_per_lag = []
                for sub, ch in units:
                    df = load_tr(sub, ch, band)
                    if df is None: continue
                    sub_st = df[df["stage"] == stage]
                    if sub_st.empty: continue
                    sub_st = sub_st.copy()
                    sub_st["mean_lag"] = (sub_st["lag1_min"]
                                          + sub_st["lag2_min"]) / 2.0
                    m = (sub_st.groupby("mean_lag")[atom]
                                .mean().reset_index()
                                .sort_values("mean_lag"))
                    vals_per_lag.append(m)
                if not vals_per_lag:
                    rows.append(None); continue
                # outer-join by mean_lag, average
                joined = vals_per_lag[0][["mean_lag", atom]].rename(
                    columns={atom: f"{atom}_0"})
                for i, v in enumerate(vals_per_lag[1:], start=1):
                    joined = joined.merge(
                        v.rename(columns={atom: f"{atom}_{i}"}),
                        on="mean_lag", how="outer")
                cols = [c for c in joined.columns if c.startswith(atom)]
                joined["mean"] = joined[cols].mean(axis=1)
                joined = joined.sort_values("mean_lag")
                rows.append(joined[["mean_lag", "mean"]])
                if lag_vals is None or len(joined) > len(lag_vals):
                    lag_vals = joined["mean_lag"].values

        # Build matrix on common lag grid
        if lag_vals is None:
            ax.set_title(lab, fontweight="bold")
            continue
        lag_vals = np.array(sorted(set(np.concatenate(
            [r["mean_lag"].values for r in rows if r is not None]))))
        mat = np.full((len(row_labels), len(lag_vals)), np.nan)
        for ri, r in enumerate(rows):
            if r is None: continue
            for _, row in r.iterrows():
                ci = np.searchsorted(lag_vals, row["mean_lag"])
                if 0 <= ci < len(lag_vals):
                    mat[ri, ci] = row["mean"]

        # Smooth each row along the lag axis (rolling mean win=11) so the
        # discriminative peaks aren't buried in lag-pair noise.
        def _smooth(arr, win=11):
            arr = np.asarray(arr, dtype=float)
            if win < 2 or np.sum(~np.isnan(arr)) <= win:
                return arr
            pad = win // 2
            valid = ~np.isnan(arr)
            x = arr.copy(); x[~valid] = np.nanmean(arr[valid])
            padded = np.pad(x, pad, mode="edge")
            sm = np.convolve(padded, np.ones(win) / win, mode="valid")[:len(arr)]
            sm[~valid] = np.nan
            return sm
        mat_sm = np.array([_smooth(mat[ri], win=11) for ri in range(mat.shape[0])])

        # z-score along each row (post-smoothing) so stage × band rows are
        # visually comparable despite different absolute scales per band.
        z = mat_sm.copy()
        for ri in range(z.shape[0]):
            row = z[ri]
            m = np.nanmean(row); s = np.nanstd(row)
            if s > 1e-12:
                z[ri] = (row - m) / s

        sns.heatmap(z, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-2.5, vmax=2.5,
                    xticklabels=[f"{v:.2f}" if i % max(1, len(lag_vals) // 6) == 0
                                 else "" for i, v in enumerate(lag_vals)],
                    yticklabels=row_labels,
                    cbar_kws={"label": "row z-score", "shrink": 0.8},
                    mask=np.isnan(z))
        # Horizontal separators between bands
        for bi in range(1, len(BANDS)):
            ax.axhline(bi * len(STAGE_ORDER), color="black",
                        lw=0.8, alpha=0.7)
        ax.set_title(lab, fontweight="bold", fontsize=12)
        ax.set_xlabel("Mean lag (min)", fontsize=10)
        ax.tick_params(labelsize=8)
        # Colour the ytick labels by stage
        for ri, label in enumerate(ax.get_yticklabels()):
            stage = row_labels[ri].split("·")[1]
            label.set_color(STAGE_COLORS.get(stage, "#000"))
            label.set_fontweight("bold")

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(
        f"Lag-resolved (band × stage) atom maps  "
        f"({n_subs} subject(s); each row z-scored across lag — "
        f"colour shows where on the lag axis the value peaks)",
        fontsize=13, fontweight="bold")
    out = OUT_DIR / f"f07_lag_profile_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 7 → {out}")


# ===========================================================================
# FIGURE 8: NREM→REM transition atom trajectories per band
# ===========================================================================
def fig08_nrem_rem_transitions(inventory, window_around_s=60):
    """Per band per atom: aligned-to-REM-onset atom trajectory averaged."""
    rows_traj = []
    for sub, ch, band in inventory:
        df = load_tr(sub, ch, band)
        if df is None: continue
        # Per-window mean atoms (across lag pairs).
        wm = (df.groupby(["window", "time_min", "stage"])[ATOMS + ["sr_ratio"]]
                 .mean().reset_index().sort_values("window"))
        # Find NREM→REM transitions
        prev_stage = None
        for i, r in wm.iterrows():
            if prev_stage in ("N2", "N3") and r["stage"] == "REM":
                t_onset = r["time_min"]
                lo = t_onset - window_around_s / 60.0
                hi = t_onset + window_around_s / 60.0
                seg = wm[(wm["time_min"] >= lo) & (wm["time_min"] <= hi)].copy()
                seg["t_rel"] = (seg["time_min"] - t_onset) * 60  # seconds
                seg["band"] = band; seg["subject"] = sub; seg["ch_short"] = ch
                rows_traj.append(seg)
            prev_stage = r["stage"]
    if not rows_traj:
        print("  fig 8: no NREM→REM transitions found"); return
    big = pd.concat(rows_traj, ignore_index=True)

    # Wider averaging bins (9 s); subtract pre-onset baseline so each panel
    # shows the *change* at REM onset, comparable across bands and atoms.
    bin_edges = np.arange(-window_around_s, window_around_s + 1, 9)

    fig, axes = plt.subplots(len(ATOMS), len(BANDS),
                             figsize=(3.6 * len(BANDS), 2.8 * len(ATOMS)),
                             constrained_layout=True, sharex=True, sharey="row")
    atom_line_colors = {"redundancy": "#2ca02c", "synergy": "#d62728",
                         "unique_0": "#1f77b4", "unique_1": "#9467bd"}

    for ai, (atom, lab) in enumerate(zip(ATOMS, ATOM_LABELS)):
        for bi, band in enumerate(BANDS):
            ax = axes[ai, bi]
            sub = big[big["band"] == band]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                if ai == 0:
                    ax.set_title(f"{band}\n({BAND_FREQS[band]})",
                                 fontsize=10, fontweight="bold")
                if bi == 0:
                    ax.set_ylabel(lab, fontsize=10, fontweight="bold")
                continue
            sub2 = sub.copy()
            t_bin = pd.cut(sub2["t_rel"], bins=bin_edges)
            sub2["t_mid"] = np.array(
                [float(iv.mid) if pd.notna(iv) else np.nan for iv in t_bin],
                dtype=float)
            grp = (sub2.dropna(subset=["t_mid"])
                       .groupby("t_mid", as_index=False)[atom]
                       .agg(["mean", "sem"]))
            grp.columns = ["t_mid", "mean", "sem"]
            grp = grp.dropna(subset=["mean"]).sort_values("t_mid")
            grp["t_mid"] = grp["t_mid"].astype(float)
            # Baseline = mean over t < 0 (pre-REM).
            base = grp.loc[grp["t_mid"] < 0, "mean"].mean()
            grp["dmean"] = grp["mean"] - base
            ax.fill_between(grp["t_mid"], grp["dmean"] - grp["sem"],
                             grp["dmean"] + grp["sem"],
                             color=atom_line_colors[atom], alpha=0.20)
            ax.plot(grp["t_mid"], grp["dmean"],
                     color=atom_line_colors[atom], lw=2.2)
            ax.axvline(0, color="#DC143C", lw=1.2, ls="--", alpha=0.8)
            ax.axhline(0, color="#666", lw=0.5)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            if ai == 0:
                ax.set_title(f"{band}\n({BAND_FREQS[band]})",
                             fontsize=10, fontweight="bold")
            if bi == 0:
                ax.set_ylabel(f"{lab}\nΔ from pre-REM (bits)",
                              fontsize=9, fontweight="bold")
            if ai == len(ATOMS) - 1:
                ax.set_xlabel("Time relative to REM onset (s)",
                              fontsize=9)

    n_trans = big.groupby(["subject", "ch_short", "band"]).size().reset_index().shape[0]
    fig.suptitle(
        f"Atom trajectory aligned to NREM→REM onset, per band\n"
        f"({n_trans} transitions averaged across (subject, channel, band) units)",
        fontsize=12, fontweight="bold")
    out = OUT_DIR / f"f08_nrem_rem_transitions_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 8 → {out}")


# ===========================================================================
# FIGURE 9: Per-band stage-comparison with bootstrap CIs — compact
# ===========================================================================
def fig09_band_stage_bootstrap(inventory):
    """One panel per metric (5 panels in a row). Within each panel the
    x-axis ticks are sleep stages; each stage has 5 grouped bars (one per
    band), coloured by band. Bootstrap 95% CI as error bars.

    Collapses the previous 5×5 panel grid down to a single horizontal row.
    """
    long = collect_per_unit_per_stage(inventory, ATOMS + ["sr_ratio"])
    if long is None: return

    metrics = [("synergy", "Synergy"), ("redundancy", "Redundancy"),
               ("unique_0", "Unique₁"), ("unique_1", "Unique₂"),
               ("sr_ratio", "S / R")]
    # Colour-blind-safe band palette ordered low→high frequency
    band_colors = {"delta": "#2c7fb8", "theta": "#7fcdbb",
                    "alpha": "#fec44f", "sigma": "#fe9929", "beta": "#d95f0e"}

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(4.6 * len(metrics), 6.5),
                             sharex=True)
    fig.subplots_adjust(top=0.85, bottom=0.20, left=0.05, right=0.985,
                        wspace=0.30)
    rng = np.random.default_rng(42)
    n_bands = len(BANDS); bar_w = 0.16
    # Add room between stage groups by spreading them out.
    xb = np.arange(len(STAGE_ORDER)) * 1.3

    for mi, (metric, label) in enumerate(metrics):
        ax = axes[mi]
        for bi, band in enumerate(BANDS):
            sub = long[long["band"] == band]
            if sub.empty: continue
            means = []; lo_arr = []; hi_arr = []
            for s in STAGE_ORDER:
                m, lo, hi = bootstrap_ci(
                    sub.loc[sub["stage"] == s, metric].dropna().values,
                    n_boot=500, rng=rng)
                means.append(m); lo_arr.append(lo); hi_arr.append(hi)
            means = np.array(means, dtype=float)
            lo_arr = np.array(lo_arr, dtype=float)
            hi_arr = np.array(hi_arr, dtype=float)
            offset = (bi - (n_bands - 1) / 2) * bar_w
            errs = np.array([means - lo_arr, hi_arr - means])
            ax.bar(xb + offset, means, bar_w,
                    color=band_colors[band], edgecolor="white",
                    linewidth=0.5,
                    label=f"{band}" if mi == 0 else None)
            ax.errorbar(xb + offset, means, yerr=errs, fmt="none",
                         ecolor="#333", capsize=2, lw=0.8)
        ax.set_xticks(xb); ax.set_xticklabels(STAGE_ORDER, fontsize=12)
        ax.tick_params(axis="x", pad=8)
        # Colour each stage label by its stage colour
        for ti, t in enumerate(ax.get_xticklabels()):
            t.set_color(STAGE_COLORS[STAGE_ORDER[ti]])
            t.set_fontweight("bold")
        ax.set_ylabel("bits" if metric != "sr_ratio" else "ratio",
                       fontsize=11)
        ax.set_title(label, fontweight="bold", fontsize=13, pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=10)
        # Slight extra ylim headroom so error bars don't crowd the title.
        yl0, yl1 = ax.get_ylim()
        ax.set_ylim(yl0, yl1 + 0.08 * (yl1 - yl0))

    # Single band legend, more space below the panels.
    handles = [plt.Rectangle((0, 0), 1, 1, color=band_colors[b]) for b in BANDS]
    labels = [f"{b} ({BAND_FREQS[b]})" for b in BANDS]
    fig.legend(handles, labels, loc="lower center", ncol=len(BANDS),
                bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12,
                title="Band", title_fontsize=11)

    n_subs = len(set(s for s, _, _ in inventory))
    fig.suptitle(f"Per-band stage comparison with bootstrap 95% CI  "
                 f"({n_subs} subject(s) × 6 channels per band)",
                 fontsize=14, fontweight="bold", y=0.96)
    out = OUT_DIR / f"f09_band_stage_bootstrap_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 9 → {out}")


# ===========================================================================
# FIGURE 10: α vs δ S/R-ratio "compass"
# ===========================================================================
def fig10_alpha_vs_delta_compass(inventory):
    """Single scatter: x = δ S/R, y = α S/R, one point per (sub, ch, stage)."""
    long = collect_per_unit_per_stage(inventory, ["sr_ratio"])
    if long is None: return
    pd_d = long[long["band"] == "delta"][["subject", "ch_short", "stage", "sr_ratio"]] \
        .rename(columns={"sr_ratio": "sr_delta"})
    pd_a = long[long["band"] == "alpha"][["subject", "ch_short", "stage", "sr_ratio"]] \
        .rename(columns={"sr_ratio": "sr_alpha"})
    m = pd_d.merge(pd_a, on=["subject", "ch_short", "stage"], how="inner")
    if m.empty:
        print("  fig 10: no (sub, ch, stage) common to both delta and alpha")
        return

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    # Ellipses + centroids
    for stage in STAGE_ORDER:
        sub = m[m["stage"] == stage]
        if sub.empty: continue
        xs = sub["sr_delta"].values; ys = sub["sr_alpha"].values
        ax.scatter(xs, ys, c=STAGE_COLORS[stage], s=110, alpha=0.7,
                   edgecolors="black", linewidths=1.0, label=stage,
                   zorder=3)
        if len(xs) >= 3:
            cov = np.cov(xs, ys)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]; vecs = vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * 2.448 * np.sqrt(vals)
            ell = Ellipse((xs.mean(), ys.mean()), width=w, height=h,
                          angle=angle, facecolor=STAGE_COLORS[stage],
                          alpha=0.16, edgecolor=STAGE_COLORS[stage],
                          lw=1.5, zorder=2)
            ax.add_patch(ell)
        ax.scatter(xs.mean(), ys.mean(), c=STAGE_COLORS[stage], s=400,
                   alpha=1.0, marker="X", edgecolors="black",
                   linewidths=1.6, zorder=5)

    # Zoom to the data range with a small pad. Drop the α = δ diagonal
    # (visually distracting because the two axes live at very different scales).
    x_lo, x_hi = m["sr_delta"].min(), m["sr_delta"].max()
    y_lo, y_hi = m["sr_alpha"].min(), m["sr_alpha"].max()
    x_pad = max(0.08 * (x_hi - x_lo), 0.05)
    y_pad = max(0.08 * (y_hi - y_lo), 0.05)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_xlabel("δ-band  S/R ratio", fontsize=13)
    ax.set_ylabel("α-band  S/R ratio", fontsize=13)
    ax.set_title(
        "α vs δ  S/R compass — band-specific stage inversion\n"
        "(dots = (subject, channel); X = stage centroid; ellipse = 95% covariance)",
        fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=11, frameon=False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    out = OUT_DIR / f"f10_alpha_vs_delta_compass_{BAND_HASH}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig 10 → {out}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    inv = list(discover())
    n_subs = len(set(s for s, _, _ in inv))
    print(f"Inventory: {len(inv)} (sub, ch, band) units across {n_subs} subject(s).")
    if not inv:
        return
    fig01_band_stage_diff(inv, contrast=("N3", "Wake"))
    fig01_band_stage_diff(inv, contrast=("REM", "Wake"),
                          save_name=f"f01_band_stage_diff_remvswake_{BAND_HASH}.png")
    fig02_discriminability_scoreboard(inv)
    fig03_phase_portrait(inv)
    fig04_band_excess_ar1(inv)
    fig05_cross_band_correlation(inv)
    fig06_sr_spectrogram(inv)
    fig07_lag_profile(inv)
    fig08_nrem_rem_transitions(inv)
    fig09_band_stage_bootstrap(inv)
    fig10_alpha_vs_delta_compass(inv)
    print("\nDone.")


if __name__ == "__main__":
    main()
