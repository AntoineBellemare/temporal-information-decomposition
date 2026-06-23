"""
EEG Sleep Temporal PID — Group-level figures for the 1-s report
================================================================

Reads per-subject CSVs under
  results/pid/eeg_sleep/PID-10-subjects-1sec/<sub-X>/<channel>/
and produces:

  - group_all_stage_comparison_<hash>.png   (boxplots, all 6 channels pooled,
                                              subject as random effect)
  - group_all_effect_sizes_<hash>.png       (Cohen's d heatmaps per atom)
  - group_block_permutation_C3_<hash>.png   (block-perm significance map,
                                              C3 composite — uses whichever
                                              subjects have block_perm NPZ).

Usage:
    python scripts/pid/eeg_sleep_group_figs.py
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
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu, levene

# UTF-8 stdout on Windows.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths + constants
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT_DIR / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
OUT_DIR = RESULTS_BASE / "group"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHASH = "ba79f4ef"
WINDOW_SEC = 1
MAX_LAG_MIN = 0.5
WINDOWS_PER_MIN = 60.0 / WINDOW_SEC
MIN_PER_WINDOW = WINDOW_SEC / 60.0

STAGE_ORDER = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "Wake": "#E8A317",
    "N1":   "#87CEEB",
    "N2":   "#4169E1",
    "N3":   "#191970",
    "REM":  "#DC143C",
}
CHANNELS = ["C3", "C4", "F3", "F4", "O1", "O2"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def available_subjects() -> list[str]:
    if not RESULTS_BASE.exists():
        return []
    out = []
    for d in sorted(RESULTS_BASE.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            if (d / "C3" / f"timeresolved_pid_filtered_{CHASH}.csv").exists():
                out.append(d.name)
    return out


def load_tr(subject, channel):
    p = RESULTS_BASE / subject / channel / f"timeresolved_pid_filtered_{CHASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_global(subject, channel):
    p = RESULTS_BASE / subject / channel / f"global_pid_matrix_{CHASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_null(subject, channel):
    p = RESULTS_BASE / subject / channel / f"block_perm_null_{CHASH}.npz"
    if not p.exists():
        return None
    return np.load(p)["null_vals"]


# ----------------------------------------------------------------------------
# Helpers used by the boxplot / effect-size figures
# ----------------------------------------------------------------------------
def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    adj = np.empty(n)
    adj[order[-1]] = pvals[order[-1]]
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(pvals[order[i]] * n / (i + 1), adj[order[i + 1]])
    return np.clip(adj, 0, 1)


def common_lag_range(tr):
    present = [s for s in STAGE_ORDER if s in tr["stage"].unique()]
    if len(present) < 2:
        return tr
    sets = []
    for s in present:
        sub = tr[tr["stage"] == s]
        sets.append(set(zip(sub["lag1_min"], sub["lag2_min"])))
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    if not common:
        return tr
    common_df = pd.DataFrame(list(common), columns=["lag1_min", "lag2_min"])
    return tr.merge(common_df, on=["lag1_min", "lag2_min"])


def aggregate_subjects(subjects):
    """Build a long-format DataFrame across (subject × channel × window × lag)."""
    rows = []
    for subj in subjects:
        for ch in CHANNELS:
            tr = load_tr(subj, ch)
            if tr is None:
                continue
            tr = common_lag_range(tr).copy()
            tr["subject"] = subj
            tr["ch_short"] = ch
            rows.append(tr)
    if not rows:
        return None
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["stage"].isin(STAGE_ORDER)]
    combined["unique_total"] = combined["unique_0"] + combined["unique_1"]
    combined["ratio"] = combined["synergy"] / combined["redundancy"].replace(0, np.nan)
    return combined


def subject_agg(combined):
    return combined.groupby(["subject", "ch_short", "stage"]).agg(
        redundancy=("redundancy", "mean"),
        synergy=("synergy", "mean"),
        unique_total=("unique_total", "mean"),
        ratio=("ratio", "mean"),
    ).reset_index()


# ----------------------------------------------------------------------------
# Figure A: group stage comparison boxplots (4 panels)
# ----------------------------------------------------------------------------
def plot_group_stage_comparison(subjects, save_path):
    combined = aggregate_subjects(subjects)
    if combined is None:
        print("  [stage_comparison] no data")
        return
    n_subj = combined["subject"].nunique()
    agg = subject_agg(combined)

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    palette = {s: STAGE_COLORS[s] for s in STAGE_ORDER}
    metrics = [
        ("redundancy", "Redundancy (bits)"),
        ("synergy", "Synergy (bits)"),
        ("unique_total", "Total Unique (bits)"),
        ("ratio", "Synergy / Redundancy"),
    ]
    for ax, (col, ylabel) in zip(axes, metrics):
        sns.boxplot(data=agg, x="stage", y=col, order=STAGE_ORDER,
                    palette=palette, ax=ax, showfliers=False, width=0.6)
        ax.set_xlabel("Sleep stage"); ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        title = "S / R Ratio" if col == "ratio" else col.replace("_", " ").title()
        if col == "ratio":
            ax.axhline(1.0, color="grey", ls="--", alpha=0.6, label="S = R")
            ax.legend(fontsize=8)

        groups = [agg.loc[agg["stage"] == s, col].dropna().values for s in STAGE_ORDER
                  if len(agg.loc[agg["stage"] == s, col].dropna()) > 0]
        if len(groups) >= 2:
            H, p_kw = kruskal(*groups)
            sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "n.s."
            ax.set_title(f"{title}\nKW: H={H:.1f}, p={p_kw:.1e} {sig}",
                         fontweight="bold", fontsize=10)
        else:
            ax.set_title(title, fontweight="bold")

        present = [s for s in STAGE_ORDER
                   if len(agg.loc[agg["stage"] == s, col].dropna()) >= 3]
        all_pairs, all_pvals = [], []
        for s1, s2 in combinations(present, 2):
            v1 = agg.loc[agg["stage"] == s1, col].dropna().values
            v2 = agg.loc[agg["stage"] == s2, col].dropna().values
            if len(v1) >= 3 and len(v2) >= 3:
                _, p_mw = mannwhitneyu(v1, v2, alternative="two-sided")
                all_pairs.append((s1, s2, p_mw)); all_pvals.append(p_mw)
        if all_pvals:
            adj = benjamini_hochberg(all_pvals)
            sig_pairs = [(s1, s2, p) for (s1, s2, _), p in zip(all_pairs, adj) if p < 0.05]
            sig_pairs.sort(key=lambda x: x[2])
            y_max = agg[col].dropna().quantile(0.95) if len(agg[col].dropna()) else 1
            step = y_max * 0.08
            for rank, (s1, s2, p_val) in enumerate(sig_pairs[:4]):
                x1 = STAGE_ORDER.index(s1); x2 = STAGE_ORDER.index(s2)
                y_bar = y_max + step * (rank + 1)
                ax.plot([x1, x1, x2, x2],
                        [y_bar - step * 0.2, y_bar, y_bar, y_bar - step * 0.2],
                        color="black", lw=0.8)
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                ax.text((x1 + x2) / 2, y_bar, stars, ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"PID by sleep stage — group (n = {n_subj} subjects × {len(CHANNELS)} channels)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [stage_comparison] {save_path}")


# ----------------------------------------------------------------------------
# Figure B: group effect sizes (Cohen's d)
# ----------------------------------------------------------------------------
def plot_group_effect_sizes(subjects, save_path):
    combined = aggregate_subjects(subjects)
    if combined is None:
        print("  [effect_sizes] no data"); return
    n_subj = combined["subject"].nunique()
    agg = subject_agg(combined)
    present = [s for s in STAGE_ORDER if s in agg["stage"].unique()]
    n = len(present)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, metric, label in zip(axes,
                                  ["redundancy", "synergy", "unique_total", "ratio"],
                                  ["Redundancy", "Synergy", "Total Unique", "S/R Ratio"]):
        d_mat = np.full((n, n), np.nan); p_mat = np.full((n, n), np.nan)
        for i, s1 in enumerate(present):
            for j, s2 in enumerate(present):
                if i >= j: continue
                v1 = agg.loc[agg["stage"] == s1, metric].dropna().values
                v2 = agg.loc[agg["stage"] == s2, metric].dropna().values
                if len(v1) < 3 or len(v2) < 3: continue
                pooled = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1)
                                 +(len(v2)-1)*np.var(v2, ddof=1)) /
                                 (len(v1)+len(v2)-2))
                if pooled > 0:
                    d_mat[i, j] = (np.mean(v1) - np.mean(v2)) / pooled
                    d_mat[j, i] = -d_mat[i, j]
                _, pv = mannwhitneyu(v1, v2, alternative="two-sided")
                p_mat[i, j] = pv; p_mat[j, i] = pv

        upper_p, upper_idx = [], []
        for i in range(n):
            for j in range(i+1, n):
                if not np.isnan(p_mat[i, j]):
                    upper_p.append(p_mat[i, j]); upper_idx.append((i, j))
        p_fdr = np.full((n, n), np.nan)
        if upper_p:
            adj = benjamini_hochberg(upper_p)
            for (i, j), ap in zip(upper_idx, adj):
                p_fdr[i, j] = ap; p_fdr[j, i] = ap
        vmax = np.nanmax(np.abs(d_mat)) if not np.all(np.isnan(d_mat)) else 1
        sns.heatmap(d_mat, ax=ax, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                    xticklabels=present, yticklabels=present, annot=True, fmt=".2f",
                    annot_kws={"fontsize": 8},
                    cbar_kws={"label": "Cohen's d", "shrink": 0.8},
                    mask=np.isnan(d_mat))
        for i in range(n):
            for j in range(n):
                if not np.isnan(p_fdr[i, j]) and p_fdr[i, j] < 0.05:
                    ax.text(j + 0.5, i + 0.8, "*", ha="center", va="center",
                            fontsize=10, fontweight="bold")
        ax.set_title(label, fontweight="bold")

    plt.suptitle(f"Effect sizes (Cohen's d) — group (n={n_subj})\n(* = FDR p < 0.05)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [effect_sizes] {save_path}")


# ----------------------------------------------------------------------------
# Figure C: group block-permutation (C3 composite)
# ----------------------------------------------------------------------------
def plot_group_block_permutation(subjects, channel, save_path):
    """Combine subjects' block_perm NPZs (where present) into a Fisher-style
    composite p-value heatmap per atom."""
    valid = []
    for subj in subjects:
        gl = load_global(subj, channel)
        nv = load_null(subj, channel)
        if gl is not None and nv is not None:
            valid.append((subj, gl, nv))
    if not valid:
        print(f"  [block_perm {channel}] no null_vals on disk"); return

    n_subj = len(valid)
    lag_vals = sorted(set(valid[0][1]["lag1_min"]) | set(valid[0][1]["lag2_min"]))
    l_idx = {v: i for i, v in enumerate(lag_vals)}
    n_lags = len(lag_vals)
    max_lag_w = int(round(MAX_LAG_MIN * WINDOWS_PER_MIN))
    lag_pairs_w = [(l1, l2) for l1 in range(1, max_lag_w)
                   for l2 in range(l1 + 1, max_lag_w + 1)]

    comps = ["redundancy", "synergy", "unique_0", "unique_1"]
    titles = ["Redundancy", "Synergy", "Unique₁", "Unique₂"]
    cmaps = ["Greens", "Reds", "Blues", "Purples"]
    comp_idx = {c: i for i, c in enumerate(comps)}

    all_p = {c: [] for c in comps}
    for subj, gl, nv in valid:
        for comp in comps:
            ci = comp_idx[comp]
            p_mat = np.full((n_lags, n_lags), np.nan)
            for li, (l1w, l2w) in enumerate(lag_pairs_w):
                l1_min = round(l1w * MIN_PER_WINDOW, 4)
                l2_min = round(l2w * MIN_PER_WINDOW, 4)
                row = gl[(gl["lag1_min"] == l1_min) & (gl["lag2_min"] == l2_min)]
                if row.empty: continue
                observed = row[comp].values[0]
                null = nv[:, li, ci]
                p = (np.sum(null >= observed) + 1) / (len(null) + 1)
                if l1_min in l_idx and l2_min in l_idx:
                    p_mat[l_idx[l1_min], l_idx[l2_min]] = p
            all_p[comp].append(p_mat)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, comp, title, cmap in zip(axes.flat, comps, titles, cmaps):
        stack = np.asarray(all_p[comp])
        combined_logp = np.full((n_lags, n_lags), np.nan)
        for i in range(n_lags):
            for j in range(n_lags):
                ps = stack[:, i, j]; valid_ps = ps[~np.isnan(ps)]
                if len(valid_ps):
                    combined_logp[i, j] = np.mean(-np.log10(np.maximum(valid_ps, 1e-10)))
        sns.heatmap(combined_logp, ax=ax, cmap=cmap, mask=np.isnan(combined_logp),
                    xticklabels=[f"{v:g}" for v in lag_vals],
                    yticklabels=[f"{v:g}" for v in lag_vals],
                    cbar_kws={"label": "mean -log₁₀(p)"})
        for li in range(n_lags):
            for lj in range(n_lags):
                if not np.isnan(combined_logp[li, lj]) and combined_logp[li, lj] > -np.log10(0.05):
                    ax.text(lj + 0.5, li + 0.5, "*", ha="center", va="center",
                            fontsize=12, fontweight="bold", color="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Lag 2 (min)"); ax.set_ylabel("Lag 1 (min)")

    plt.suptitle(f"Block-permutation significance — group ({channel}, n={n_subj})\n"
                 f"(mean -log₁₀(p) across subjects, * = p < 0.05)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [block_perm {channel}] {save_path}")


# ----------------------------------------------------------------------------
# Figure D: group distributional analysis (replicates per-subject plot 16
# across all subjects × all channels)
# ----------------------------------------------------------------------------
def _smooth(arr, win=7):
    arr = np.asarray(arr, dtype=float)
    if win < 2 or len(arr) <= win:
        return arr.copy()
    pad = win // 2
    padded = np.pad(arr, pad, mode='edge')
    return np.convolve(padded, np.ones(win) / win, mode='valid')[:len(arr)]


def plot_group_distribution_analysis(subjects, save_path):
    """Group-level CV analysis across stages — paper-ready 2-row layout.

    Row 1: within-window CV per (subject, channel) unit per stage.
           Bars with bootstrap 95% CI, KW + pairwise BH-FDR brackets.
    Row 2: CV by timescale (mean lag), raw at α=0.25, rolling-mean smoothed
           on top at α=0.95.
    """
    combined = aggregate_subjects(subjects)
    if combined is None:
        print("  [group_distribution] no data"); return
    n_subj = combined['subject'].nunique()

    combined = combined.copy()
    combined['unique_total'] = combined['unique_0'] + combined['unique_1']
    combined['sr_ratio'] = np.where(
        combined['redundancy'] > 0,
        combined['synergy'] / combined['redundancy'], np.nan)

    present = [s for s in STAGE_ORDER if s in combined['stage'].unique()]
    metrics = [
        ('synergy',      'Synergy'),
        ('redundancy',   'Redundancy'),
        ('unique_total', 'Total Unique'),
        ('sr_ratio',     'S / R Ratio'),
    ]

    fig, axes = plt.subplots(2, len(metrics), figsize=(5 * len(metrics), 9),
                             constrained_layout=True)

    # ---- Row 1: per-(subject, channel) within-window CV bars ------------
    rng = np.random.default_rng(42); n_boot = 500
    for ci, (metric, label) in enumerate(metrics):
        ax = axes[0, ci]
        cv_groups = {}
        cv_means, cv_lo, cv_hi = [], [], []
        for stage in present:
            sub = combined[combined['stage'] == stage]
            if sub.empty:
                cv_means.append(np.nan); cv_lo.append(np.nan); cv_hi.append(np.nan); continue
            per_w = (sub.groupby(['subject', 'ch_short', 'window'])[metric]
                       .agg(['mean', 'std']).reset_index())
            per_w['cv'] = np.where(per_w['mean'] > 0,
                                   per_w['std'] / per_w['mean'], np.nan)
            per_unit = per_w.dropna(subset=['cv']).groupby(
                ['subject', 'ch_short'])['cv'].mean().values
            if len(per_unit) < 2:
                cv_means.append(np.nan); cv_lo.append(np.nan); cv_hi.append(np.nan); continue
            cv_groups[stage] = per_unit
            cv_means.append(np.mean(per_unit))
            boot = np.array([np.mean(rng.choice(per_unit, len(per_unit), replace=True))
                             for _ in range(n_boot)])
            cv_lo.append(np.percentile(boot, 2.5))
            cv_hi.append(np.percentile(boot, 97.5))

        x = np.arange(len(present))
        colors = [STAGE_COLORS[s] for s in present]
        errs = np.array([[m - lo, hi - m]
                         for m, lo, hi in zip(cv_means, cv_lo, cv_hi)]).T
        ax.bar(x, cv_means, color=colors, edgecolor='white',
               alpha=0.9, linewidth=1.2)
        ax.errorbar(x, cv_means, yerr=errs, fmt='none',
                    ecolor='#222', capsize=4, lw=1.3)
        ax.set_xticks(x); ax.set_xticklabels(present, fontsize=11)
        ax.set_ylabel('CV  (std / mean)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=10)

        # Tight y-limits with headroom for brackets.
        valid_cv = [v for v in cv_means if not np.isnan(v)]
        valid_lo = [v for v in cv_lo if not np.isnan(v)]
        valid_hi = [v for v in cv_hi if not np.isnan(v)]
        if valid_cv:
            lo_v = min(valid_lo + valid_cv)
            hi_v = max(valid_hi + valid_cv)
            pad = max(0.05 * (hi_v - lo_v), 0.003)
            ax.set_ylim(max(0, lo_v - pad), hi_v + 7 * pad)

        grp_list = [v for v in cv_groups.values() if len(v) >= 2]
        if len(grp_list) >= 2:
            H, p = kruskal(*grp_list)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.set_title(f'{label}\nKW: H={H:.1f}, p={p:.1e} {stars}',
                         fontweight='bold', fontsize=11)
            pairs = list(combinations([s for s in present if s in cv_groups], 2))
            pvals = []
            for s1, s2 in pairs:
                if len(cv_groups[s1]) >= 2 and len(cv_groups[s2]) >= 2:
                    _, pw = mannwhitneyu(cv_groups[s1], cv_groups[s2],
                                         alternative='two-sided')
                    pvals.append(pw)
                else:
                    pvals.append(1.0)
            adj_p = benjamini_hochberg(np.array(pvals))
            sig = [(pairs[i], adj_p[i]) for i in range(len(pairs)) if adj_p[i] < 0.05]
            sig.sort(key=lambda x: x[1])
            # Position brackets in axes-fraction units so they always sit
            # cleanly near the top of the panel regardless of ylim.
            yl0, yl1 = ax.get_ylim()
            span = yl1 - yl0
            top_band_start = yl0 + 0.78 * span
            step = 0.06 * span
            for rank, ((s1, s2), pv) in enumerate(sig[:3]):
                i1, i2 = present.index(s1), present.index(s2)
                y = top_band_start + rank * step
                ax.plot([i1, i1, i2, i2],
                        [y - 0.012 * span, y, y, y - 0.012 * span],
                        lw=1.0, c='#444', clip_on=False)
                star = '***' if pv < 0.001 else '**' if pv < 0.01 else '*'
                ax.text((i1 + i2) / 2, y + 0.005 * span, star,
                        ha='center', va='bottom', fontsize=10,
                        fontweight='bold', clip_on=False)

    # ---- Row 2: CV by timescale (within-recording form) ----------------
    # For each (subject, channel, lag pair): CV = std/mean across windows.
    # Then average those within-recording CVs across (subject, channel) for
    # each lag pair → no between-subject variance contaminates the curve.
    # Finally collapse to mean_lag.
    legend_handles = []
    legend_labels = []
    for ci, (metric, label) in enumerate(metrics):
        ax = axes[1, ci]
        for stage in present:
            sub = combined[(combined['stage'] == stage) & combined[metric].notna()]
            if sub.empty:
                continue
            per_unit = (sub.groupby(['subject', 'ch_short',
                                     'lag1_min', 'lag2_min'])[metric]
                          .agg(['mean', 'std']).reset_index())
            per_unit['cv'] = np.where(per_unit['mean'] > 0,
                                      per_unit['std'] / per_unit['mean'],
                                      np.nan)
            lag_cv = (per_unit.dropna(subset=['cv'])
                              .groupby(['lag1_min', 'lag2_min'])['cv']
                              .mean().reset_index())
            lag_cv['mean_lag'] = (lag_cv['lag1_min'] + lag_cv['lag2_min']) / 2
            grp = lag_cv.groupby('mean_lag')['cv'].mean().reset_index()
            grp = grp.sort_values('mean_lag').reset_index(drop=True)
            x = grp['mean_lag'].values
            y = grp['cv'].values
            ax.plot(x, y, color=STAGE_COLORS[stage], lw=0.9, alpha=0.25, zorder=1)
            h, = ax.plot(x, _smooth(y, win=7), color=STAGE_COLORS[stage],
                         lw=2.4, alpha=0.95, zorder=3, label=stage)
            if ci == 0 and stage not in legend_labels:
                legend_handles.append(h); legend_labels.append(stage)
        ax.set_xlabel('Mean lag (min)', fontsize=11)
        ax.set_ylabel('Within-recording CV  (mean ± subj × ch)',
                      fontsize=11)
        ax.set_title(f'{label} — CV by timescale',
                     fontweight='bold', fontsize=11)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    if legend_handles:
        fig.legend(legend_handles, legend_labels,
                   loc='lower center', ncol=len(legend_labels),
                   frameon=False, fontsize=11,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f'Distributional variability of PID atoms across sleep stages\n'
        f'group (n = {n_subj} subjects × {len(CHANNELS)} channels)',
        fontsize=13, fontweight='bold')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [group_distribution] {save_path}")


# ----------------------------------------------------------------------------
def main():
    subjects = available_subjects()
    print("=" * 60)
    print(f"Subjects with data: {len(subjects)} → {subjects}")
    print("=" * 60)
    if not subjects:
        print("No per-subject data found. Run eeg_sleep_compute.py first.")
        sys.exit(1)
    plot_group_stage_comparison(subjects,
                                OUT_DIR / f"group_all_stage_comparison_{CHASH}.png")
    plot_group_effect_sizes(subjects,
                            OUT_DIR / f"group_all_effect_sizes_{CHASH}.png")
    plot_group_block_permutation(subjects, "C3",
                                 OUT_DIR / f"group_block_permutation_C3_{CHASH}.png")
    plot_group_distribution_analysis(
        subjects, OUT_DIR / f"group_distribution_analysis_{CHASH}.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
