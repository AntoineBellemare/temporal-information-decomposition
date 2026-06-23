"""
EEG Sleep Temporal PID — Extra figures for report
===================================================
Generates group-level figures:

  C. Per-subject consistency dot-line plot (N3 vs Wake synergy)
  D. Electrode topography — mean N3 synergy per electrode (group)
  E. Within-stage synergy vs delta-band power, N3 vs Wake (confound check)

Usage:
    python scripts/pid/eeg_sleep_extra_figs.py
"""

import hashlib
import json
import glob
import sys
import warnings
from pathlib import Path

# Force UTF-8 on stdout/stderr — Windows consoles default to cp1252 and crash
# on stray Unicode glyphs (arrows, Greek, checkmarks).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats
from scipy.signal import welch

warnings.filterwarnings("ignore")

try:
    import mne
    mne.set_log_level('ERROR')
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent.parent
RESULTS_BASE = PROJECT_DIR / "results" / "pid" / "eeg_sleep" / "PID-10-subjects-1sec"
OUT_DIR      = RESULTS_BASE / "group"
DATA_DIR     = PROJECT_DIR / "data" / "ds005555"

# 1-s pass parameters.
CHASH        = "ba79f4ef"
WINDOW_SEC   = 1
DURATION_HRS = 3.5
DELTA_BAND   = (0.5, 4.0)   # Hz

STAGE_ORDER  = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "Wake": "#E8A317",
    "N1":   "#87CEEB",
    "N2":   "#4169E1",
    "N3":   "#191970",
    "REM":  "#DC143C",
}
CHANNELS = ["C3", "C4", "F3", "F4", "O1", "O2"]

# Electrode positions for a simple schematic (normalised -1..1)
ELEC_POS = {
    "F3": (-0.35,  0.55),
    "F4": ( 0.35,  0.55),
    "C3": (-0.35,  0.0),
    "C4": ( 0.35,  0.0),
    "O1": (-0.35, -0.55),
    "O2": ( 0.35, -0.55),
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_global_pid(subject: str, channel: str) -> pd.DataFrame | None:
    """Load global_pid_matrix CSV for one subject/channel (all lag pairs)."""
    p = RESULTS_BASE / subject / channel / f"global_pid_matrix_{CHASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_timeresolved(subject: str, channel: str) -> pd.DataFrame | None:
    """Load per-window time-resolved PID CSV for one subject/channel."""
    # prefer filtered version (stage-filtered windows, one row per window × lag-pair)
    p = RESULTS_BASE / subject / channel / f"timeresolved_pid_filtered_{CHASH}.csv"
    if not p.exists():
        p = RESULTS_BASE / subject / channel / f"timeresolved_pid_{CHASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_stage_labels(subject: str, channel: str) -> list[str] | None:
    p = RESULTS_BASE / subject / channel / f"stage_labels_{CHASH}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)["stage"].tolist()


def available_subjects(channel: str = "C3") -> list[str]:
    subs = []
    for d in sorted(RESULTS_BASE.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            cand = d / channel / f"global_pid_matrix_{CHASH}.csv"
            if cand.exists():
                subs.append(d.name)
    return subs


# ---------------------------------------------------------------------------
# Figure C: Per-subject consistency dot-line plot
# ---------------------------------------------------------------------------

def fig_C_subject_consistency():
    """
    For each subject and each electrode, compute the mean synergy and redundancy
    per stage (pooled over lag pairs).  Then plot N3 vs Wake synergy as paired
    dots+lines, one dot per (subject × electrode), 10 subjects × 6 channels = 60
    observations.  Confirms the effect is not driven by outliers.
    """
    records = []
    subjects = available_subjects("C3")  # use C3 to get subject list
    for subj in subjects:
        for ch in CHANNELS:
            tr = load_timeresolved(subj, ch)
            if tr is None or "synergy" not in tr.columns or "stage" not in tr.columns:
                continue
            for stage in STAGE_ORDER:
                grp = tr[tr["stage"] == stage]["synergy"]
                red = tr[tr["stage"] == stage]["redundancy"]
                if len(grp) == 0:
                    continue
                s_mean = grp.mean()
                r_mean = red.mean()
                records.append({
                    "subject": subj, "channel": ch, "stage": stage,
                    "synergy": s_mean, "redundancy": r_mean,
                    "sr_ratio": s_mean / (s_mean + r_mean) if (s_mean + r_mean) > 0 else np.nan,
                })

    if not records:
        print("  [C] No data found — skipping.")
        return None

    df = pd.DataFrame(records)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Per-subject \u00d7 electrode PID by sleep stage\n"
                 f"(n = {len(subjects)} subjects \u00d7 {len(CHANNELS)} electrodes = "
                 f"{len(subjects) * len(CHANNELS)} observations)", fontsize=12)

    atoms = [("synergy", "Synergy (bits)"), ("redundancy", "Redundancy (bits)"),
             ("sr_ratio", "S/R ratio")]
    for ax, (atom, ylabel) in zip(axes, atoms):
        # jitter x positions slightly per channel so overlap is visible
        ch_jitter = {ch: 0.06 * (i - (len(CHANNELS) - 1) / 2)
                     for i, ch in enumerate(CHANNELS)}

        for subj in subjects:
            for ch in CHANNELS:
                vals = []
                xs   = []
                for si, stage in enumerate(STAGE_ORDER):
                    row = df[(df["subject"] == subj) &
                             (df["channel"] == ch) &
                             (df["stage"] == stage)]
                    if len(row) == 0:
                        continue
                    vals.append(row[atom].values[0])
                    xs.append(si + ch_jitter[ch])
                if vals:
                    ax.plot(xs, vals, "-", color="grey", alpha=0.15, lw=0.8,
                            zorder=1)

        # Colored dots (all subjects × channels pooled per stage)
        for si, stage in enumerate(STAGE_ORDER):
            grp = df[df["stage"] == stage][atom]
            for j, ch in enumerate(CHANNELS):
                d = df[(df["stage"] == stage) & (df["channel"] == ch)][atom]
                ax.scatter(
                    np.full(len(d), si + ch_jitter[ch]),
                    d,
                    color=STAGE_COLORS[stage],
                    alpha=0.6, s=22, zorder=3, linewidths=0,
                )

        # Stage medians as larger markers
        medians = df.groupby("stage")[atom].median().reindex(STAGE_ORDER)
        ax.scatter(range(len(STAGE_ORDER)), medians.values,
                   color=[STAGE_COLORS[s] for s in STAGE_ORDER],
                   s=120, zorder=5, edgecolors="k", linewidths=1.2)
        ax.plot(range(len(STAGE_ORDER)), medians.values,
                "k--", lw=1.0, zorder=4, alpha=0.5)

        ax.set_xticks(range(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(atom.capitalize() if atom != "sr_ratio" else "S/R Ratio")
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.6, len(STAGE_ORDER) - 0.4)

    plt.tight_layout()
    out = OUT_DIR / f"c_subject_consistency_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [C] Saved → {out}")
    return out



# ---------------------------------------------------------------------------
# Figure D: Electrode topography — group N3 synergy
# ---------------------------------------------------------------------------

def fig_D_electrode_topo():
    """
    For each electrode and subject, compute mean synergy per stage (pooled over
    all lag pairs).  Show a 2-panel figure:
      Left:  simple head schematic with electrode discs coloured by mean N3 synergy
      Right: grouped bar chart showing mean ± SEM for all stages × all electrodes
    """
    subjects = available_subjects("C3")

    records = []
    for subj in subjects:
        for ch in CHANNELS:
            tr = load_timeresolved(subj, ch)
            if tr is None or "synergy" not in tr.columns:
                continue
            for stage in STAGE_ORDER:
                grp = tr[tr["stage"] == stage]["synergy"]
                if len(grp) == 0:
                    continue
                records.append({"subject": subj, "channel": ch,
                                 "stage": stage, "synergy": grp.mean()})

    if not records:
        print("  [E] No data found — skipping.")
        return None

    df = pd.DataFrame(records)

    # --- Left panel: head schematic ---
    n3_by_ch = df[df["stage"] == "N3"].groupby("channel")["synergy"].agg(["mean", "sem"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [1, 1.8]})
    fig.suptitle(
        f"Electrode-level Temporal PID — group (n = {len(subjects)} subjects)\n"
        "Left: N3 synergy; Right: mean ± SEM by stage and electrode", fontsize=12
    )

    ax_head = axes[0]
    ax_head.set_aspect("equal")

    # Draw head circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_head.plot(np.cos(theta), np.sin(theta), "k-", lw=2)
    # Nose
    ax_head.plot([-.1, 0, .1], [.98, 1.12, .98], "k-", lw=2)
    # Ears
    for xi in [-1, 1]:
        ax_head.plot([xi, xi * 1.08, xi * 1.08, xi],
                     [0.15, 0.1, -0.1, -0.15], "k-", lw=2)

    vmin = n3_by_ch["mean"].min()
    vmax = n3_by_ch["mean"].max()
    cmap = plt.cm.get_cmap("hot_r")

    for ch, (x, y) in ELEC_POS.items():
        if ch not in n3_by_ch.index:
            continue
        val  = n3_by_ch.loc[ch, "mean"]
        colour = cmap((val - vmin) / (vmax - vmin + 1e-12))
        circ = plt.Circle((x, y), radius=0.13, color=colour, zorder=3)
        ax_head.add_patch(circ)
        ax_head.text(x, y, ch, ha="center", va="center",
                     fontsize=8, fontweight="bold", zorder=4,
                     color="white",
                     path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_head, fraction=0.04, pad=0.02)
    cbar.set_label("Mean synergy N3 (bits)", fontsize=8)
    ax_head.set_xlim(-1.3, 1.3)
    ax_head.set_ylim(-1.3, 1.3)
    ax_head.axis("off")
    ax_head.set_title("N3 synergy", fontsize=10)

    # --- Right panel: grouped bar chart ---
    ax_bar = axes[1]
    x_base = np.arange(len(CHANNELS))
    bar_width = 0.14
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(STAGE_ORDER))

    for si, stage in enumerate(STAGE_ORDER):
        means = []
        sems  = []
        for ch in CHANNELS:
            grp = df[(df["stage"] == stage) & (df["channel"] == ch)]["synergy"]
            means.append(grp.mean() if len(grp) else np.nan)
            sems.append(grp.sem() if len(grp) > 1 else 0)
        ax_bar.bar(x_base + offsets[si], means, bar_width,
                   color=STAGE_COLORS[stage], label=stage, alpha=0.85,
                   yerr=sems, capsize=2, error_kw={"elinewidth": 0.8})

    ax_bar.set_xticks(x_base)
    ax_bar.set_xticklabels(CHANNELS)
    ax_bar.set_xlabel("Electrode")
    ax_bar.set_ylabel("Mean synergy (bits)")
    ax_bar.set_title("Synergy by electrode and stage", fontsize=10)
    ax_bar.legend(title="Stage", fontsize=8, title_fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / f"d_electrode_topo_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [D] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Figure E: Within-stage synergy vs delta-band power, N3 vs Wake (group)
# ---------------------------------------------------------------------------

def fig_E_synergy_vs_delta():
    """
    2-panel scatter: per-window absolute synergy vs delta-band power for
    N3 (left) and Wake (right), group level.
    Tests whether instantaneous delta amplitude predicts synergy within
    each stage (confound check for delta-power inflation of PID).
    Requires MNE (HAS_MNE = True).
    """
    if not HAS_MNE:
        print("  [E] MNE not available — skipping.")
        return None

    CHAN_TO_PSG = {
        "C3": "PSG_C3", "C4": "PSG_C4",
        "F3": "PSG_F3", "F4": "PSG_F4",
        "O1": "PSG_O1", "O2": "PSG_O2",
    }

    def _delta_fracs(raw, psg_ch, n_windows):
        """Welch delta power per 30-s window (0.5-4 Hz / 0.5-45 Hz)."""
        if psg_ch not in raw.ch_names:
            return None
        idx = raw.ch_names.index(psg_ch)
        data = raw.get_data(picks=[idx])[0]
        fs = raw.info["sfreq"]
        win_samps = int(WINDOW_SEC * fs)
        out = np.full(n_windows, np.nan)
        for w in range(min(n_windows, len(data) // win_samps)):
            seg = data[w * win_samps: (w + 1) * win_samps]
            if np.std(seg) < 1e-15:
                continue
            f_psd, pxx = welch(seg, fs=fs, nperseg=min(int(4 * fs), len(seg)))
            dm = (f_psd >= DELTA_BAND[0]) & (f_psd <= DELTA_BAND[1])
            tm = (f_psd >= DELTA_BAND[0]) & (f_psd <= 45.0)
            if tm.any():
                out[w] = pxx[dm].sum() / (pxx[tm].sum() + 1e-30)
        return out

    rows = []
    for subj in available_subjects("C3"):
        edf_files = sorted(glob.glob(
            str(DATA_DIR / subj / "**" / "*psg_eeg.edf"), recursive=True
        ))
        if not edf_files:
            print(f"  [E] {subj}: no EDF found, skipping")
            continue
        try:
            raw = mne.io.read_raw_edf(edf_files[0], preload=True,
                                       stim_channel=None, verbose=False)
        except Exception as exc:
            print(f"  [E] {subj}: EDF load failed: {exc}")
            continue
        n_windows = int(raw.times[-1] / WINDOW_SEC)

        for ch in CHANNELS:
            psg_ch = CHAN_TO_PSG.get(ch, ch)
            delta_frac = _delta_fracs(raw, psg_ch, n_windows)
            if delta_frac is None:
                continue
            tr = load_timeresolved(subj, ch)
            if tr is None or "synergy" not in tr.columns or "redundancy" not in tr.columns:
                continue
            # per-window mean S/R ratio (pooled over lag pairs) — amplitude-invariant
            tr["sr_ratio"] = tr["synergy"] / (tr["synergy"] + tr["redundancy"])
            win_sr    = tr.groupby("window")["sr_ratio"].mean().reset_index()
            win_stage = tr.groupby("window")["stage"].first().reset_index()
            merged = win_sr.merge(win_stage, on="window")
            merged["delta_frac"] = merged["window"].map(
                lambda w, _df=delta_frac: _df[w] if w < len(_df) else np.nan
            )
            merged = merged.dropna(subset=["delta_frac"])
            merged["subject"] = subj
            merged["channel"] = ch
            rows.append(merged)

        del raw
        import gc; gc.collect()

    if not rows:
        print("  [E] No data collected — skipping.")
        return None

    df = pd.concat(rows, ignore_index=True)
    n_tot = df["stage"].isin(["N3", "Wake"]).sum()

    # ---- 2-panel figure: N3 (left) and Wake (right) ----
    fig, (ax_n3, ax_wake) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    n_subj = df["subject"].nunique()
    n_chan  = df["channel"].nunique()
    _title = ("Within-stage S/R ratio vs delta-band power — confound check  "              f"({n_subj} subjects × {n_chan} ch, n = {n_tot:,} N3+Wake windows)")
    fig.suptitle(_title, fontsize=11)

    MAX_PTS = 3000
    for ax, stage in [(ax_n3, "N3"), (ax_wake, "Wake")]:
        sub = df[df["stage"] == stage].dropna(subset=["sr_ratio", "delta_frac"])
        if sub.empty:
            ax.set_title(stage)
            continue
        rho, pval = stats.spearmanr(sub["delta_frac"], sub["sr_ratio"])
        s_plot = sub.sample(min(MAX_PTS, len(sub)), random_state=42)
        ax.scatter(s_plot["delta_frac"], s_plot["sr_ratio"],
                   s=5, alpha=0.25, color=STAGE_COLORS[stage], rasterized=True)
        m, b = np.polyfit(sub["delta_frac"], sub["sr_ratio"], 1)
        xr = np.array([sub["delta_frac"].min(), sub["delta_frac"].max()])
        ax.plot(xr, m * xr + b, color="black", lw=1.5, linestyle="--")
        p_str = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
        label = (
            f"Spearman ρ = {rho:+.2f}  |  {p_str}"
            f"  |  n = {len(sub):,}"
        )
        ax.text(0.97, 0.95, label, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        ax.set_xlim(left=0, right=np.nanpercentile(sub["delta_frac"], 99))
        ax.set_xlabel("Delta power (fraction of 0.5–45 Hz)")
        ax.set_title(stage, fontsize=12, fontweight="bold",
                     color=STAGE_COLORS[stage])
        ax.grid(alpha=0.3)

    ax_n3.set_ylabel("Mean S/R ratio per window")
    plt.tight_layout()
    out = OUT_DIR / f"e_sr_vs_delta_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [E] Saved → {out}")
    return out


# ---------------------------------------------------------------------------
# Shared data loader for timescale figures F and G
# ---------------------------------------------------------------------------

def _load_timescale_data():
    """
    Load all time-resolved PID CSVs, compute sr_ratio, and return a dict with
    commonly needed aggregates for figs F and G.
    """
    records = []
    subjects = available_subjects("C3")
    for subj in subjects:
        for ch in CHANNELS:
            tr = load_timeresolved(subj, ch)
            if tr is None or "synergy" not in tr.columns or "redundancy" not in tr.columns:
                continue
            tr = tr.copy()
            denom = tr["synergy"] + tr["redundancy"]
            tr["sr_ratio"] = np.where(denom > 0, tr["synergy"] / denom, np.nan)
            tr["subject"] = subj
            tr["channel"] = ch
            records.append(tr)

    if not records:
        return None

    df = pd.concat(records, ignore_index=True)
    atoms     = ["synergy", "redundancy", "sr_ratio"]
    lag1_u    = sorted(df["lag1_min"].unique())
    lag2_u    = sorted(df["lag2_min"].unique())
    n_lag     = len(lag1_u)
    full_idx  = pd.Index(lag2_u, name="lag2_min")
    full_col  = pd.Index(lag1_u, name="lag1_min")

    sc_grp  = (df.groupby(["subject", "channel", "stage", "lag1_min", "lag2_min"])[atoms]
                 .mean().reset_index())
    grp_mean = sc_grp.groupby(["stage", "lag1_min", "lag2_min"])[atoms].mean()

    win_counts    = (df.groupby(["subject", "channel", "stage", "lag1_min", "lag2_min"])
                       .size().reset_index(name="_n"))
    stage_per_obs = win_counts.groupby("stage")["_n"].mean()
    stage_n       = df.groupby("stage").size()
    reliable_stages = [s for s in STAGE_ORDER
                       if s in df["stage"].unique()
                       and stage_per_obs.get(s, 0) >= 10]

    sc_marg    = (sc_grp.groupby(["subject", "channel", "stage", "lag1_min"])[atoms]
                        .mean().reset_index())
    marg_stats = (sc_marg.groupby(["stage", "lag1_min"])[atoms]
                         .agg(["mean", "sem"]).reset_index())

    def _pivot(data, val_col):
        return data.pivot(index="lag2_min", columns="lag1_min",
                          values=val_col).reindex(index=full_idx, columns=full_col)

    return dict(df=df, atoms=atoms, lag1_u=lag1_u, lag2_u=lag2_u, n_lag=n_lag,
                full_idx=full_idx, full_col=full_col, sc_grp=sc_grp,
                grp_mean=grp_mean, stage_n=stage_n, reliable_stages=reliable_stages,
                sc_marg=sc_marg, marg_stats=marg_stats, _pivot=_pivot,
                subjects=subjects)


# ---------------------------------------------------------------------------
# Figure F: Marginal τ₁ profiles — all reliable stages (standalone)
# ---------------------------------------------------------------------------

def fig_F_timescale_marginals():
    """
    3-panel figure (S, R, S/R) of marginal τ₁ profiles per stage.
    Only reliable stages (≥10 windows per subj×ch×lag) are shown.
    S/R panel includes per-stage slope test and N3 vs Wake comparison.
    """
    d = _load_timescale_data()
    if d is None:
        print("  [F] No data — skipping.")
        return None

    atoms          = d["atoms"]
    atom_labels    = ["Synergy (bits)", "Redundancy (bits)", "S/R ratio"]
    atom_titles    = ["Synergy", "Redundancy", "S/R ratio"]
    marg_stats     = d["marg_stats"]
    reliable_stages = d["reliable_stages"]
    stage_n        = d["stage_n"]
    n_lag          = d["n_lag"]
    sc_marg        = d["sc_marg"]

    # Per-(subj, ch, stage) slope of S/R vs τ₁
    slope_rows = []
    for (subj, ch, stage), g in sc_marg.groupby(["subject", "channel", "stage"]):
        if stage not in reliable_stages:
            continue
        g = g.dropna(subset=["sr_ratio"])
        if len(g) < 3:
            continue
        x_s = g["lag1_min"].values.astype(float)
        y_s = g["sr_ratio"].values.astype(float)
        slope, _, _, _, _ = stats.linregress(x_s, y_s)
        slope_rows.append({"subject": subj, "channel": ch,
                            "stage": stage, "slope": slope})
    slope_df = pd.DataFrame(slope_rows) if slope_rows else pd.DataFrame()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("Marginal \u03c4\u2081 profiles across sleep stages "
                 f"(reliable stages only, \u226510 windows/subject/lag)",
                 fontsize=12)

    for col, (atom, ylabel, title) in enumerate(zip(atoms, atom_labels, atom_titles)):
        ax = axes[col]
        for stage in reliable_stages:
            sd = marg_stats[marg_stats["stage"] == stage]
            if sd.empty:
                continue
            x = sd["lag1_min"].values
            y = sd[(atom, "mean")].values
            c = STAGE_COLORS.get(stage, "grey")
            n_win = stage_n.get(stage, 0)

            # Individual subject×channel traces (thin, very transparent — background only)
            subj_sd = sc_marg[sc_marg["stage"] == stage]
            for _, grp in subj_sd.groupby(["subject", "channel"]):
                grp = grp.sort_values("lag1_min")
                ax.plot(grp["lag1_min"].values, grp[atom].values,
                        color=c, lw=0.5, alpha=0.06)

            # Group mean — thick with white outline so it always reads over traces
            ax.plot(x, y, color=c,
                    label=f"{stage} (n\u2248{n_win // n_lag:,}/lag)",
                    lw=3, marker="o", ms=5, zorder=4,
                    path_effects=[pe.Stroke(linewidth=5, foreground="white"),
                                  pe.Normal()])
            if atom == "sr_ratio":
                m, b, _, _, _ = stats.linregress(x.astype(float), y.astype(float))
                ax.plot(x, m * x + b, color=c, lw=1.2, linestyle="--", alpha=0.7,
                        zorder=4)
        ax.set_xlabel("\u03c4\u2081 (min)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)

        if atom == "sr_ratio" and not slope_df.empty:
            annot_lines = []
            stage_slopes = {}
            for stage in reliable_stages:
                sv = slope_df[slope_df["stage"] == stage]["slope"].values
                if len(sv) < 3:
                    continue
                stage_slopes[stage] = sv
                _, p_val = stats.ttest_1samp(sv, 0)
                p_str = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
                annot_lines.append(
                    f"{stage}: \u03b2={sv.mean():+.5f}\u00b1{stats.sem(sv):.5f} ({p_str})"
                )
            for pair in [("N3", "Wake"), ("REM", "Wake")]:
                a, b_ = pair
                if a in stage_slopes and b_ in stage_slopes:
                    sv_a = stage_slopes[a];  sv_b = stage_slopes[b_]
                    n = min(len(sv_a), len(sv_b))
                    _, p2 = stats.ttest_rel(sv_a[:n], sv_b[:n]) if n >= 3 else (np.nan, np.nan)
                    p2_str = "p<0.001" if p2 < 0.001 else f"p={p2:.3f}"
                    annot_lines.append(f"{a} vs {b_}: {p2_str}")
            ax.text(0.98, 0.98, "\n".join(annot_lines),
                    transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
            ax.legend(fontsize=7, loc="lower left")
        else:
            ax.legend(fontsize=7)

    out = OUT_DIR / f"f_timescale_marginals_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [F] Saved \u2192 {out}")
    return out


# ---------------------------------------------------------------------------
# Figure G: Stage comparison heatmaps — N3 vs Wake and REM vs Wake
# ---------------------------------------------------------------------------

def fig_G_stage_comparisons():
    """
    3-row × 3-col figure:
      Row 0: (N3  − Wake) diff heatmaps for S, R, S/R
      Row 1: (REM − Wake) diff heatmaps for S, R, S/R
      Row 2: Std of atom values across ALL reliable stages (including Wake)
             — shows which lag pairs maximally discriminate any stage pair
    """
    d = _load_timescale_data()
    if d is None:
        print("  [G] No data — skipping.")
        return None

    grp_mean        = d["grp_mean"]
    _pivot          = d["_pivot"]
    atoms           = d["atoms"]
    atom_titles     = ["Synergy", "Redundancy", "S/R ratio"]
    reliable_stages = d["reliable_stages"]
    present_stages  = d["df"]["stage"].unique()
    full_idx        = d["full_idx"]
    full_col        = d["full_col"]
    sc_grp          = d["sc_grp"]

    comparisons = [("N3", "Wake"), ("REM", "Wake")]
    comparisons = [(a, b) for a, b in comparisons
                   if a in present_stages and b in present_stages]
    if not comparisons:
        print("  [G] Required stages not present — skipping.")
        return None

    non_wake = [s for s in reliable_stages if s != "Wake" and s in present_stages]

    fig, axes = plt.subplots(3, 3, figsize=(14, 15), constrained_layout=True)
    fig.suptitle(
        "Stage-vs-Wake difference heatmaps + pairwise spread\n"
        "Rows 1\u20132: N3/REM \u2212 Wake  \u2502  Row 3: std across all stages",
        fontsize=11)

    def _stage_piv(stage, atom):
        s = grp_mean.loc[stage][atom].reset_index()
        return _pivot(s, atom)

    def _smart_heatmap(ax, data_piv, title, seq_only=False):
        """Plot a pcolormesh with adaptive colormap. seq_only forces sequential (for std)."""
        vals = data_piv.values[np.isfinite(data_piv.values)]
        if len(vals) == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(title, fontsize=9)
            return
        if seq_only:
            vmin = float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
            cmap = "YlOrRd"
        else:
            all_pos = (vals >= 0).all()
            all_neg = (vals <= 0).all()
            if all_pos:
                vmin = float(np.nanpercentile(vals, 2))
                vmax = float(np.nanpercentile(vals, 98))
                cmap = "Reds"
            elif all_neg:
                vmin = float(np.nanpercentile(vals, 2))
                vmax = float(np.nanpercentile(vals, 98))
                cmap = "Blues_r"
            else:
                v = float(np.nanpercentile(np.abs(vals), 98))
                vmin, vmax, cmap = -v, v, "RdBu_r"
        im = ax.pcolormesh(data_piv.columns, data_piv.index, data_piv.values,
                           cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_title(f"{title}\nmean={vals.mean():.4f}, std={vals.std():.4f}", fontsize=9)
        ax.set_xlabel("\u03c4\u2081 (min)")
        ax.set_ylabel("\u03c4\u2082 (min)")

    def _cell_sig(stage_a, stage_b, atom):
        """
        Per-cell paired t-test across subjects (averaged over channels) for each
        (lag1, lag2) pair, with BH FDR correction across all cells.
        Returns a DataFrame of BH-adjusted p-values pivoted to the same
        shape as the diff heatmap.
        """
        def _subj_piv(stage):
            sub = (sc_grp[sc_grp["stage"] == stage]
                   .groupby(["subject", "lag1_min", "lag2_min"])[atom]
                   .mean().reset_index())
            return sub
        pa = _subj_piv(stage_a)
        pb = _subj_piv(stage_b)
        merged = pa.merge(pb, on=["subject", "lag1_min", "lag2_min"],
                          suffixes=("_a", "_b"))
        p_rows = []
        for (l1, l2), g in merged.groupby(["lag1_min", "lag2_min"]):
            da = g[f"{atom}_a"].values
            db = g[f"{atom}_b"].values
            if len(da) >= 3:
                _, p = stats.ttest_rel(da, db)
            else:
                p = np.nan
            p_rows.append({"lag1_min": l1, "lag2_min": l2, "p": p})
        if not p_rows:
            return None
        p_df = pd.DataFrame(p_rows)

        # BH FDR correction across all cells
        finite_mask = p_df["p"].notna()
        p_arr = p_df.loc[finite_mask, "p"].values
        n = len(p_arr)
        order = np.argsort(p_arr)
        adj = np.minimum.accumulate((p_arr[order] * n / (np.arange(n) + 1))[::-1])[::-1]
        adj = np.clip(adj[np.argsort(order)], 0, 1)
        p_df.loc[finite_mask, "p"] = adj

        return p_df.pivot(index="lag2_min", columns="lag1_min",
                          values="p").reindex(index=full_idx, columns=full_col)

    def _stars(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    def _overlay_sig(ax, p_piv):
        """Print significance stars at the centre of each cell."""
        if p_piv is None:
            return
        for j, lag2 in enumerate(p_piv.index):
            for i, lag1 in enumerate(p_piv.columns):
                p = p_piv.iloc[j, i]
                s = _stars(p)
                if s:
                    ax.text(lag1, lag2, s, ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")

    # Rows 0–1: individual comparisons (N3-Wake, REM-Wake)
    for row, (stage_a, stage_b) in enumerate(comparisons):
        for col, (atom, title) in enumerate(zip(atoms, atom_titles)):
            ax = axes[row, col]
            try:
                diff = _stage_piv(stage_a, atom).subtract(_stage_piv(stage_b, atom))
                _smart_heatmap(ax, diff, f"{stage_a} \u2212 {stage_b}: {title}")
                p_piv = _cell_sig(stage_a, stage_b, atom)
                _overlay_sig(ax, p_piv)
            except Exception as exc:
                ax.text(0.5, 0.5, str(exc), transform=ax.transAxes,
                        ha="center", fontsize=7)
                ax.set_title(f"{stage_a} \u2212 {stage_b}: {title}", fontsize=9)

    def _cell_sig_kw(atom):
        """
        Per-cell Friedman (or Kruskal-Wallis across stages) test using per-subject
        means for each (lag1, lag2).  BH FDR correction is applied across all cells.
        Returns a DataFrame of BH-adjusted p-values pivoted over the lag grid.
        """
        valid_stages = [s for s in reliable_stages if s in grp_mean.index]
        if len(valid_stages) < 2:
            return None

        # per-subject (averaged over channels) per-stage per-cell mean
        sub_vals = (sc_grp[sc_grp["stage"].isin(valid_stages)]
                    .groupby(["subject", "stage", "lag1_min", "lag2_min"])[atom]
                    .mean().reset_index())

        raw_p = []
        for (l1, l2), g in sub_vals.groupby(["lag1_min", "lag2_min"]):
            groups = [g[g["stage"] == s][atom].values for s in valid_stages]
            # Only keep groups with ≥3 observations
            groups = [gr for gr in groups if len(gr) >= 3]
            if len(groups) >= 2:
                try:
                    _, p = stats.kruskal(*groups)
                except Exception:
                    p = np.nan
            else:
                p = np.nan
            raw_p.append({"lag1_min": l1, "lag2_min": l2, "p": p})

        if not raw_p:
            return None
        p_df = pd.DataFrame(raw_p)

        # BH FDR across all cells
        finite_mask = p_df["p"].notna()
        p_arr = p_df.loc[finite_mask, "p"].values
        order = np.argsort(p_arr)
        n = len(p_arr)
        adj = np.minimum.accumulate((p_arr[order] * n / (np.arange(n) + 1))[::-1])[::-1]
        adj = np.clip(adj[np.argsort(order)], 0, 1)
        p_df.loc[finite_mask, "p"] = adj

        return p_df.pivot(index="lag2_min", columns="lag1_min",
                          values="p").reindex(index=full_idx, columns=full_col)

    # Row 2: std across ALL reliable stages — unbiased pairwise spread
    all_stages_list = ", ".join(reliable_stages)
    for col, (atom, title) in enumerate(zip(atoms, atom_titles)):
        ax = axes[2, col]
        try:
            stage_arrays = np.stack(
                [_stage_piv(s, atom).values
                 for s in reliable_stages if s in grp_mean.index],
                axis=0)  # shape: (n_stages, n_lag2, n_lag1)
            std_vals = np.nanstd(stage_arrays, axis=0)
            std_piv = pd.DataFrame(std_vals,
                                   index=full_idx, columns=full_col)
            _smart_heatmap(ax, std_piv,
                           f"Std across [{all_stages_list}]: {title}",
                           seq_only=True)
            p_piv_kw = _cell_sig_kw(atom)
            _overlay_sig(ax, p_piv_kw)
        except Exception as exc:
            ax.text(0.5, 0.5, str(exc), transform=ax.transAxes,
                    ha="center", fontsize=7)
            ax.set_title(f"Std across all stages: {title}", fontsize=9)

    out = OUT_DIR / f"g_stage_comparisons_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [G] Saved \u2192 {out}")
    return out


# ---------------------------------------------------------------------------
# Figure H: Pairwise stage mean-diff summary heatmaps
# ---------------------------------------------------------------------------

def fig_H_pairwise_summary():
    """
    For each atom (S, R, S/R), compute the mean difference across all
    (τ1, τ2) lag pairs for every pair of reliable stages, with paired
    t-tests across subjects and BH FDR correction per atom.
    Cell (i,j) shows: mean diff + significance stars (BH-corrected p).
    """
    d = _load_timescale_data()
    if d is None:
        print("  [H] No data — skipping.")
        return None

    sc_grp          = d["sc_grp"]
    reliable_stages = d["reliable_stages"]
    atoms           = d["atoms"]
    atom_titles     = ["Synergy", "Redundancy", "S/R ratio"]

    n_stages = len(reliable_stages)
    if n_stages < 2:
        print("  [H] Not enough stages — skipping.")
        return None

    # Per-subject mean per stage (averaged over channel × lag pairs)
    subj_stage = (sc_grp.groupby(["subject", "stage"])[atoms]
                        .mean())

    def _bh_correct(raw_pvals):
        """Benjamini-Hochberg FDR correction, returns adjusted p-values."""
        n = len(raw_pvals)
        if n == 0:
            return np.array([])
        arr = np.array(raw_pvals, dtype=float)
        order = np.argsort(arr)
        sorted_p = arr[order]
        adj = np.minimum.accumulate((sorted_p * n / (np.arange(n) + 1))[::-1])[::-1]
        adj = np.clip(adj, 0, 1)
        result = np.empty(n)
        result[order] = adj
        return result

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    # Build matrices: mean_diff, p_raw, p_adj per atom
    mean_mats = {}
    padj_mats = {}
    for atom in atoms:
        mean_mat = np.full((n_stages, n_stages), np.nan)
        praw_mat = np.full((n_stages, n_stages), np.nan)

        # Collect all off-diagonal raw p-values for BH correction
        pairs = [(i, j) for i in range(n_stages)
                         for j in range(n_stages) if i != j]
        raw_ps = []
        for i, j in pairs:
            si, sj = reliable_stages[i], reliable_stages[j]
            try:
                vi = subj_stage.loc[(slice(None), si), atom].values
                vj = subj_stage.loc[(slice(None), sj), atom].values
            except KeyError:
                raw_ps.append(np.nan)
                continue
            # Match by subjects present in both stages
            si_idx = subj_stage.index.get_level_values("stage") == si
            sj_idx = subj_stage.index.get_level_values("stage") == sj
            subjs_i = set(subj_stage.index.get_level_values("subject")[si_idx])
            subjs_j = set(subj_stage.index.get_level_values("subject")[sj_idx])
            common = sorted(subjs_i & subjs_j)
            if len(common) < 3:
                raw_ps.append(np.nan)
                continue
            vi_m = subj_stage.loc[(common, si), atom].values
            vj_m = subj_stage.loc[(common, sj), atom].values
            mean_mat[i, j] = float(np.nanmean(vi_m - vj_m))
            _, p = stats.ttest_rel(vi_m, vj_m)
            raw_ps.append(float(p))

        # BH correct only finite p-values
        finite_mask = np.isfinite(raw_ps)
        adj_ps = np.full(len(raw_ps), np.nan)
        if finite_mask.any():
            adj_ps[finite_mask] = _bh_correct(
                np.array(raw_ps)[finite_mask])

        for k, (i, j) in enumerate(pairs):
            praw_mat[i, j] = raw_ps[k]
            padj_mats.setdefault(atom, np.full((n_stages, n_stages), np.nan))
            padj_mats[atom][i, j] = adj_ps[k]
        mean_mats[atom] = mean_mat

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(
        "Pairwise mean differences across reliable sleep stages\n"
        "(paired t-test across subjects, BH-corrected; cell (i,j) = stage_i \u2212 stage_j)",
        fontsize=11)

    for ax, atom, title in zip(axes, atoms, atom_titles):
        mat  = mean_mats[atom]
        padj = padj_mats.get(atom, np.full((n_stages, n_stages), np.nan))

        off_diag  = mat[~np.eye(n_stages, dtype=bool)]
        finite_od = off_diag[np.isfinite(off_diag)]
        if len(finite_od) == 0:
            ax.set_title(title); continue
        all_pos = (finite_od >= 0).all()
        all_neg = (finite_od <= 0).all()
        if all_pos:
            vmin = float(np.nanpercentile(finite_od, 2))
            vmax = float(np.nanpercentile(finite_od, 98))
            cmap = "Reds"
        elif all_neg:
            vmin = float(np.nanpercentile(finite_od, 2))
            vmax = float(np.nanpercentile(finite_od, 98))
            cmap = "Blues_r"
        else:
            v = float(np.nanpercentile(np.abs(finite_od), 98))
            vmin, vmax, cmap = -v, v, "RdBu_r"

        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set_xticks(range(n_stages))
        ax.set_yticks(range(n_stages))
        ax.set_xticklabels(reliable_stages, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(reliable_stages, fontsize=9)
        ax.set_xlabel("Stage j")
        ax.set_ylabel("Stage i")
        ax.set_title(title, fontsize=11)

        for i in range(n_stages):
            for j in range(n_stages):
                normed = ((mat[i, j] - vmin) / (vmax - vmin + 1e-12)
                          if np.isfinite(mat[i, j]) else 0.5)
                tc = "white" if (normed < 0.30 or normed > 0.75) else "black"
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=9, color=tc)
                else:
                    diff_str = (f"{mat[i, j]:+.4f}"
                                if np.isfinite(mat[i, j]) else "")
                    p_val = padj[i, j]
                    sig = _stars(p_val) if np.isfinite(p_val) else ""
                    p_str = (f"p={p_val:.3f}" if (np.isfinite(p_val) and p_val >= 0.001)
                             else ("p<0.001" if np.isfinite(p_val) else ""))
                    ax.text(j, i, f"{diff_str}\n{sig}\n{p_str}",
                            ha="center", va="center",
                            fontsize=6.5, color=tc, linespacing=1.4)

    out = OUT_DIR / f"h_pairwise_summary_{CHASH}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [H] Saved \u2192 {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {OUT_DIR}")

    fig_C_subject_consistency()
    fig_D_electrode_topo()
    fig_E_synergy_vs_delta()
    fig_F_timescale_marginals()
    fig_G_stage_comparisons()
    fig_H_pairwise_summary()

    print("Done.")
