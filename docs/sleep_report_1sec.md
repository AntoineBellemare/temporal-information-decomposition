# Temporal PID of EEG Sleep — 1-second Window Pass

This is the **short-window companion** to [EEG_SLEEP_TEMPORAL_PID.md](EEG_SLEEP_TEMPORAL_PID.md). The pipeline, atom definitions, and stage filter are unchanged; what differs is the temporal scale of analysis and the responses to specific reviewer feedback.

---

## What changed vs. the 30-s broadband report

| Change | 30-s pass | 1-s pass | Why |
|---|---|---|---|
| Window duration `WINDOW_SEC` | 30 s | **1 s** | Pedro: 30 s is a very long time in EEG; 30-s block shuffling preserves nearly all the structure that matters. Dropping to 1 s pulls the analysis into the EEG-relevant range. |
| Maximum lag `MAX_LAG_MIN` | 10 min | **0.5 min** (= 30 s) | At 1-s windows, lag pairs of seconds–tens-of-seconds are the natural unit; spanning minutes at 1-s resolution would explode the pair count without adding interpretable structure. |
| Quantile bins `N_BINS` | 6 | **4** | Each PID joint has `n_bins³` cells. At 1-s × 256 Hz = 256 samples/window, n_bins=6 → 216 cells (≈ 1.2 samples/cell). n_bins=4 → 64 cells (4 samples/cell), the regime where the empirical joint is reasonable. |
| Stage filter `CONTINUOUS_STAGE_FILTER` | False | **True** | Hypnogram is scored in 30-s blocks. With lags up to 30 s, a triplet can straddle a stage boundary; the continuous filter rejects any triplet whose interior windows are not all the same stage. |
| Frequency bands | 5 bands (δ θ α σ β) | **broadband** | First-pass; band-resolved 1-s pass to follow. |
| AR(1) baseline | one global fit | **stage-conditional** | Pedro flagged that across-stage spectral differences contaminate "excess vs AR(1)". The AR(1) is now fit per stage, so excess is isolated *within* a stage. |
| Block-permutation null | 30-s blocks | **1-s blocks** | Direct response to Pedro: at this resolution the block shuffle actually destroys the linear autocorrelation of interest. |
| Subject duration | 5 h | **3.5 h** | Sufficient for two NREM–REM cycles and per-stage AR(1) fits; trims wall-time. |
| `TARGET_STEP` | n/a | **3** | Strides the time-resolved PID target axis (one estimate every 3 s). Each estimate still uses the full 1-s window — the stride only thins the time axis. Global PID, AR(1) and block-perm are unaffected. |
| PID backend | dit (`PID_MMI`) | **numpy closed-form MMI** | Mathematically identical (verified to 1e-15); ~500–1500× faster per call at 256-sample joints, which made the run feasible. dit version retained as `compute_pid_from_arrays_dit` for forensic checks. |

---

## Method

The signal (Bitbrain ds005555, sub-1, PSG channels F3, F4, C3, C4, O1, O2, 256 Hz) is bandpass-filtered (0.5–60 Hz) and notch-filtered (50/60 Hz), then divided into **non-overlapping 1-s windows**. Each window is independently quantile-binned into 4 levels — this yields a within-window amplitude-invariant symbolic stream that removes any DC offset or slow drift on the window timescale.

For a target window at time *t*, two source windows are selected at lags τ₁ and τ₂ seconds into the past (1 s ≤ τ₁ < τ₂ ≤ 30 s, in 1-s steps). The three aligned symbol vectors form a **temporal triplet** *(s₁, s₂, target)* of L = 256 co-occurring symbols. The empirical joint distribution *p(s₁, s₂, target)* is computed by `bincount` over `64 = N_b³` cells, and **PID-MMI** is then evaluated in closed form:

- *R* = min(*I*(S₁; T), *I*(S₂; T))
- *U_i* = *I*(S_i; T) − *R*
- *S* = *I*({S₁, S₂}; T) − *I*(S₁; T) − *I*(S₂; T) + *R*

### Stage filter

A triplet is retained only if **every window in the span from source₂ to target** shares the same scored sleep stage. At 1-s windows and τ₂ up to 30 s, this guarantees the triplet lies entirely within one 30-s scored bout.

### Stage-conditional AR(1) baseline

For each sleep stage with ≥ 20 windows, an AR(1) coefficient φ and noise σ are fit on that stage's discretised windows alone. A synthetic AR(1) stream is generated with those parameters and its PID matrix computed; this serves as the **stage-matched linear baseline**. Excess = Actual − stage-AR(1) is then a within-stage measure of non-linear / higher-order temporal structure.

### Block-permutation null

100 permutations of the 1-s window order, with PID recomputed on the shuffled stream. At Δt = 1 s the block shuffle effectively destroys all linear autocorrelation, so atoms surviving the shuffle reflect structure on the sub-second / within-window scale.

---

## Configuration

| Parameter | Value |
|---|---|
| `WINDOW_SEC` | 1 |
| `MAX_LAG_MIN` | 0.5 (= 30 s) |
| `N_BINS` | 4 |
| `DISCRETIZE_PER_WINDOW` | True |
| `CONTINUOUS_STAGE_FILTER` | True |
| `BANDS` | None (broadband) |
| `DURATION_HOURS` | 3.5 |
| `TARGET_STEP` | 3 |

Cache hash (filename suffix): see `results/pid/eeg_sleep/params_<hash>.json` (one is written per run; the 1-s pass is the most recent entry).

---

## How to read this report

Each figure is produced **per channel**, written to `results/pid/eeg_sleep/<channel>/<plot_name>_<hash>.png`. The descriptions below refer to the C3 instance as a canonical example; the same plot exists for F3, F4, C4, O1, O2. Cross-electrode summaries are written once per config to `results/pid/eeg_sleep/cross_electrode_comparison_<hash>.png`.

---

## Figures

### Plot 1 — Global PID Matrix
**Files:** `<ch>/global_pid_matrix_<hash>.png`

Four τ₁ × τ₂ triangular heatmaps (redundancy, synergy, unique₁, unique₂) computed over **all valid triplets in the entire recording**. The y-axis is τ₁ (shorter lag, in minutes — at this pass 1 / 60 = 0.017 min ≈ 1 s up to 0.5 min = 30 s), the x-axis τ₂. Reveals which lag combinations carry redundant vs. synergistic information about the present.

### Plot 2 — Hypnogram + PID Time Series
**Files:** `<ch>/combined_timeseries_<hash>.png`

Top: hypnogram. Below: mean redundancy / synergy / total unique per target window (averaged across all lag pairs), as a time series with stage-shaded background. Tracks the evolution of temporal information structure across the night at the 3-s time-axis resolution set by `TARGET_STEP`.

### Plot 3 — Time × Lag Heatmaps
**Files:** `<ch>/time_lag_heatmaps_<hash>.png`

Three heatmaps (redundancy, synergy, unique₁): x = time (min), y = τ₂ (τ₁ fixed at the shortest available lag). A stage colour bar is drawn above. Shows how the timescale profile of each atom evolves over the recording.

### Plot 4 — Stage Comparison with Statistics
**Files:** `<ch>/stage_comparison_<hash>.png`

Boxplots of per-window mean PID atoms grouped by sleep stage (Wake / N1 / N2 / N3 / REM): redundancy, synergy, total unique, and **S/R ratio**. Restricted to the common lag-pair range shared by all stages. Includes a Kruskal–Wallis test and pairwise Mann–Whitney U brackets. The S/R ratio panel is the primary stage discriminator robust to total-MI shifts (Pedro's recommended summary).

### Plot 5 — Lag-Difference Heatmap
**Files:** `<ch>/lagdiff_heatmap_<hash>.png`

x = time, y = lag difference (τ₂ − τ₁), colour = mean atom. Collapses the 2-D lag space into one dimension (timescale gap) and shows whether certain timescale separations carry more information at specific times.

### Plot 6 — Global PID: Actual vs AR(1)
**Files:** `<ch>/global_pid_vs_ar1_<hash>.png`

3 × 4 grid of τ₁ × τ₂ heatmaps. Top row: actual PID. Middle row: AR(1) baseline (here, the stage-conditional AR(1) matrices averaged across stages — for stage-resolved excess see Plot 7b). Bottom row: excess (actual − AR(1)), diverging colourmap centred at zero. Shows whether the observed PID structure exceeds a linear-Gaussian baseline matched to the data's autocorrelation.

### Plot 7 — Stage Comparison: Actual vs AR(1)
**Files:** `<ch>/stage_comparison_vs_ar1_<hash>.png`

Top row: per-stage boxplots comparing actual (filled) vs stage-matched AR(1) (grey) for redundancy, synergy, total unique. Bottom row: excess per stage with Kruskal–Wallis statistic. Confirms whether non-linear excess differs across stages.

### Plot 7b — Per-stage PID Excess vs Stage-Conditional AR(1) — NEW
**Files:** `<ch>/global_pid_vs_ar1_per_stage_<hash>.png`

Grid with **one row per sleep stage** and four columns (R, S, U₁, U₂). Each cell is the per-stage τ₁ × τ₂ excess matrix (Actual − stage-AR(1)) with diverging colourmap centred at zero and per-atom shared `vmax` across stages so colours are directly comparable.

A final row carries the **double-dissociation strip**: one horizontal bar plot per atom showing Cohen's *d* of per-window excess for every stage pair. Pairs whose *d*-vector has opposite signs across atoms are highlighted in red and listed in the figure suptitle as double-dissociation candidates. This figure was added to address Pedro's question about the dissociation seen in Fig 7 of the prior report.

### Plot 8 — Autocorrelation vs PID
**Files:** `<ch>/autocorrelation_vs_pid_<hash>.png`

Top: cross-window autocorrelation R(τ) heatmap (time × lag). Middle: synergy heatmap at τ₁ = shortest lag (same axes). Bottom: scatter of mean autocorrelation vs mean synergy per window, coloured by stage. Reveals whether PID changes merely track linear autocorrelation or capture something beyond it.

### Plot 9 — S/R Ratio vs Autocorrelation Diagnostic
**Files:** `<ch>/sr_ratio_vs_autocorr_<hash>.png`

Three panels: (1) S/R ratio vs mean autocorrelation, coloured by stage with regression lines; (2) S/R ratio boxplot per stage; (3) autocorrelation boxplot per stage. If stages show different S/R ratios at similar autocorrelation levels, the PID structure is genuinely different — not just an autocorrelation confound. The S/R ratio is the headline measure Pedro endorsed.

### Plot 10 — PID Atom Matrices Per Stage
**Files:** `<ch>/pid_per_stage_matrix_<hash>.png`

Separate τ₁ × τ₂ synergy and redundancy heatmaps for each sleep stage. Columns are stages; top row synergy, bottom row redundancy. Reveals whether stages have distinct *timescale fingerprints*, independent of overall magnitude.

### Plot 11 — Atom Fractions
**Files:** `<ch>/atom_fractions_<hash>.png`

Left: total mutual information per stage (boxplot + Kruskal–Wallis). Right: stacked bar chart of the *fraction* of total MI carried by redundancy / synergy / unique₁ / unique₂ per stage. Restricted to the common lag-pair range. This is the "compositional" view that partially addresses Pedro's point 2 (the sum of atoms ∝ past–future MI ∝ 1/entropy-rate): even if magnitudes co-move, the composition need not.

### Plot 12 — NREM→REM Cycle Dynamics
**Files:** `<ch>/nrem_rem_cycles_<hash>.png`

PID atoms aligned to REM onset (time = 0), averaged across all NREM→REM transitions in the recording (±15 min). Mean ± SEM. At 3.5 h there are typically 1–2 transitions; the SEM bands will be wider than in the 5 h pass.

### Plot 13 — Effect Sizes (Cohen's d)
**Files:** `<ch>/effect_sizes_<hash>.png`

Four heatmaps (R, S, total U, S/R ratio) showing Cohen's *d* between all stage pairs. Cells annotated with the d value; asterisks mark p < 0.05 (Mann-Whitney U). Diverging colourmap. Highlights which stage contrasts carry the largest information-theoretic differences.

### Plot 14 — Block-permutation Significance
**Files:** `<ch>/block_permutation_<hash>.png`

p-value heatmaps per atom from the 1-s block-permutation null. Cells where the observed atom exceeds the 95th percentile of the null are starred. Crucially, the block size now matches the window size, so the shuffle destroys nearly all the temporal structure of interest — surviving signal is genuinely non-trivial.

### Plot 15 — Optimal Timescales
**Files:** `<ch>/optimal_timescales_<hash>.png`

Per-stage profile of mean atom vs lag-pair separation, with marker on the lag pair that maximises each atom. Compact summary of *where* in lag-space each stage's information lives.

### Plot 16 — Stage Distribution Analysis
**Files:** `<ch>/stage_distributions_<hash>.png`

Per-stage distributions of atom values (and S/R ratio), with Levene's test for equality of variance. Differences in *spread* are often as informative as differences in *mean*.

### Cross-electrode Comparison
**File:** `cross_electrode_comparison_<hash>.png`

Topographic summary across the 6 PSG channels. Generated automatically when ≥ 2 channels have results for the same config hash.

---

## Outstanding items (per the running feedback)

| # | Pedro | Status |
|---|---|---|
| 1 | 30-s block shuffle uninformative | Addressed by 1-s windows ⇒ 1-s block shuffle |
| 2 | All atoms increase with MI | Partial via Plot 11 (compositions). Mediano-lab entropy-rate-normalised PID still pending. |
| 3 | S/R ratio robust | Promoted to the primary stage panel in Plots 4, 9 |
| 4 | "Unaffected by slow amplitude drifts" claim | Soft-fixed in the per-window quantile-binning note above; AM-sine empirical demo still pending. |
| 5 | Double dissociation in Fig 7 | Plot 7b added (per-stage excess + opposite-sign d strip) |

---

## Reproduce

```
python scripts/pid/eeg_sleep_compute.py     # all 6 channels, ~25 min each (C3 cached after first run)
python scripts/pid/eeg_sleep_plot.py        # all 16+1 figures × 6 channels + cross-electrode
```

Per-plot caching: each PNG has its own existence check, so a re-run only renders missing or `--force`-requested figures.
