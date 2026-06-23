# Temporal PID of EEG Sleep

## Method

A continuous EEG signal (PSG channel C3, 256 Hz) from an overnight polysomnography recording is analysed for temporal information structure across minute-scale lags.

### Preprocessing

The signal is divided into **non-overlapping windows** of duration Δt. Each window is independently discretized into N_b levels via **quantile binning** (equal-probability bins), producing an amplitude-invariant symbolic sequence. This removes slow power drifts — the dominant driver of long-range linear correlation — so that information-theoretic measures capture temporal *pattern* rather than amplitude covariation.

### Temporal Triplet Construction

For a target window at time *t*, two source windows are selected at lags τ₁ and τ₂ minutes into the past. The three aligned discrete vectors form a **temporal triplet** *(s₁, s₂, target)*. Sample-by-sample alignment across windows yields L = Δt × f_s co-occurring symbols per triplet.

### PID Decomposition

The empirical joint distribution *p(s₁, s₂, target)* over N_b³ states is estimated by counting co-occurrences. **Partial Information Decomposition** (PID, using the Minimum Mutual Information measure) then decomposes the total predictive information *I({s₁, s₂}; target)* into four non-negative atoms:

| Atom | Meaning |
|------|---------|
| **Redundancy** | Information about the target that *both* past timescales provide independently |
| **Synergy** | Information available *only* when both past timescales are considered jointly |
| **Unique₁** | Information exclusively carried by source 1 (shorter lag) |
| **Unique₂** | Information exclusively carried by source 2 (longer lag) |

### Stage Filtering

Sleep stage is determined per window from scored annotations. A triplet is **retained only if all three windows** (target and both sources) share the same sleep stage, ensuring that stage-stratified results reflect within-stage temporal dynamics rather than cross-boundary transitions. An optional stricter mode (`CONTINUOUS_STAGE_FILTER`) requires every window in the span from source₂ to target to be the same stage.

Because stages have different typical bout lengths (e.g., N3 bouts can exceed 30 min while N1 rarely exceeds 3 min), not all lag pairs are available for all stages. This creates a **coverage bias**: averaging across all available lag pairs would inflate values for short-bout stages (which only contribute high-information short lags). To correct this, all cross-stage comparisons (Plots 4, 9, 14, 16) restrict to the **common lag range** — the intersection of lag pairs present in every stage — so that per-window means are computed over identical timescale subsets.

### AR(1) Baseline

A sample-level AR(1) process with matched autocorrelation coefficient and marginal distribution is generated as a linear-Gaussian baseline. Any PID in excess of the AR(1) baseline reflects nonlinear or higher-order temporal structure.

---

## Figures

### Plot 1 — Global PID Matrix

**File:** `global_pid_matrix.png`

Four lag₁ × lag₂ triangular heatmaps (redundancy, synergy, unique₁, unique₂) computed over the entire recording. Each cell shows the PID atom value for that pair of lags. Reveals the overall timescale fingerprint of temporal information — which lag combinations carry redundant vs. synergistic information about the present.

---

### Plot 2 — Hypnogram + PID Time Series

**File:** `hypnogram_pid_timeseries.png`

Four-panel figure. Top: hypnogram (sleep stage over time). Below: mean redundancy, synergy, and total unique information per window (averaged across all lag pairs), plotted as time series with sleep-stage background shading. Shows how temporal information structure evolves across the night and tracks sleep architecture.

---

### Plot 3 — Time × Lag Heatmaps

**File:** `time_lag_heatmaps.png`

Three heatmaps (redundancy, synergy, unique₁) with x = time, y = lag₂ (lag₁ fixed at 1 min). A sleep-stage colour bar is drawn above. Reveals how the timescale profile of each atom evolves over the recording — for example, whether deep sleep shows synergy at different lag scales than REM.

---

### Plot 4 — Stage Comparison with Statistics

**File:** `stage_comparison.png`

Boxplots of per-window mean PID atoms grouped by sleep stage: redundancy, synergy, total unique, and synergy/redundancy ratio. **Restricted to common lag pairs** shared by all stages to avoid coverage bias. Each panel includes:
- **Kruskal-Wallis** test (H-statistic and p-value in title)
- **Pairwise Mann-Whitney U** significance brackets for the top 4 most significant pairs

This is the primary plot for answering whether PID structure differs across sleep stages.

---

### Plot 5 — PID Distribution Over Time

**File:** `pid_distribution_over_time.png`

Percentile bands for each PID atom over time: for each window, the spread across all lag pairs is shown as median + IQR + 10th–90th percentile. Sleep-stage background shading is overlaid. Shows not just the central tendency but the *variability* of temporal information across timescales within each window.

---

### Plot 6 — Lag-Difference Heatmap

**File:** `lagdiff_heatmap.png`

Heatmap with x = time, y = lag difference (lag₂ − lag₁), colour = metric. Each cell is the mean metric across all lag pairs with that separation. Collapses the two-dimensional lag space into one dimension (timescale gap), making it easier to see whether certain timescale separations carry more information at specific times.

---

### Plot 7 — Global PID: Actual vs AR(1)

**File:** `global_pid_vs_ar1.png`

3 × 4 grid of lag₁ × lag₂ heatmaps: top row = actual PID, middle row = AR(1) baseline (averaged across the per-stage fits), bottom row = excess (actual − AR(1)). Each column is one atom. The excess row uses a diverging colourmap centred at zero. Shows whether the observed PID structure exceeds what a linear-Gaussian baseline matched to the data's marginal autocorrelation would produce. For stage-resolved excess, see Plot 7b.

---

### Plot 7b — Per-stage PID Excess vs Stage-Conditional AR(1)

**File:** `global_pid_vs_ar1_per_stage.png`

Grid with one row per sleep stage and four columns (redundancy, synergy, unique₁, unique₂). Each cell is the lag₁ × lag₂ excess matrix (Actual − stage-matched AR(1)) for that stage and that atom, with a diverging colourmap centred at zero and a per-atom shared vmax across stages so colours are comparable.

A final row carries the **double-dissociation strip**: one bar plot per atom showing Cohen's *d* of the per-window excess for every stage pair. Stage pairs whose *d* vector has opposite signs across atoms (e.g., *d* > 0 for redundancy but *d* < 0 for synergy) are highlighted in red and listed in the suptitle. This is the quantitative version of the "double dissociation" Pedro flagged in Fig 7: it answers *where* in stage space the dissociation lives and which atom contrast carries it.

---

### Plot 8 — Time Series vs AR(1)

**File:** `timeseries_vs_ar1.png`

Hypnogram on top, then three panels (redundancy, synergy, total unique) showing the actual PID time series, the AR(1) baseline (grey fill), and the excess (dashed). Sleep-stage shading is overlaid. Shows the temporal evolution of nonlinear temporal structure across the night.

---

### Plot 9 — Stage Comparison: Actual vs AR(1)

**File:** `stage_comparison_vs_ar1.png`

Top row: side-by-side boxplots per stage comparing actual (filled) vs AR(1) (grey) for redundancy, synergy, and total unique. Bottom row: excess (actual − AR(1)) per stage. **Restricted to common lag pairs.** Shows whether nonlinear temporal structure beyond linear prediction differs across sleep stages.

---

### Plot 10 — Autocorrelation vs PID

**File:** `autocorrelation_vs_pid.png`

Three-panel figure. Top: cross-window autocorrelation R(τ) heatmap (time × lag). Middle: synergy heatmap at lag₁ = 1 min (same axes for comparison). Bottom: scatter of mean autocorrelation vs. mean synergy per window, coloured by stage. Reveals whether PID changes merely track linear autocorrelation or capture something beyond it.

---

### Plot 11 — S/R Ratio vs Autocorrelation Diagnostic

**File:** `sr_ratio_vs_autocorr.png`

Key diagnostic figure. Three panels: (1) scatter of synergy/redundancy ratio vs mean autocorrelation, coloured by stage with regression lines; (2) S/R ratio boxplot per stage; (3) autocorrelation boxplot per stage. If stages show different S/R ratios at similar autocorrelation levels, the PID structure is genuinely different — not just an autocorrelation confound.

---

### Plot 12 — PID Atom Matrices Per Stage

**File:** `pid_per_stage_matrix.png`

Separate lag₁ × lag₂ synergy and redundancy heatmaps for each sleep stage. Each column is one stage; top row = synergy, bottom row = redundancy. The number of contributing windows is shown in the title. Reveals whether stages have different *timescale fingerprints* — not just different magnitudes, but different lag-pair profiles.

---

### Plot 13 — Timescale Decay

**File:** `timescale_decay.png`

Mean PID atom as a function of lag separation (lag₂ − lag₁), one line per stage with SEM error bands. Three panels: redundancy, synergy, and S/R ratio. Shows how fast information decays with increasing timescale gap and whether stages differ in their decay rate — e.g., does N3 (slow-wave sleep) maintain information over longer separations?

---

### Plot 14 — Atom Fractions

**File:** `atom_fractions.png`

Left: total mutual information per stage (boxplot with Kruskal-Wallis statistic). Right: stacked bar chart showing the *fraction* of total MI that is redundancy, synergy, unique₁, and unique₂ per stage. **Restricted to common lag pairs.** Reveals whether stages differ in *how* they represent information (composition), not just *how much* (magnitude).

---

### Plot 15 — NREM→REM Cycle Dynamics

**File:** `nrem_rem_cycles.png`

PID atoms aligned to REM onset (time = 0), averaged across all NREM→REM transitions in the recording (±15 min window). Mean ± SEM for redundancy, synergy, and total unique. Shows whether there is a stereotyped trajectory of temporal information dynamics at sleep stage transitions — a potential signature of the NREM→REM switching process.

---

### Plot 16 — Effect Sizes (Cohen's d)

**File:** `effect_sizes.png`

Four heatmaps (redundancy, synergy, total unique, S/R ratio) showing Cohen's d between all stage pairs. Cells are annotated with the d value; asterisks mark pairs significant at p < 0.05 (Mann-Whitney U). Uses a diverging colourmap. **Restricted to common lag pairs.** Highlights which stage contrasts have the largest information-theoretic differences and provides standardised effect sizes for comparison.

---

## Configuration (current pass — short windows)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SEC` | 1 | Window duration in seconds — addresses Pedro's point that 30-s blocks are too long for EEG |
| `MAX_LAG_MIN` | 0.5 | Maximum lag in minutes (= 30 s) |
| `N_BINS` | 4 | Number of quantile bins (reduced from 6 because each 1-s window has only `fs·Δt` samples for the joint distribution) |
| `DISCRETIZE_PER_WINDOW` | True | Per-window quantile binning |
| `CONTINUOUS_STAGE_FILTER` | True | At Δt = 1 s, triplets can span a 30-s lag — requiring every interior window to share the stage prevents straddling boundaries |
| `BANDS` | None | Broadband first pass; band-resolved pass to follow |
| `DURATION_HOURS` | 3.5 | Captures ~2 full NREM–REM cycles on Bitbrain ds005555 sub-1 — enough N3, ≥1 REM bout |
| `TARGET_STEP` | 3 | Stride for the time-resolved PID target axis (every 3 s). The window itself is still 1 s; only the *time-density* of estimates is thinned. Global PID, AR(1), block-perm null are unaffected. |

### AR(1) baseline — stage-conditional

φ and σ are fit per sleep stage on that stage's windows only; an AR(1) PID matrix is generated per stage and matched to each target window at broadcast time. This isolates excess (Actual − AR(1)) *within* a stage from across-stage spectral differences, and is what underwrites the new per-stage excess figure.

### Block-permutation null

At Δt = 1 s, the block-permutation null shuffles **1-second blocks**, which actually destroys most of the linear autocorrelation of interest — directly responding to Pedro's concern that 30-s block shuffles preserved too much structure to be informative.

## Scripts

```
python scripts/pid/eeg_sleep_compute.py        # all 6 channels, broadband
python scripts/pid/eeg_sleep_plot.py           # all figures, including Plot 7b
```

Results are saved to `results/pid/eeg_sleep/<channel>/`.
