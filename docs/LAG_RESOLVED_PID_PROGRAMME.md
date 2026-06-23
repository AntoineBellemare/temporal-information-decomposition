# Multi-scale partial information decomposition (MS-PID)
## Lag-resolved information decay as a window into temporal information dynamics

A research-programme plan separate from the sleep PID paper.

---

## 1. Motivation

The 1-s sleep-PID analysis turned up an empirical observation: the
coefficient of variation of PID atoms across recording windows varies
systematically with the lag of the predictor pair. Wake/N1/N2 show
monotonic decline (correlation-decay regime); REM shows an *increase*;
N3 shows a U-shape. The opposing slopes cannot be explained by
spectral content alone, and the S/R-ratio CV — which is amplitude-
invariant by construction — also shows opposing slopes.

That observation prompts a more general question:

> Does the partial information decomposition show lawful scaling
> across the temporal lag of its source pair, and if so what does
> that scaling reveal about the underlying dynamics?

Existing literature analogues:
- **1/f spectral scaling** (He 2014, Voytek et al. 2015) — scale-free amplitude variability.
- **Multi-scale entropy** (Costa et al. 2002) — scale-invariance of complexity via coarse-graining.
- **PhiID Temporal Integration Index** ([scripts/phiid/temporal_integration_index_eeg.py](../scripts/phiid/temporal_integration_index_eeg.py) and [docs/TEMPORAL_INTEGRATION_FRAMEWORK.md](TEMPORAL_INTEGRATION_FRAMEWORK.md)) — per-window scale-invariance of 16 atoms.

None of these uses PID atom statistics as the carrier. The programme
proposes that as a new axis.

---

## 2. Working hypothesis

For a stationary signal with characteristic autocorrelation timescale
τ₀ and dominant rhythm period T:

- At lag τ ≪ τ₀ the predictor pair is strongly coupled to the target;
  atom values are large and reproducible across windows → CV is high but
  saturated.
- At lag τ ~ τ₀ the autocorrelation knee is being traversed; some
  windows happen to align with the dominant rhythm, others don't →
  CV is at its maximum.
- At lag τ ≫ τ₀ the predictor pair is decorrelated from the target;
  atoms approach a discretisation-noise floor → CV drops to a
  stage-specific minimum.

This predicts CV(τ) curves that decay monotonically as τ grows past
the autocorrelation knee. **Positive slopes or U-shapes are
inconsistent with this simple account** and require additional
mechanisms — within-stage heterogeneity at multiple timescales, sub-
state alternation, or true scale-free dynamics.

The headline empirical question:

> When CV(τ) does decay, is it a power law (`A·τ^(-β) + C`) or an
> exponential decay (`A·exp(-τ/τ₀) + C`), and does the exponent β (or
> τ₀) carry physiological information that the 1/f spectral exponent
> does not?

---

## 3. Programme structure

### Phase 1 — Methodological validation on synthetic systems

Generate signals with known statistical properties and verify that the
MS-PID CV-decay fitter reports the expected scaling. Without this step
nothing else means anything.

| Test signal | Expected CV(τ) behaviour | Tests |
|---|---|---|
| White noise | Flat at noise floor | Sanity: does the pipeline produce zero scaling? |
| AR(1) with φ ∈ {0.3, 0.6, 0.9} | Exponential decay with τ₀ ∝ −1/log(φ) | Does exponential model win? Does fitted τ₀ recover the true φ? |
| Fractional Brownian motion (β ∈ {0.5, 1.0, 1.5}) | Power-law decay with exponent linked to β | Does power-law win? Is β recoverable? |
| Multifractal cascade | Genuine multi-scale; mixed | What does the fitter say? |
| Two-regime piecewise-stationary (e.g. REM phasic ↔ tonic) | Positive slope at lags matching switching timescale | Reproduces empirical REM observation |

Deliverable: a unit-tested fitter that:
- Fits exponential and power-law to CV(τ) curves
- Compares by AIC / BIC
- Returns scaling exponent + bootstrap CI
- Reports model-comparison evidence

### Phase 2 — Extend the empirical lag range

Current range 1–30 s = 1.5 decades is insufficient for power-law
claims (which conventionally need ≥ 2 decades).

| Goal | Mechanism |
|---|---|
| Short lag floor: ~100 ms | Sub-second target windows or overlapping windows |
| Long lag ceiling: ~10 min | Longer recordings, longer stage bouts, or coarser sampling at long lags |
| Log-spaced lag grid | Replace linear 1, 2, …, 30 s with e.g. 0.1, 0.2, 0.4, … 600 s |

Deliverable: re-runnable `eeg_sleep_compute.py` config that accepts a
log-spaced custom lag list, plus updated PID joint-distribution machinery
that handles much-longer lag offsets.

### Phase 3 — Empirical test on the sleep dataset

With the validated fitter and extended lag range:

| Question | Analysis |
|---|---|
| Does CV(τ) decay follow a power law in any stage / band? | Fit per (subject, channel, stage, band, atom); report exponent + CI |
| Do stages cluster by exponent? | Mixed-effects model on β with random subject intercepts |
| Does the CV-decay exponent correlate with 1/f spectral slope per stage? | Spearman between MS-PID β and FOOOF aperiodic slope |
| Where do positive slopes appear, and do they localise to specific bands? | Test the within-stage heterogeneity story explicitly |

Deliverable: figure(s) showing per-stage CV(τ) curves with fitted
power-law / exponential overlays, and a scatter of MS-PID exponent
vs 1/f exponent (showing independence or redundancy).

### Phase 4 — Theoretical / generative model

Develop the simplest toy model that generates the observed CV-decay
patterns:

1. Coupled hierarchy of oscillators with power-law-distributed time
   constants — does this produce power-law CV(τ) in PID atoms?
2. Two-state hidden Markov model (e.g. phasic vs tonic) with switching
   rate λ — does this produce a positive-slope CV regime at τ ~ 1/λ?
3. Stochastic dynamical system on a manifold with multi-scale curvature
   — does this produce stage-like signatures?

Deliverable: a small library of generative models + the corresponding
MS-PID signature, used as the mechanistic anchor for empirical claims.

### Phase 5 — Cross-system validation

| System | Why | Test |
|---|---|---|
| ECG / HRV | Different timescales, well-characterised multi-scale structure | Does cardiac MS-PID recover known multi-scale entropy results? |
| fMRI BOLD | Very slow (seconds to minutes), no high-frequency confound | Does the framework still work in this regime? |
| Anaesthesia / sedation depth | Known monotonic loss of consciousness with dose | Does MS-PID exponent move monotonically with depth? |
| Disorders of consciousness (vegetative / minimally conscious) | Established complexity / Φ findings | Does MS-PID separate states more cleanly than 1/f? |

---

## 4. Where the existing TII framework fits

The PhiID TII framework already in the repo computes per-window
scale-invariance for the 16 PhiID atoms (and the 4 IIT-inspired
metrics) via `divergence(t) = std across τ` and `TII = 1/(1+div)`.

In the MS-PID programme this is one of several scale-invariance
measures:

| Measure | What it captures |
|---|---|
| **Per-window divergence** (PhiID-TII style applied to PID) | How much do the 4 atoms vary across the 2-D lag grid at this moment? |
| **Across-window CV(τ)** (current sleep figure) | How variable is each atom across windows at a fixed lag, parameterised by lag? |
| **Power-law / exponential fit** (new) | Does the across-window CV-vs-lag relationship follow a known scaling law? |
| **Cross-lag correlation** (PhiID-TII complement) | Are the atom values at different lags correlated across windows? |

The current sleep paper uses #2 only. The MS-PID programme would
unify #1, #2, and #3 into one analysis framework, with #4 as a
secondary measure.

A PID-TII analogue is therefore **a derivable measure inside the
programme** but **not a separately publishable contribution on its
own** — its value is as one of several scale-invariance views.

---

## 5. Deliverables and rough timeline

| Phase | Output | Estimated effort |
|---|---|---|
| 1 | Synthetic validation, fitter library, unit tests | 2–3 weeks |
| 2 | Extended lag-range pipeline | 1–2 weeks |
| 3 | Sleep MS-PID figures and statistics | 2–3 weeks |
| 4 | Toy generative models | 3–4 weeks |
| 5 | One cross-system replication | 2–3 weeks |
| Writing | Methods/theory paper draft | 4–6 weeks |
| **Total to a first paper draft** | | **~4–6 months** |

Realistically a separate person-month commitment than the sleep paper
finishing pass.

---

## 6. Risks and pre-mortems

| Risk | Mitigation |
|---|---|
| CV(τ) is well-fit by exponential everywhere → no 1/t story | That's a useful negative result; reframe as "lag-resolved PID is an exponential signature of stage autocorrelation, complementary to 1/f". |
| Synthetic 1/f systems do not produce power-law CV decay in PID atoms | Indicates the analogy was loose; programme pivots to "PID atoms as autocorrelation-timescale probe". |
| Extended lag range exposes non-stationarity within stage bouts | Use shorter bout-internal windows or model the non-stationarity explicitly. |
| Multi-scale entropy literature already covers this | Direct comparison shows what (if anything) PID-based scaling adds beyond complexity-based scaling. |
| Within-stage heterogeneity (the candidate mechanism for positive slopes) cannot be operationalised | Use REM phasic / tonic scoring (well-defined in the literature) as ground truth and replicate. |

---

## 7. Decision points

1. After Phase 1: does the fitter reliably recover known exponents
   from synthetic systems? If no, the rest of the programme is
   premature.
2. After Phase 3: do real-EEG MS-PID exponents separate stages
   *independently of* 1/f exponents? If no, the framework is
   redundant with established methods.
3. After Phase 4: does at least one toy model reproduce the
   opposing-slopes pattern empirically? If no, the candidate
   mechanism story is untenable and the programme should pivot.

If decision points 1 and 2 fail, the programme should be wound down
and the sleep paper's framing kept descriptive (which it already is).

---

## 8. Scope explicitly excluded

- Replacing or competing with PhiID-TII — they're complementary.
- Multi-scale entropy or LZ-complexity scaling — different decompositions, separate questions.
- Connectivity / cross-channel PID — programme is about temporal self-PID, not multivariate.
- Real-time / closed-loop applications — late-stage if anything.

---

## 9. Open questions to resolve before committing

- Is the right "carrier" the four atoms separately, or aggregated
  measures (atom fractions, S/R ratio, total MI)?
- Do we fit CV(τ) or CV(τ₁, τ₂) — i.e., 1-D collapse or 2-D surface?
  The 2-D version has more information but is harder to fit and
  interpret.
- What is the minimum recording length per stage bout to fit a
  scaling law reliably?
- Is the within-recording form of CV the right one, or should we
  also report a between-subject form (does the framework
  generalise across subjects)?

These are first-order questions to answer in Phase 1 before any
real-data fitting.
