# Toy Examples Validation Report

Computational validation of the toy example scripts in `scripts/pid/toy_examples.py` and `scripts/phiid/toy_examples.py`. Each finding was verified by running the code and inspecting actual outputs.

## Overall Assessment

The core math, information-theoretic computations, and PhiID integration are **correct**. The Takens embedding is implemented with the right vector ordering. Toy model selection is good and covers the important categories (baseline, memory, synergy, oscillatory, nonlinear). The issues found are in edge cases of process design, a visualization bug, and overly strict/loose validation thresholds.

---

## PID Issues

### 1. Noiseless XOR is degenerate (period-3 sequence)

**File:** `scripts/pid/toy_examples.py`, `generate_xor_process` (line 70)

**Problem:** Unlike the PhiID version (which adds 5% noise), the PID XOR generator is purely deterministic: `x[t] = x[t-1] ^ x[t-2]`. This produces a **period-3 sequence** (e.g., 0,1,1,0,1,1,...), which means only 3 of the 4 possible XOR input-output triples are ever observed:

```
Observed:   (0,1,1), (1,0,1), (1,1,0)   — each with P = 1/3
Missing:    (0,0,0)                       — never occurs
Ideal XOR:  all four triples with P = 1/4 each
```

**Impact on PID values:**

|            | Ideal XOR | Empirical (period-3) | Delta |
|------------|-----------|----------------------|-------|
| Redundancy | 0.000     | 0.252                | +0.252 (spurious) |
| Synergy    | 1.000     | 0.667                | -0.333 (lost) |

The missing triple biases the marginals: P(source=0) = 1/3 instead of 1/2. This creates spurious redundancy (0.252 bits) and loses one-third of the synergy. The `compare_pid_measures` function compares this against `bivariates['synergy']`, making the gap visible but without explaining why.

**Severity:** Moderate. The qualitative conclusion (XOR = synergistic) still holds, but the quantitative values are substantially distorted.

**Fix:** Add noise as the PhiID version does: `noise_prob=0.05`.

### 2. Heatmap symmetrization is wrong for unique atoms

**File:** `scripts/pid/toy_examples.py`, `plot_lag_sweep_heatmaps` (line 511-513)

**Problem:** The code symmetrizes the heatmap for ALL four PID components:

```python
matrix[l1-1, l2-1] = row[comp]
matrix[l2-1, l1-1] = row[comp]  # Symmetric
```

Redundancy and synergy are symmetric in the source ordering, but `unique_0` and `unique_1` are **not**. Swapping the lag pair swaps which source is "first":

```
Lags (1,3): R=0.1003  U0=0.0361  U1=0.0000  S=0.0614
Lags (3,1): R=0.1003  U0=0.0000  U1=0.0361  S=0.0614
                       ^^^^^^^^^^  ^^^^^^^^^^
                       These swap when lags swap
```

The heatmap copies the same value to both `(i,j)` and `(j,i)`, but `unique_0` at position `(1,3)` ≠ `unique_0` at position `(3,1)`. The analysis only computes `lag1 < lag2` so the underlying data is correct; only the visualization is misleading.

**Severity:** Visualization bug. Affects `lag_sweep_heatmaps.png`.

**Fix:** Only symmetrize for redundancy and synergy. For unique atoms, either show only the upper triangle or compute both orderings.

### 3. Duplicate file listing in main()

**File:** `scripts/pid/toy_examples.py`, lines 751-754

The file listing loop appears twice, printing every output file name twice:

```python
for f in sorted(RESULTS_DIR.glob("*")):
    print(f"  - {f.name}")
for f in sorted(RESULTS_DIR.glob("*")):   # <-- duplicate
    print(f"  - {f.name}")
```

**Severity:** Trivial. Confirmed in actual output (file list appears twice at the end).

---

## PhiID Issues

### 4. `generate_copy_process` is dead code

**File:** `scripts/phiid/toy_examples.py`, line 140

The function generates a perfectly constant signal (`x[t] = x[t-1]`, no noise). It is defined but never instantiated in `main()`. The processes list includes `NoisyCopy` (random walk) instead.

If someone did use it with `kind='gaussian'`, the signal would have zero variance in all 4 Takens vectors, producing a singular covariance matrix and numerical errors. The docstring at the top mentions "COPY: High rtr" which could mislead users into thinking it's tested.

**Severity:** Minor (dead code, potential confusion).

**Fix:** Remove the function or add noise to make it usable.

### 5. NoisyCopy (random walk) is non-stationary

**File:** `scripts/phiid/toy_examples.py`, `generate_noisy_copy` (line 153)

**Problem:** `np.cumsum(noise)` is a random walk with variance growing as O(t). Gaussian copula PhiID assumes stationarity.

```
Signal range:                [-7.9, 11.2]
Std of first 1000 samples:   0.956
Std of last 1000 samples:    0.655
Std of all samples:           5.031
```

For small τ the 4 Takens vectors overlap extensively, so their statistics are similar. For large τ:

```
Takens vectors at τ=50:
  v0 (t)     mean=0.795  std=5.000
  v1 (t+τ)   mean=0.772  std=5.018
  v2 (t+2τ)  mean=0.749  std=5.038
  v3 (t+3τ)  mean=0.725  std=5.060
```

The systematic drift is small relative to std, so the qualitative result (very high rtr = 2.601 at τ=5) holds. But this isn't a clean validation case because the high rtr could partly reflect non-stationarity artifacts rather than genuine temporal memory.

**Severity:** Moderate. AR1_0.99 (φ=0.99) already tested in the same script serves the same purpose with a stationary process.

**Fix:** Replace with a high-φ AR(1) or use differenced random walk.

### 6. Damped oscillation is 90% noise, produces artifacts

**File:** `scripts/phiid/toy_examples.py`, `generate_damped_oscillation` (called with `decay=0.002, N=12000` at line 1254)

**Problem:** The oscillation envelope decays below the noise floor (0.1) by sample ~1150:

```
t=    0: amplitude=1.0000, SNR=10.00x
t=  500: amplitude=0.3679, SNR=3.68x
t= 1000: amplitude=0.1353, SNR=1.35x
t= 1500: amplitude=0.0498, SNR=0.50x  <-- below noise floor
```

Only 9.6% of the signal is above the noise floor. Since PhiID averages over the entire series, the results are dominated by noise.

**Actual results show artifacts, not the expected "atoms decay with signal" pattern:**

```
tau_embed    rtr       xtx      sts
    1       0.046    0.000    0.001
    5       0.000    0.148    0.290   ← large, non-decaying
   10       0.140    0.005    0.032
   20       0.126    0.000    0.028   ← still large at τ=period
   50       0.085   -0.008    0.017
  100       0.051    0.011    0.013
```

The non-stationarity (variance = 0.22 at t=0 vs 0.009 at t=2000+) creates spurious information structure: knowing x(t) is large implies early in the sequence, which predicts x(t+τ) is also large. This is envelope correlation, not genuine oscillation dynamics.

**Severity:** Moderate. The toy process doesn't validate what it claims to.

**Fix:** Use `decay=0.0001` (envelope drops to 0.30 by end of series, staying above noise) or reduce `n_samples` to 2000.

### 7. AR1 verification thresholds are too strict

**File:** `scripts/phiid/toy_examples.py`, verification block (lines 1399-1409)

The code checks `rtr > 0.1` for `AR1_0.5` and `AR1_0.9` at τ=5. Both fail:

```
AR1_0.5 rtr at τ=5 = 0.000029  →  > 0.1? False  ✗
AR1_0.9 rtr at τ=5 = 0.023854  →  > 0.1? False  ✗
```

For AR(1) with φ=0.5, the autocorrelation at lag 5 is φ^5 = 0.031 — barely above noise. Even AR(1) with φ=0.9 has ρ(5) = 0.59, but the Gaussian PhiID rtr is only 0.024. The threshold of 0.1 is too high for both.

The script prints "?" for these checks, which could be mistaken for a validation failure:

```
AR1_0.5: rtr = 0.000 (expected > 0) ?
AR1_0.9: rtr = 0.024 (expected > 0) ?
```

**Severity:** Minor. The values are actually correct — the threshold is just too strict.

**Fix:** Use `rtr > 0.001` for AR1_0.5 and `rtr > 0.01` for AR1_0.9, or check at τ=1 instead of τ=5.

### 8. Total_redundancy and Total_synergy metrics overlap

**File:** `scripts/phiid/toy_examples.py`, `IIT_METRICS` (line 96)

Atoms `str` (synergy→redundancy) and `rts` (redundancy→synergy) appear in **both** metrics:

```
Total_redundancy = rtr + xtr + ytr + str + rtx + rty + rts
Total_synergy    = str + stx + sty + sts + xts + yts + rts
                   ^^^                              ^^^
                   counted in both
```

Demonstrated with XOR at τ=5:

```
str = 0.2459 (counted in BOTH)
rts = 0.2459 (counted in BOTH)
Total_redundancy = 0.493
Total_synergy    = 0.499
Sum              = 0.992  >  Total all 16 atoms = 0.746
```

The metrics are internally consistent (they're "all atoms touching redundancy" and "all atoms touching synergy") but not complementary. Anyone summing them expecting a partition of total information would get inflated values.

**Severity:** Documentation issue.

**Fix:** Add a comment noting the overlap, or define non-overlapping variants that exclude cross-type atoms.

---

## Confirmed Correct

The following were verified and found to be sound:

- **Takens embedding vector ordering** matches what `phyid` expects: `[src_past, trg_past, src_future, trg_future]`
- **DYNAMICS_GROUPS** match Mediano et al. (2021) definitions for storage, copy, transfer, erasure, upward/downward causation
- **IID baselines** produce near-zero atoms (IID_Binary total = 0.001, IID_Gauss total = 0.000)
- **AR(1) memory decay** profile is correct: rtr decays exponentially with τ, rate matches φ
- **AR(1) noise scaling** `sqrt(1-φ²)` for unit stationary variance is correct
- **AR(2) noise variance formula** matches Yule-Walker derivation; `max(0.1, ...)` safety clamp is appropriate
- **AR(2) stationarity conditions** satisfied for all 5 tested coefficient pairs
- **AR(2) oscillatory parameterization** `(2r·cos(2πf), -r²)` is textbook correct
- **XOR synergy** (PhiID version): sts=0.271, str=0.246 — correct high synergy
- **Hierarchical timescales** (PID): redundancy peaks at lags (1,2), synergy at (5,6) — successful separation
- **PID measure comparison** (MMI, BROJA, CCS) runs correctly on both ideal and empirical distributions
- **Logistic map** r=3.9 and **Hénon map** a=1.4, b=0.3 use standard chaotic parameters
- **Discretization sensitivity** test (bins 2-10) properly shows increasing information capture with more bins
- **Sine wave at τ=T/4** shows high sts (1.954) — synergy from quadrature relationship, as predicted

---

## Summary Table

| # | Issue | Severity | File | Line(s) |
|---|-------|----------|------|---------|
| 1 | Noiseless XOR is degenerate (period-3, missing 1 of 4 triples) | moderate | pid/toy_examples.py | 70-78 |
| 2 | Unique atom heatmaps incorrectly symmetrized | vis. bug | pid/toy_examples.py | 511-513 |
| 3 | Duplicate file listing in summary | trivial | pid/toy_examples.py | 751-754 |
| 4 | Dead `generate_copy_process` (zero-variance, never called) | minor | phiid/toy_examples.py | 140-150 |
| 5 | NoisyCopy (random walk) violates stationarity assumption | moderate | phiid/toy_examples.py | 153-159 |
| 6 | Damped oscillation is 90% noise, produces artifacts | moderate | phiid/toy_examples.py | 266-275, 1254 |
| 7 | AR1 verification thresholds too strict at τ=5 | minor | phiid/toy_examples.py | 1399-1409 |
| 8 | Total_redundancy/Total_synergy double-count `str`, `rts` | documentation | phiid/toy_examples.py | 96-101 |
