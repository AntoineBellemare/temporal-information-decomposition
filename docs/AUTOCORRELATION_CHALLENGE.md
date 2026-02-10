## Interpreting Results: Autocorrelation vs Genuine Structure

### The Challenge

Both PID and PhiID applied to temporal embeddings will capture **two types of structure**:

1. **Linear autocorrelation** — trivial, expected from signal smoothness
2. **Nonlinear/higher-order structure** — the genuinely interesting dynamics

At small τ (overlapping windows), linear autocorrelation dominates and obscures the interesting structure. The question is: **How do we know if our results reflect genuine temporal dynamics or just trivial smoothness?**

### Three Approaches to Isolate Genuine Structure

| Approach | Method | What It Reveals | Applies To |
|----------|--------|-----------------|------------|
| **A. Prewhitening** | Remove linear autocorrelation via AR model, analyze residuals | Nonlinear structure at ALL timescales | PID, PhiID |
| **B. Large τ** | Use τ beyond the correlation time | Only long-range dependencies | PID, PhiID |
| **C. Null Model Comparison** | Compare to matched AR(1) baseline | Excess structure over expected autocorrelation | PID, PhiID |

### Key Insight: The Sine Wave Example

A noisy sine wave with period T=20 analyzed at τ=5 (quarter period) demonstrates that **genuine structure exists beyond autocorrelation**:

| Measurement | Value | Interpretation |
|-------------|-------|----------------|
| Autocorrelation | 0.00 | Phases at 0°, 90°, 180°, 270° are orthogonal |
| Synergy (sts) | ~2.0 bits | Knowing all 4 points reveals the phase |

The autocorrelation is zero, yet synergy is high. This proves that PID/PhiID capture structure that **is not** linear autocorrelation.

### τ Regimes and Interpretation

For a signal with correlation time $τ_{corr}$:

| Regime | Relationship | Interpretation |
|--------|--------------|----------------|
| $τ \ll τ_{corr}$ | Overlapping | Total structure (autocorr + nonlinear mixed) |
| $τ \approx τ_{corr}$ | Transition | Autocorrelation decaying, nonlinear emerging |
| $τ \gg τ_{corr}$ | Non-overlapping | Pure long-range structure only |

**Important**: You do NOT need zero overlap to see beyond autocorrelation. The key is to:

1. Analyze across multiple τ values
2. Look for patterns that **differ from AR(1) baseline** with matched autocorrelation
3. Focus on τ values near known characteristic timescales

### Validity for PID vs PhiID

Both frameworks measure the same underlying information-theoretic quantities, just with different decompositions:

| Framework | Atoms | Autocorrelation Captured In | Genuine Structure Appears In |
|-----------|-------|----------------------------|------------------------------|
| **PID** (4 atoms) | R, Ux, Uy, S | Redundancy (R) dominates | Synergy (S), Unique info |
| **PhiID** (16 atoms) | 4×4 grid | Diagonal (rtr, xtx, yty, sts) | Off-diagonal transfers, asymmetries |

For both:
- **Redundancy/Storage** inflated by linear autocorrelation
- **Synergy** more likely to reflect genuine nonlinear structure
- **Asymmetries** (Unique X ≠ Unique Y, or transfer patterns) indicate directional dynamics beyond correlation

### Practical Recommendations

1. **Always analyze multiple τ values** — don't trust a single timescale
2. **Include an AR(1) matched baseline** — if your results look like AR(1), it's probably just autocorrelation
3. **Focus on synergy and asymmetries** — these are less contaminated by linear structure
4. **For strong claims, use prewhitening** — AR residuals reveal purely nonlinear dynamics
5. **Report τ relative to correlation time** — "τ = 2×τ_corr" is more informative than "τ = 50"
