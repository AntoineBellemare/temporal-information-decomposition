# Temporal PhiID: Interpretation of Atoms and Metrics

## Overview

This document explains how to interpret the 16 PhiID atoms and derived metrics when using **Takens delay embedding** on a single time series. The standard PhiID framework assumes two distinct signals (X, Y), but our temporal approach embeds one signal at 4 regularly-spaced time points.

---

## 1. The Temporal Embedding

### Standard PhiID (Two Signals)
```
Source 1 (X): X_past → X_future
Source 2 (Y): Y_past → Y_future
```

### Temporal Takens PhiID (Single Signal)
```
signal x(t) embedded as:
  p1 = x(t)        →  "X_past"
  p2 = x(t+τ)      →  "Y_past"  
  t1 = x(t+2τ)     →  "X_future"
  t2 = x(t+3τ)     →  "Y_future"

Timeline: t ──τ── t+τ ──τ── t+2τ ──τ── t+3τ
          p1      p2        t1        t2
```

**Key insight**: "X" and "Y" are not independent signals—they are **phase-shifted copies** of the same process at different points in the embedding window.

---

## 2. The 16 PhiID Atoms

### Atom Notation

Each atom is denoted `αβ` where:
- **First letter (α)**: Information type in the SOURCE (past)
- **Second letter (β)**: Information type in the TARGET (future)

| Letter | Meaning | In Temporal Context |
|--------|---------|---------------------|
| **r** | Redundancy | Info shared by BOTH p1 and p2 |
| **x** | Unique to "X" | Info unique to p1 (earliest point) |
| **y** | Unique to "Y" | Info unique to p2 (early-mid point) |
| **s** | Synergy | Info requiring BOTH p1 AND p2 together |

### The 4×4 Atom Grid

|  | → **r** (Red) | → **x** (Un1) | → **y** (Un2) | → **s** (Syn) |
|---|---|---|---|---|
| **r** → | `rtr` | `rtx` | `rty` | `rts` |
| **x** → | `xtr` | `xtx` | `xty` | `xts` |
| **y** → | `ytr` | `ytx` | `yty` | `yts` |
| **s** → | `str` | `stx` | `sty` | `sts` |

### Temporal Interpretation of Each Atom

#### Diagonal Atoms (Self-Transfer)

| Atom | Standard Meaning | **Temporal Meaning** |
|------|------------------|----------------------|
| `rtr` | Redundant info stays redundant | **Persistent memory**: Info stable across ALL 4 timepoints |
| `xtx` | X-unique info stays X-unique | **Local persistence at t→t+2τ**: Position 0 predicts position 2 |
| `yty` | Y-unique info stays Y-unique | **Local persistence at t+τ→t+3τ**: Position 1 predicts position 3 |
| `sts` | Synergy creates synergy | **Temporal integration**: Info only decodable with ALL 4 points |

#### Transfer Atoms (Cross-Position)

| Atom | Standard Meaning | **Temporal Meaning** |
|------|------------------|----------------------|
| `xty` | X-unique becomes Y-unique | **Odd-even phase coupling**: t→t+2τ info shifts to t+τ→t+3τ |
| `ytx` | Y-unique becomes X-unique | **Even-odd phase coupling**: t+τ→t+3τ info shifts to t→t+2τ |
| `xtr` | X-unique becomes redundant | **Info spreading**: Early-unique info becomes shared |
| `ytr` | Y-unique becomes redundant | **Info spreading**: Mid-unique info becomes shared |

#### Erasure Atoms (Information Loss)

| Atom | Standard Meaning | **Temporal Meaning** |
|------|------------------|----------------------|
| `rtx` | Redundant becomes X-unique | **Memory decay (partial)**: Shared info lost at position 1,3 |
| `rty` | Redundant becomes Y-unique | **Memory decay (partial)**: Shared info lost at position 0,2 |
| `rts` | Redundant becomes synergistic | **Complexity emergence**: Shared info becomes entangled |

#### Causation Atoms (Up/Down)

| Atom | Standard Meaning | **Temporal Meaning** |
|------|------------------|----------------------|
| `xts` | X-unique creates synergy | **Upward causation**: Early info combines with other to create emergence |
| `yts` | Y-unique creates synergy | **Upward causation**: Mid info combines to create emergence |
| `stx` | Synergy becomes X-unique | **Downward causation**: Whole constrains early positions |
| `sty` | Synergy becomes Y-unique | **Downward causation**: Whole constrains mid positions |
| `str` | Synergy becomes redundant | **Downward spread**: Emergent info becomes persistent |

---

## 3. High-Order Metrics: DYNAMICS_GROUPS

### Storage: `['rtr', 'xtx', 'yty', 'sts']`

**Standard**: Information preserved over time in both X and Y

**Temporal interpretation**: 
- **Total temporal persistence** across the embedding window
- Measures how predictable the signal is from ANY earlier point to ANY later point
- High storage = strong autocorrelation structure at timescale τ

**Processes with high Storage**:
- COPY: `x[t] = x[t-1]`
- AR(1) with high φ
- Slowly varying signals

---

### Copy: `['xtx', 'yty']`

**Standard**: Self-continuity within X and within Y

**Temporal interpretation**:
- **Autocorrelation at lag 2τ** (skip-one correlation)
- `xtx`: How well does x(t) predict x(t+2τ)?
- `yty`: How well does x(t+τ) predict x(t+3τ)?

**Important**: This is NOT "copying" in the colloquial sense—it measures correlation at **double** the embedding lag.

**Processes with high Copy**:
- Signals with strong 2τ autocorrelation
- Oscillations with period ~4τ (quarter-period spacing lands on same phase)

---

### Transfer: `['xty', 'ytx']`

**Standard**: Information flow between X→Y and Y→X

**Temporal interpretation**:
- **Phase relationship between embedding positions**
- `xty`: Info at (t, t+2τ) that predicts (t+τ, t+3τ) pattern
- `ytx`: Info at (t+τ, t+3τ) that predicts (t, t+2τ) pattern
- Measures "cross-talk" between even/odd positions

**NOT true transfer entropy** in the temporal case—there's no separate "source" and "target" process.

**Processes with high Transfer**:
- Oscillations (π/2 phase offset between even/odd positions)
- AR(2) processes
- Signals with specific phase relationships

---

### Erasure: `['rtx', 'rty']`

**Standard**: Redundant information that becomes unique (lost from one stream)

**Temporal interpretation**:
- **Memory decay within the embedding window**
- Info that was shared at (t, t+τ) but only persists to one of (t+2τ, t+3τ)
- Indicates **partial forgetting** on timescale ~2τ

**Processes with high Erasure**:
- Signals with mixed persistence (some components decay faster)
- Transitions between states
- Edge effects in windows

---

### Upward_causation: `['xts', 'yts', 'rts']`

**Standard**: Parts (unique/redundant) create synergistic whole

**Temporal interpretation**:
- **Emergence of temporal patterns**
- Information from individual timepoints that **only becomes meaningful** when combined
- Measures how local dynamics create global structure

**Processes with high Upward causation**:
- XOR-like processes: `x[t] = x[t-1] ⊕ x[t-2]`
- Systems where multi-scale interactions matter
- Nonlinear processes with delayed dependencies

---

### Downward_causation: `['stx', 'sty', 'str']`

**Standard**: Synergistic whole constrains the parts

**Temporal interpretation**:
- **Constraint by global temporal structure**
- Future synergistic patterns that **constrain** what can appear at individual timepoints
- Measures how the overall trajectory limits local variability

**Processes with high Downward causation**:
- Strongly oscillatory signals (global rhythm constrains local values)
- Attractor dynamics
- Signals with strong temporal structure

---

## 4. High-Order Metrics: IIT_METRICS

These are inspired by Integrated Information Theory (Mediano et al.) and the goofi PhiID implementation.

### Information_storage: `['xtx', 'yty', 'rtr', 'sts']`

**Same as Storage** in DYNAMICS_GROUPS.

**Temporal interpretation**: Total information preserved across the embedding window.

---

### Transfer_entropy: `['xty', 'xtr', 'str', 'sty']`

**Standard**: Information transferred between processes

**Temporal interpretation**:
- **Cross-position information flow** within the embedding
- NOT equivalent to true transfer entropy (which requires separate source/target)
- Measures "off-diagonal" flow: how info at one phase predicts another phase

**Components**:
- `xty`: Even→Odd phase coupling
- `xtr`: Early-unique spreading to redundant
- `str`: Synergy becoming redundant
- `sty`: Synergy becoming Y-unique

**Warning**: In temporal PhiID, this does NOT measure causal influence between systems—it measures internal phase relationships.

---

### Causal_density: `['xtr', 'ytr', 'sty', 'str', 'str', 'xty', 'ytx', 'stx']`

**Standard**: Complexity of causal interactions (note: `str` appears twice in original definition)

**Temporal interpretation**:
- **Mixing of information types** across the embedding
- High causal density = info doesn't stay in neat categories
- Measures how much information "moves around" between redundancy, unique, and synergy

**Processes with high Causal density**:
- Complex/chaotic dynamics
- Signals with multiple interacting timescales
- Non-stationary processes

---

### Integrated_information: `['rts', 'xts', 'sts', 'sty', 'str', 'yts', 'ytx', 'stx', 'xty']`

**Standard**: Φ-like measure: synergistic integration beyond the parts

**Temporal interpretation**:
- **Net temporal integration** across the embedding window
- In the code, `rtr` is subtracted: `integrated = sum(atoms) - rtr`
- This removes "trivial" redundancy to measure true emergence

**Components** (all involve synergy):
- Synergy-producing: `rts`, `xts`, `yts` (upward causation)
- Synergy-maintaining: `sts`
- Synergy-distributing: `sty`, `str`, `stx` (downward causation)
- Transfer: `ytx`, `xty`

**Processes with high Integrated information**:
- XOR processes (pure synergy)
- Complex temporal dependencies
- Signals where knowing one timepoint tells you nothing without others

---

## 5. Summary Table: What Each Metric Actually Measures

| Metric | Standard Interpretation | **Temporal Takens Interpretation** |
|--------|------------------------|-----------------------------------|
| **Storage** | Info preserved in X and Y | Total autocorrelation structure at scale τ |
| **Copy** | Self-continuity | Autocorrelation at lag 2τ |
| **Transfer** | X↔Y flow | Phase coupling between even/odd embedding positions |
| **Erasure** | Info loss from shared pool | Partial memory decay within window |
| **Upward** | Parts→Whole | Local dynamics creating global patterns |
| **Downward** | Whole→Parts | Global structure constraining local dynamics |
| **Integrated_info** | Φ (synergistic integration) | True temporal emergence minus trivial persistence |

---

## 6. Key Differences from Standard PhiID

1. **No independent sources**: X and Y are the SAME process, so "transfer" measures internal phase relationships, not cross-system influence.

2. **Autocorrelation dominates**: Redundancy (`r*` atoms) primarily reflects temporal autocorrelation, not shared information between independent sources.

3. **Synergy = temporal integration**: High `s*` atoms indicate information that requires MULTIPLE timepoints to decode—true temporal binding.

4. **τ is everything**: The embedding lag τ determines what timescale you're probing. Different τ values probe different temporal structures.

5. **"Copy" ≠ copying**: The Copy metric measures correlation at 2τ, not moment-to-moment persistence.

---

## 7. Practical Interpretation Guidelines

### When TII is HIGH (near 1):
- Same information dynamics at all τ values
- Scale-invariant temporal structure
- Consistent memory/integration across timescales

### When TII is LOW (near 0):
- Different dynamics at different τ values  
- Multi-scale processing
- Timescale-specific information structure

### High Storage + Low Synergy:
- Simple, persistent signal
- Strong autocorrelation
- Example: Random walk, slow drift

### Low Storage + High Synergy:
- Complex, integrated dynamics
- Information requires multiple timepoints
- Example: XOR process, chaotic systems

### High Transfer:
- Strong phase relationships within embedding
- Oscillatory structure
- Example: Sine wave at appropriate τ

### High Erasure:
- Memory decay within the window
- Transient dynamics
- Example: State transitions, non-stationary signals

---

## 8. References

- Mediano, P. A., et al. (2021). Towards an extended taxonomy of information dynamics via Integrated Information Decomposition.
- Rosas, F. E., et al. (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data.
- Lizier, J. T. (2012). JIDT: An information-theoretic toolkit for studying the dynamics of complex systems.

---

## Appendix: Atom Quick Reference

```
Atom  │ Source → Target      │ Temporal Meaning
──────┼──────────────────────┼──────────────────────────────────
rtr   │ Redundant → Redundant │ Persistent across ALL 4 points
rtx   │ Redundant → X-unique  │ Memory decay (lost at pos 1,3)
rty   │ Redundant → Y-unique  │ Memory decay (lost at pos 0,2)
rts   │ Redundant → Synergy   │ Shared info becomes entangled
──────┼──────────────────────┼──────────────────────────────────
xtr   │ X-unique → Redundant  │ Early info spreads to all
xtx   │ X-unique → X-unique   │ Persistence at lag 2τ (pos 0→2)
xty   │ X-unique → Y-unique   │ Even→Odd phase coupling
xts   │ X-unique → Synergy    │ Early info creates emergence
──────┼──────────────────────┼──────────────────────────────────
ytr   │ Y-unique → Redundant  │ Mid info spreads to all
ytx   │ Y-unique → X-unique   │ Odd→Even phase coupling
yty   │ Y-unique → Y-unique   │ Persistence at lag 2τ (pos 1→3)
yts   │ Y-unique → Synergy    │ Mid info creates emergence
──────┼──────────────────────┼──────────────────────────────────
str   │ Synergy → Redundant   │ Emergent info becomes persistent
stx   │ Synergy → X-unique    │ Whole constrains even positions
sty   │ Synergy → Y-unique    │ Whole constrains odd positions
sts   │ Synergy → Synergy     │ Pure temporal integration
```
