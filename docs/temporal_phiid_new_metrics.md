# Temporal PhiID: Proposed Higher-Order Metrics for a Single-Signal Embedding

## Overview

This document proposes a set of **new derived metrics** for **temporal PhiID** when PhiID is applied not to two separate systems, but to a **Takens delay embedding of a single time series**.

In standard PhiID, atoms are often grouped into metrics such as storage, transfer, erasure, and integrated information. These groupings are useful, but in the **temporal single-process case**, some of them become less natural because:

- "X" and "Y" are not different systems,
- "transfer" is not really inter-system transfer,
- the most important structure is often the **temporal geometry of information** across the embedding window.

The goal of this document is therefore to define **more adapted higher-order metrics** that summarize the 16 atoms in ways that are especially meaningful for:

- persistence,
- phase asymmetry,
- temporal integration,
- structural reformatting,
- broadcasting vs fragmentation,
- and multiscale temporal organization.

---

## 1. Temporal Embedding Reminder

We use a 4-point embedding of a single signal:

```text
p1 = x(t)
p2 = x(t+τ)
t1 = x(t+2τ)
t2 = x(t+3τ)
```

with timeline:

```text
t ──τ── t+τ ──τ── t+2τ ──τ── t+3τ
p1      p2        t1         t2
```

We map this into PhiID as:

```text
Sources (past): p1, p2
Targets (future): t1, t2
```

So the 16 atoms still exist, but they now describe how information **changes form across a temporal window of one process**.

---

## 2. The 16 Atoms

|  | → **r** | → **x** | → **y** | → **s** |
|---|---|---|---|---|
| **r** → | `rtr` | `rtx` | `rty` | `rts` |
| **x** → | `xtr` | `xtx` | `xty` | `xts` |
| **y** → | `ytr` | `ytx` | `yty` | `yts` |
| **s** → | `str` | `stx` | `sty` | `sts` |

Where:

- `r` = redundant information
- `x` = information unique to `p1`
- `y` = information unique to `p2`
- `s` = synergistic information

---

## 3. Guiding Principle for New Metrics

In the temporal case, it is often more useful to ask:

- Does information **stay in the same form**, or change form?
- Does it remain **local**, become **shared**, or become **integrated**?
- Does the embedding show **temporal symmetry** or **directional imbalance**?
- Is the system mainly **preserving**, **assembling**, **broadcasting**, or **fragmenting** information?
- How do these structures evolve as a function of **τ**?

The metrics below are designed to answer these questions directly.

---

# 4. Proposed Higher-Order Metrics

## 4.1 Persistence Hierarchy

Instead of one single "storage" metric, separate persistence into three levels.

### 4.1.1 Global Persistence
```text
Global_persistence = rtr
```

**Interpretation**:  
Information that remains redundant across the whole embedding window.  
This reflects **broad temporal persistence** or **global memory**.

High values suggest:
- strong slow structure,
- stable background state,
- information that is present across all four positions.

---

### 4.1.2 Phase-Specific Persistence
```text
Phase_persistence = xtx + yty
```

**Interpretation**:  
Information that remains tied to one phase class across the window:
- `xtx`: p1 → t1
- `yty`: p2 → t2

This reflects **lag-2τ persistence within each phase channel**.

High values suggest:
- skip-lag predictability,
- oscillatory or alternating structure,
- stable local phase identity.

---

### 4.1.3 Integrated Persistence
```text
Integrated_persistence = sts
```

**Interpretation**:  
Information that is synergistic in the past and remains synergistic in the future.

This reflects **persistent temporal integration**, i.e. information that can only be decoded by considering combinations of positions on both sides of the embedding.

High values suggest:
- stable temporal wholes,
- nonlocal temporal structure,
- coherent integrated motifs.

---

### 4.1.4 Total Persistence
```text
Total_persistence = rtr + xtx + yty + sts
```

**Interpretation**:  
All information that preserves its structural type across the embedding.

This is the temporal analogue of **self-consistency** or **closure of information type**.

---

## 4.2 Structural Lability

```text
Structural_lability = 
rtx + rty + rts +
xtr + xty + xts +
ytr + ytx + yts +
str + stx + sty
```

**Interpretation**:  
All atoms where information **changes type** from source to target.

This measures how much the system **reformats information** over the temporal window rather than simply preserving it.

High values suggest:
- dynamic reorganization,
- temporal transformation,
- regime transitions,
- structural instability or flexibility.

---

## 4.3 Closure vs Reformatting

### Closure
```text
Closure = rtr + xtx + yty + sts
```

### Reformatting
```text
Reformatting = Total_atoms - Closure
```

or explicitly:

```text
Reformatting =
rtx + rty + rts +
xtr + xty + xts +
ytr + ytx + yts +
str + stx + sty
```

**Interpretation**:  
This is a simple but powerful contrast:

- **Closure** = information stays in the same representational form
- **Reformatting** = information changes form across the window

This distinction is often more meaningful in temporal PhiID than the usual storage/transfer split.

---

## 4.4 Phase Exchange

### Unsigned phase exchange
```text
Phase_exchange = xty + ytx
```

**Interpretation**:  
Information switches between the two phase channels:
- `xty`: X-like past becomes Y-like future
- `ytx`: Y-like past becomes X-like future

This measures **even/odd exchange**, or **cross-phase coupling**.

High values suggest:
- oscillatory structure,
- alternation,
- parity coupling across the embedding.

---

### Signed phase bias
```text
Phase_bias = xty - ytx
```

**Interpretation**:  
Whether the exchange is balanced or directionally biased.

- positive: stronger `p1 → t2` style mapping
- negative: stronger `p2 → t1` style mapping

Useful for detecting subtle temporal asymmetries.

---

## 4.5 Broadcast Index

```text
Broadcast = xtr + ytr + str
```

**Interpretation**:  
Information that ends up **redundant in the future**, regardless of whether it began as unique or synergistic.

This captures the tendency of the process to **spread information across future positions**.

Components:
- `xtr`: local early info becomes shared
- `ytr`: local mid info becomes shared
- `str`: integrated info becomes shared

High values suggest:
- convergence toward common future structure,
- temporal homogenization,
- stabilization into a shared state.

---

## 4.6 Fragmentation Index

```text
Fragmentation = rtx + rty + stx + sty
```

**Interpretation**:  
Information that begins as a more distributed structure and ends up localized to one phase channel.

Components:
- `rtx`, `rty`: shared info breaks into one-sided persistence
- `stx`, `sty`: synergistic info collapses into phase-specific information

High values suggest:
- decomposition of global structure,
- specialization,
- breakdown of shared or integrated temporal organization.

---

## 4.7 Integration Formation

```text
Integration_formation = rts + xts + yts
```

**Interpretation**:  
Information that becomes synergistic in the future.

This measures the extent to which:
- shared information,
- phase-local information,
- or both

are being assembled into **higher-order temporal patterns**.

High values suggest:
- emergence,
- temporal binding,
- nonlinear joint predictability,
- formation of integrated motifs.

---

## 4.8 Integration Deployment

```text
Integration_deployment = str + stx + sty
```

**Interpretation**:  
Synergistic information in the past that gets expressed in simpler forms in the future.

This is not necessarily "loss." It can reflect:
- unpacking of a temporal whole,
- projection of integrated structure into local channels,
- stabilization of an emergent pattern.

High values suggest:
- top-down constraint,
- resolution of temporal integration,
- expression of holistic structure into simpler observables.

---

## 4.9 Net Integration Balance

```text
Net_integration_balance = 
(rts + xts + yts) - (str + stx + sty)
```

**Interpretation**:  
Whether the temporal window is, overall:

- **building integration** (positive),
- or **deploying / dissolving integration** (negative).

This is a very useful summary metric.

### Positive values
The embedding tends to **assemble** information into temporally integrated patterns.

### Negative values
The embedding tends to **unpack** integrated information into more local or shared forms.

### Near zero
Balanced exchange between integration formation and integration deployment.

---

## 4.10 Temporal Asymmetry

### Source-side asymmetry
```text
Source_asymmetry =
(xtr + xtx + xty + xts) -
(ytr + ytx + yty + yts)
```

**Interpretation**:  
Whether the earliest source position (`p1`) or the next source position (`p2`) contributes more strongly to the temporal information dynamics.

- positive: `p1` structurally dominates
- negative: `p2` structurally dominates

This can be useful for detecting:
- asymmetry in temporal influence,
- anticipation vs continuation,
- edge effects in the embedding.

---

### Target-side asymmetry
```text
Target_asymmetry =
(rtx + xtx + ytx + stx) -
(rty + xty + yty + sty)
```

**Interpretation**:  
Whether future information preferentially lands in the `t1`-like or `t2`-like phase channel.

This gives a measure of **future-phase imbalance**.

---

## 4.11 Shared-to-Local Decay

```text
Shared_to_local_decay = rtx + rty
```

**Interpretation**:  
Redundant information in the past that no longer remains shared in the future, but survives only in one phase channel.

This is a refined version of "erasure."

It specifically captures:
- loss of broad temporal memory,
- narrowing of a shared representation,
- partial forgetting.

---

## 4.12 Synergy-to-Local Collapse

```text
Synergy_to_local_collapse = stx + sty
```

**Interpretation**:  
Integrated temporal structure that does not remain integrated, but instead collapses into one phase channel.

This can be interpreted as:
- breakdown of holistic dynamics,
- specialization of an integrated pattern,
- local readout of a global temporal state.

---

## 4.13 Synergy Maintenance Ratio

```text
Synergy_maintenance_ratio = sts / (rts + xts + yts + sts + str + stx + sty)
```

**Interpretation**:  
Among all synergy-related dynamics, how much corresponds to **synergy remaining synergy**?

High values suggest:
- stable integrated temporal motifs,
- persistent joint structure.

Low values suggest:
- transient or unstable synergy,
- integration that is quickly created or dissolved.

---

## 4.14 Broadcast-to-Fragmentation Ratio

```text
Broadcast_fragmentation_ratio = 
(xtr + ytr + str) / (rtx + rty + stx + sty)
```

**Interpretation**:  
Whether the temporal dynamics tend more toward:

- **spreading and sharing information** across future positions,
or
- **breaking down distributed structure** into local channels.

This provides a compact measure of whether the system is becoming more globally coherent or more locally specialized.

---

# 5. Recommended Metric Families

A practical way to organize the atoms in temporal PhiID is into six families.

---

## 5.1 Persistence Family
```text
rtr, xtx, yty, sts
```

**Meaning**:  
Information preserves its structural type.

**Suggested derived metrics**:
- `Global_persistence`
- `Phase_persistence`
- `Integrated_persistence`
- `Total_persistence`

---

## 5.2 Exchange Family
```text
xty, ytx
```

**Meaning**:  
Information moves between phase channels.

**Suggested derived metrics**:
- `Phase_exchange`
- `Phase_bias`

---

## 5.3 Broadcast Family
```text
xtr, ytr, str
```

**Meaning**:  
Information becomes broadly shared in the future.

**Suggested derived metrics**:
- `Broadcast`
- `Broadcast_fragmentation_ratio`

---

## 5.4 Fragmentation Family
```text
rtx, rty, stx, sty
```

**Meaning**:  
Distributed structure becomes localized.

**Suggested derived metrics**:
- `Fragmentation`
- `Shared_to_local_decay`
- `Synergy_to_local_collapse`

---

## 5.5 Integration Formation Family
```text
rts, xts, yts
```

**Meaning**:  
Information becomes synergistic.

**Suggested derived metrics**:
- `Integration_formation`
- contribution-specific versions:
  - `Redundancy_to_integration = rts`
  - `X_to_integration = xts`
  - `Y_to_integration = yts`

---

## 5.6 Integration Deployment Family
```text
str, stx, sty
```

**Meaning**:  
Synergy is resolved into simpler future structures.

**Suggested derived metrics**:
- `Integration_deployment`
- contribution-specific versions:
  - `Integration_to_shared = str`
  - `Integration_to_X = stx`
  - `Integration_to_Y = sty`

---

# 6. Minimal Core Set of New Metrics

If only a few new metrics are kept, the following set is probably the most useful.

## Core set

### 1. Total Persistence
```text
rtr + xtx + yty + sts
```
How much information keeps the same structural identity.

### 2. Structural Lability
```text
sum of all off-diagonal atoms
```
How much information changes form across the window.

### 3. Phase Exchange
```text
xty + ytx
```
How much information swaps phase channel.

### 4. Broadcast
```text
xtr + ytr + str
```
How much information becomes broadly shared.

### 5. Fragmentation
```text
rtx + rty + stx + sty
```
How much distributed structure collapses into local channels.

### 6. Integration Formation
```text
rts + xts + yts
```
How much information becomes synergistically integrated.

### 7. Integration Deployment
```text
str + stx + sty
```
How much synergistic information is unpacked or resolved.

### 8. Net Integration Balance
```text
(rts + xts + yts) - (str + stx + sty)
```
Whether the temporal window is integration-building or integration-resolving.

---

# 7. Suggested Interpretation Table

| Metric | Atom Combination | Temporal Meaning |
|--------|------------------|------------------|
| **Global_persistence** | `rtr` | Broad memory shared across all four positions |
| **Phase_persistence** | `xtx + yty` | Persistence within each lag-2τ phase channel |
| **Integrated_persistence** | `sts` | Stable temporal integration |
| **Total_persistence** | `rtr + xtx + yty + sts` | Total preservation of information type |
| **Structural_lability** | all off-diagonals | Degree of structural reformatting |
| **Phase_exchange** | `xty + ytx` | Even/odd or cross-phase coupling |
| **Phase_bias** | `xty - ytx` | Directional phase asymmetry |
| **Broadcast** | `xtr + ytr + str` | Information becomes shared in the future |
| **Fragmentation** | `rtx + rty + stx + sty` | Distributed information becomes localized |
| **Integration_formation** | `rts + xts + yts` | Information becomes synergistically integrated |
| **Integration_deployment** | `str + stx + sty` | Synergy resolves into simpler forms |
| **Net_integration_balance** | formation - deployment | Net tendency toward integration or unpacking |
| **Source_asymmetry** | x-row minus y-row | Whether p1 or p2 dominates |
| **Target_asymmetry** | x-column minus y-column | Whether t1 or t2 dominates |

---

# 8. Multiscale Extension Across τ

These metrics become especially informative when tracked across different embedding lags `τ`.

Instead of interpreting each metric at one lag only, consider each one as a **curve across τ**:

```text
Metric(τ)
```

Examples:
- `Total_persistence(τ)`
- `Phase_exchange(τ)`
- `Net_integration_balance(τ)`
- `Broadcast(τ)`
- `Fragmentation(τ)`

This allows you to characterize:

- preferred timescales,
- scale-specific integration,
- transitions between persistent and emergent regimes,
- temporal scale invariance or scale selectivity.

### Useful summary descriptors across τ

For any metric curve:
- **peak τ**: timescale where the metric is maximal
- **center of mass across τ**: weighted characteristic timescale
- **entropy across τ**: whether the metric is narrowband or broadband
- **smoothness / rigidity across τ**: whether the structure changes gradually or abruptly
- **correlation between metrics across τ**: e.g. whether persistence and integration co-occur or trade off

This may be more informative than a single scalar TII value.

---

# 9. Practical Interpretation Examples

## High Total Persistence + Low Structural Lability
- stable temporal regime
- information stays in the same form
- simple or strongly regular dynamics

## High Phase Exchange
- alternating or oscillatory organization
- strong phase coupling across the embedding

## High Broadcast
- diverse past structures converge toward a common future structure
- temporal stabilization or global spreading

## High Fragmentation
- global or integrated structure breaks into local channels
- temporal specialization or partial breakdown of coherence

## High Integration Formation
- local/shared signals combine into irreducible temporal wholes
- emergence of temporal patterns

## High Integration Deployment
- integrated patterns are being expressed or unpacked into simpler future forms
- top-down constraint or resolution of a whole into parts

## Positive Net Integration Balance
- the system tends to build higher-order temporal structure

## Negative Net Integration Balance
- the system tends to deploy, collapse, or resolve integrated structure

---

# 10. Final Recommendation

For temporal PhiID, the most meaningful derived metrics are usually not those borrowed directly from the two-system setting, but those that capture:

1. **persistence vs reformatting**
2. **exchange vs asymmetry**
3. **broadcast vs fragmentation**
4. **integration formation vs integration deployment**
5. **their variation across τ**

A good practical default set is:

```text
Total_persistence
Structural_lability
Phase_exchange
Broadcast
Fragmentation
Integration_formation
Integration_deployment
Net_integration_balance
```

These metrics preserve the interpretability of the 16 atoms while giving a cleaner temporal vocabulary for single-process embeddings.
