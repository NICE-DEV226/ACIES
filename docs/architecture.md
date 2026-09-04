# Architecture

## Overview

ACIES treats visual perception as a **resource allocation problem**. Instead of processing every input at maximum resolution, the controller adaptively selects what to perceive by solving a cost-risk optimization at each step.

```
Input (image/frame)
    │
    ▼
┌─────────────────────────────────────────────────┐
│                 APCController                    │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│  │  BeliefState  │  │ClarityLearner│  │ Safety │ │
│  │              │  │              │  │ Layer  │ │
│  │ P(Y=1|obs)  │  │ Thompson     │  │        │ │
│  │ Bayesian    │←→│ Sampling     │←→│ Risk   │ │
│  │ filter      │  │ Beta(2,2)    │  │ guard  │ │
│  └──────┬───────┘  └──────┬──────┘  └───┬────┘ │
│         │                 │              │       │
│  ┌──────┴───────┐  ┌──────┴──────┐      │       │
│  │  Conviction   │  │Change Point │      │       │
│  │  Anti-        │  │  Detection  │      │       │
│  │  oscillation  │  │  Bayesian   │      │       │
│  └──────────────┘  └─────────────┘      │       │
│                                         │       │
│  ┌──────────────────────────────────────┘       │
│  │         Action Space                         │
│  │  64p │ 128p │ 224p │ 320p │ 512p │ 1024p   │
│  │  crop_224 │ crop_320 │ crop_512             │
│  └──────────────────────────────────────────┐   │
│                                              │   │
│  ┌──────────────────────────────────────────┘   │
│  │         HardwareProfile                      │
│  │  Jetson │ RPi │ GPU │ TPU │ Default          │
│  └──────────────────────────────────────────────┘
└─────────────────────────────────────────────────┘
    │
    ▼
Decision (class + confidence)
```

## Control Loop

Each perception step follows this cycle:

```
1. SAMPLE    → Thompson Sampling estimates clarity for each action
2. SCORE     → Compute ΔR/C (risk reduction per unit cost) for each action
3. ADJUST    → Conviction mechanism boosts high-clarity actions near threshold
4. FILTER    → Safety layer rejects actions that could exceed risk threshold
5. EXECUTE   → Run the selected action (e.g., resize image to 224p)
6. OBSERVE   → Get observation from the environment
7. UPDATE    → Update Bayesian belief and Thompson posteriors
8. CHECK     → Change-point detector monitors for distribution shifts
9. DECIDE    → If confidence ≥ threshold, output decision; otherwise loop
```

## Belief Tracking

The `BeliefState` maintains P(Y=1 | observations) using exact Bayesian updating:

```
P(Y=1 | obs) = P(obs | Y=1) * P(Y=1) / P(obs)
```

Where:
- `P(Y=1)` is the current belief (starts at prior)
- `P(obs | Y=1) = clarity` if obs=1, else `1 - clarity`
- `P(obs) = P(obs | Y=1) * P(Y=1) + P(obs | Y=0) * P(Y=0)`

Temperature calibration compensates for miscalibrated models:
```
logit_adjusted = logit / temperature
```

## Thompson Sampling

The `ClarityLearner` estimates P(correct | action) for each action using Beta-Bernoulli Thompson Sampling:

- **Prior**: Beta(2,2) — conservative, centers initial estimates at 0.5
- **Update**: After observing correctness, increment α (correct) or β (incorrect)
- **Sample**: Draw from posterior Beta(α, β) for exploration

This provides:
- Natural exploration/exploitation tradeoff
- Calibrated uncertainty that decreases with observations
- O(|A|) per step — negligible computational cost

## Safety Layer

The safety layer is a hard constraint that never lets the controller make a decision that could exceed the risk threshold:

```
if current_risk ≥ emergency_risk:
    → Force most informative action (emergency override)
elif expected_risk(action) > max_risk:
    → Reject action
elif confidence ≥ threshold:
    → STOP (decision made)
else:
    → Execute best safe action
```

## Conviction Mechanism

Prevents oscillation when confidence is near the threshold:

- Detects "conviction zone" (confidence ∈ [0.90, 1.0])
- Boosts scores of high-clarity actions (×1.3)
- Penalizes low-clarity actions (×0.5)
- Safety layer remains final authority

## Change-Point Detection

Bayesian Online Change-Point Detection (Adams & MacKay 2007):

- Maintains P(run_length = k | observations) for each action
- When P(run_length = 0) > threshold → change point detected
- Resets the Thompson posterior for the affected action
- Adapts to sudden shifts in image quality or content

## Action Space

| Action | Type | Resolution | Pixel Ratio | Latency (ms) | Energy (mJ) |
|--------|------|:----------:|:-----------:|:------------:|:-----------:|
| 64p | Resolution | 64×64 | 0.4% | 2 | 0.5 |
| 128p | Resolution | 128×128 | 1.6% | 5 | 2 |
| 224p | Resolution | 224×224 | 4.8% | 12 | 6 |
| 320p | Resolution | 320×320 | 9.8% | 25 | 13 |
| 512p | Resolution | 512×512 | 25.0% | 60 | 35 |
| 1024p | Resolution | 1024×1024 | 100% | 200 | 140 |
| crop_224 | Crop | 224×224 | 0.5% | 8 | 4 |
| crop_320 | Crop | 320×320 | 1.3% | 15 | 8 |
| crop_512 | Crop | 512×512 | 2.8% | 35 | 20 |

## Hardware Profiles

| Profile | Latency | Energy | Memory | Use Case |
|---------|:-------:|:------:|:------:|----------|
| default | 1.0× | 1.0× | 1.0× | Generic |
| jetson | 0.6× | 0.8× | 1.0× | NVIDIA Jetson Orin Nano |
| rpi | 2.5× | 0.4× | 0.8× | Raspberry Pi 5 |
| gpu | 0.1× | 3.0× | 2.0× | Desktop GPU (RTX 4090) |
| tpu | 0.3× | 0.1× | 0.3× | Google Edge TPU (Coral) |
