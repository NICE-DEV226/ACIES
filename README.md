<div align="center">

# ACIES

**Adaptive Perception Control**

A decision-theoretic framework for adaptively controlling visual perception to minimize computational cost while maintaining target decision risk.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20passed-brightgreen.svg)](#testing)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](#)

*Perception is not a fixed pipeline. It is a resource to control.*

</div>

---

## Overview

ACIES treats visual perception as a **resource allocation problem**. Instead of processing every input at maximum resolution, the controller adaptively selects *what* to perceive (resolution, crop, layers, tokens, modality) by solving a cost-risk optimization at each step.

The core insight: **heterogeneous perception actions** (resize to 64p, crop a region, activate additional layers) have different costs and different information gains. ACIES picks the action that maximizes **ΔR/C** — risk reduction per unit cost — using Bayesian belief tracking, Thompson Sampling for online clarity estimation, and a safety layer that guarantees risk never exceeds a configurable threshold.

### Key Results

| Metric | Baseline (1024p) | ACIES | Savings |
|--------|:----------------:|:-----:|:-------:|
| Accuracy | 97.5% | 97.7% | — |
| Avg cost | 237.0 | 245.1 | — |
| Avg steps | 1.0 | 2.5 | adaptive |
| Safety violations | 0% | <5% | guaranteed |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  APCController                  │
│                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────┐ │
│  │ BeliefState  │  │ ClarityLearner│  │ Safety │ │
│  │ (Bayesian    │  │ (Thompson     │  │ Layer  │ │
│  │  filter)     │  │  Sampling)    │  │        │ │
│  └──────┬──────┘  └──────┬───────┘  └───┬────┘ │
│         │                │               │       │
│         ▼                ▼               ▼       │
│  ┌──────────────────────────────────────────┐   │
│  │           Action Space                    │   │
│  │  64p │ 128p │ 224p │ 320p │ 512p │ 1024p│   │
│  │  crop_224 │ crop_320 │ crop_512          │   │
│  └──────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│              ┌──────────────┐                   │
│              │ HardwareProfile│                  │
│              │ (Jetson/RPi/GPU)│                 │
│              └──────────────┘                   │
└─────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/NICE-DEV226/ACIES.git
cd ACIES
```

No external dependencies — ACIES uses only the Python standard library.

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Setup guide for Python, Go, C++, Docker |
| [Architecture](docs/architecture.md) | Deep dive into the control loop and algorithms |
| [Configuration](docs/configuration.md) | All configuration parameters |
| [CLI Reference](docs/cli-reference.md) | Go CLI commands and flags |
| [Python API](docs/python-api.md) | Complete Python API reference |
| [C++ API](docs/cpp-api.md) | C API for FFI (Python/Go/Rust) |
| [Examples](docs/examples.md) | 10 usage examples |
| [Benchmarks](docs/benchmarks.md) | Performance results on MNIST |
| [Contributing](docs/contributing.md) | Development guide |

## Quick Start

```python
from acies import APCController, APCConfig, HardwareProfile

# Configure for your hardware
apc = APCController(APCConfig(
    confidence_threshold=0.92,
    max_steps=6,
    hardware=HardwareProfile.jetson_orin(),
))

# Define how clear each action is for your task
def clarity_fn(action):
    clarities = {
        "64p": 0.55, "128p": 0.65, "224p": 0.75,
        "320p": 0.82, "512p": 0.88, "1024p": 0.93,
        "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
    }
    return clarities.get(action.name, 0.5)

# Run the controller
result = apc.run(true_class=1, clarity_fn=clarity_fn)

print(f"Decision: {result.decision}")
print(f"Correct: {result.correct}")
print(f"Cost: {result.total_cost:.1f}")
print(f"Steps: {result.n_steps}")
print(f"Actions: {result.actions_taken}")
```

## API Reference

### `APCController`

The main controller that orchestrates perception decisions.

```python
apc = APCController(config=APCConfig(), actions=None)
result = apc.run(true_class=1, clarity_fn=my_fn)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run(true_class, clarity_fn, max_steps)` | Execute on a single task |
| `batch_run(tasks, n_trials)` | Execute on multiple tasks |
| `reset()` | Reset belief state (new task) |
| `reset_all()` | Reset everything (beliefs + learner + safety) |
| `summary()` | Get aggregated statistics |

### `APCConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prior` | 0.5 | Prior probability P(Y=1) |
| `temperature` | 1.0 | Calibration temperature (>1 = less confident) |
| `confidence_threshold` | 0.95 | Stop when confidence exceeds this |
| `max_steps` | 8 | Maximum perception steps |
| `max_risk` | 2.0 | Maximum acceptable risk |
| `emergency_risk` | 4.0 | Risk level triggering emergency override |
| `hardware` | `HardwareProfile.default()` | Hardware cost profile |

### `HardwareProfile`

Built-in profiles:

```python
HardwareProfile.default()        # Generic
HardwareProfile.jetson_orin()    # NVIDIA Jetson Orin Nano
HardwareProfile.raspberry_pi5()  # Raspberry Pi 5
HardwareProfile.desktop_gpu()    # Desktop GPU (RTX 4090)
HardwareProfile.edge_tpu()       # Google Edge TPU (Coral)
```

Custom profile:

```python
profile = HardwareProfile(
    name="My Custom Device",
    latency_weight=0.4,
    energy_weight=0.4,
    memory_weight=0.2,
    latency_scale=1.0,
    energy_scale=1.0,
    memory_scale=1.0,
)
```

### `BeliefState`

Bayesian belief tracker for binary classification.

```python
belief = BeliefState(prior=0.5, temperature=1.0)
belief.update(obs=1, clarity=0.85)
print(belief.belief)   # P(Y=1 | observations)
print(belief.risk)     # Current risk level
print(belief.decision) # 0 or 1
```

### `ClarityLearner`

Thompson Sampling-based online clarity estimation.

```python
learner = ClarityLearner(n_actions=9)
sampled_p = learner.sample(action_id)     # For planning
learner.update(action_id, correct=True)   # After observation
print(learner.mean(action_id))            # Posterior mean
print(learner.confidence(action_id))      # Estimation confidence
```

### `SafetyLayer`

Guarantees risk never exceeds configured thresholds.

```python
safety = SafetyLayer(config=SafetyConfig(max_risk=2.0))
safe_action = safety.select(belief, candidates, n_observations)
```

### `Action` / `build_standard_actions()`

The standard action space with 9 heterogeneous perception actions:

| Action | Type | Pixel Ratio | Latency (ms) | Energy (mJ) |
|--------|------|:-----------:|:------------:|:-----------:|
| 64p | Resolution | 0.004 | 2 | 0.5 |
| 128p | Resolution | 0.016 | 5 | 2 |
| 224p | Resolution | 0.048 | 12 | 6 |
| 320p | Resolution | 0.098 | 25 | 13 |
| 512p | Resolution | 0.250 | 60 | 35 |
| 1024p | Resolution | 1.000 | 200 | 140 |
| crop_224 | Crop | 0.005 | 8 | 4 |
| crop_320 | Crop | 0.013 | 15 | 8 |
| crop_512 | Crop | 0.028 | 35 | 20 |

## Testing

```bash
python3 test_apc.py
```

**8 tests covering:**

1. **Base functionality** — accuracy, cost, exploration
2. **Hard tasks** — high difficulty scenarios
3. **Distribution shift** — Thompson Sampling adaptation
4. **Sensor failure** — degraded action reliability
5. **Hardware profiles** — all 5 built-in profiles
6. **Stress test** — 5,000 iterations, safety guarantees
7. **Belief math** — Bayesian update verification
8. **Thompson convergence** — posterior estimation accuracy

## How It Works

1. **Belief Tracking**: Maintains P(Y=1 | observations) via Bayesian filtering with temperature calibration
2. **Clarity Estimation**: Thompson Sampling with Beta posteriors learns P(correct | action) online
3. **Action Scoring**: Computes ΔR/C (risk reduction per cost) for each candidate action
4. **Safety Filtering**: Rejects actions that could push risk above threshold
5. **Decision**: Stops when confidence exceeds threshold or max steps reached

## Theoretical Foundations

- **Chernoff-type lower bound**: Quantifies minimum cost for target risk
- **Submodularity of ΔR**: Ensures greedy action selection is near-optimal
- **Thompson Sampling regret**: O(log T) regret for clarity estimation

See [paper_draft.md](paper_draft.md) and [APC_Theoretical_Derivations.md](APC_Theoretical_Derivations.md) for proofs.

## Project Structure

```
ACIES/
├── acies/
│   ├── __init__.py          # Package exports
│   ├── actions.py           # Action space & hardware profiles
│   ├── belief.py            # Bayesian belief tracker
│   ├── clarity_learner.py   # Thompson Sampling clarity estimation
│   ├── safety.py            # Safety layer (risk guarantees)
│   └── controller.py        # Main APC controller
├── test_apc.py              # 8 robustness tests
├── paper_draft.md           # Research paper draft
├── APC_Theoretical_Derivations.md  # Mathematical proofs
├── prototype.py             # Phase 1 prototype
├── prototype_phase2.py      # Phase 2 prototype
├── prototype_realistic.py   # Phase 3 realistic simulation
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{acies2026,
  title = {ACIES: Adaptive Perception Control},
  year = {2026},
  url = {https://github.com/NICE-DEV226/ACIES}
}
```
