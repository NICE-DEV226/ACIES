# Python API Reference

## Module: `acies`

```python
from acies import (
    APCController, APCConfig, APCResult, APCStep,
    BeliefState,
    ClarityLearner, BetaPosterior,
    SafetyLayer, SafetyConfig, SafetyState,
    Conviction, ConvictionConfig, ConvictionState,
    ChangePointDetector, ChangePointConfig,
    Action, ActionType, HardwareProfile, build_standard_actions,
)
```

---

## APCController

Main controller that orchestrates perception decisions.

### Constructor

```python
APCController(config: APCConfig = None, actions: List[Action] = None)
```

### Methods

#### `run(true_class, clarity_fn, max_steps=None) → APCResult`

Execute APC on a single task.

```python
result = apc.run(
    true_class=1,
    clarity_fn=lambda a: 0.85,  # clarity for each action
    max_steps=6,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `true_class` | int | True class label (0 or 1) |
| `clarity_fn` | Callable[[Action], float] | Function returning clarity for an action |
| `max_steps` | int | Maximum steps (overrides config) |

#### `batch_run(tasks, n_trials=1) → List[APCResult]`

Execute on multiple tasks.

```python
tasks = [(1, 0.0, clarity_fn), (0, 0.5, clarity_fn)]
results = apc.batch_run(tasks, n_trials=10)
```

#### `reset()`

Reset belief state for a new task.

#### `reset_all()`

Reset everything (beliefs + learner + safety + history).

#### `summary() → dict`

Get aggregated statistics.

```python
{
    "n_runs": 100,
    "avg_cost": 248.63,
    "avg_accuracy": 0.993,
    "avg_latency_ms": 45.2,
    "avg_epc": 250.4,
    "learner": {...},
    "safety": {...},
    "conviction": {...},
    "change_points": {...},
}
```

---

## APCResult

Result of a single APC execution.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `decision` | int | Final decision (0 or 1) |
| `correct` | bool | Was the decision correct? |
| `total_cost` | float | Total computational cost |
| `total_latency_ms` | float | Total latency in milliseconds |
| `total_energy_mJ` | float | Total energy in millijoules |
| `peak_memory_MB` | float | Peak memory usage |
| `n_steps` | int | Number of perception steps taken |
| `n_emergency` | int | Number of emergency overrides |
| `steps` | List[APCStep] | Detailed step history |
| `final_belief` | float | Final P(Y=1) |
| `final_risk` | float | Final risk level |
| `abstained` | bool | Did the controller abstain? |
| `actions_taken` | List[str] | Names of actions taken |

### Methods

#### `summary() → dict`

```python
result.summary()
# {"decision": 1, "correct": True, "total_cost": 147.1, ...}
```

---

## APCStep

Detailed record of a single perception step.

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Step number (0-indexed) |
| `action` | Action | Action that was executed |
| `observation` | int | Observation received (0 or 1) |
| `belief_before` | float | Belief before this step |
| `belief_after` | float | Belief after this step |
| `risk_before` | float | Risk before this step |
| `risk_after` | float | Risk after this step |
| `score` | float | ΔR/C score for this action |
| `clarity_sampled` | float | Thompson-sampled clarity |
| `clarity_true` | float | True clarity (from clarity_fn) |
| `cost` | float | Cost of this action |
| `latency_ms` | float | Latency of this action |
| `safe` | bool | Was this action safe? |

---

## BeliefState

Bayesian belief tracker for binary classification.

### Constructor

```python
BeliefState(prior=0.5, temperature=1.0)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `belief` | float | Current P(Y=1 \| observations) |
| `risk` | float | Current risk level (0-5) |
| `confidence` | float | Confidence in decision (0-1) |
| `decision` | int | Optimal decision (0 or 1) |
| `entropy` | float | Belief entropy (uncertainty) |
| `n_updates` | int | Number of updates performed |

### Methods

#### `update(obs, clarity)`

Update belief with a binary observation.

```python
belief.update(obs=1, clarity=0.85)
```

#### `delta_risk(clarity) → float`

Expected risk reduction for an action with this clarity.

#### `delta_risk_efficiency(clarity, cost) → float`

Risk reduction per unit cost: ΔR/C.

#### `reset()`

Reset to prior.

---

## ClarityLearner

Thompson Sampling-based online clarity estimation.

### Constructor

```python
ClarityLearner(n_actions: int)
```

### Methods

#### `sample(action_id) → float`

Sample estimated clarity for planning.

#### `update(action_id, correct: bool)`

Update posterior with observation result.

#### `reset_posterior(action_id)`

Reset posterior for one action (after change point).

#### `mean(action_id) → float`

Posterior mean (no sampling).

#### `confidence(action_id) → float`

Confidence in the estimation.

#### `best_action() → int`

Action with highest mean clarity.

#### `exploration_ratio() → float`

Ratio of least-observed to most-observed action.

---

## SafetyLayer

Guarantees risk never exceeds configured thresholds.

### Constructor

```python
SafetyLayer(config: SafetyConfig = None)
```

### Methods

#### `select(belief, candidates, n_observations, clarity_estimates=None) → Optional[Action]`

Select a safe action from candidates. Returns `None` if abstention is preferred.

#### `check_post_action(belief, action) → bool`

Check if an action was safe after execution.

#### `should_abstain(belief, n_observations) → bool`

Should the controller abstain from making a decision?

---

## Conviction

Anti-oscillation mechanism for the conviction zone.

### Constructor

```python
Conviction(config: ConvictionConfig = None)
```

### Methods

#### `update(confidence)`

Update conviction state with current confidence.

#### `adjust_scores(scores, clarity_estimates) → list`

Adjust ΔR/C scores when in the conviction zone.

#### `should_force_commit() → bool`

Should we force a decision due to oscillation?

---

## ChangePointDetector

Bayesian online change-point detection.

### Constructor

```python
ChangePointDetector(n_actions: int, config: ChangePointConfig = None)
```

### Methods

#### `update(action_id, observation) → bool`

Update with new observation. Returns `True` if change point detected.

#### `get_clarity_stats(action_id, window=20) → Optional[Tuple[float, float]]`

Get (mean, variance) of recent observations.

---

## Action / HardwareProfile

### Action

| Property | Type | Description |
|----------|------|-------------|
| `id` | int | Unique identifier |
| `name` | str | Human-readable name |
| `action_type` | ActionType | RESOLUTION or CROP |
| `resolution` | int | Image resolution |
| `crop_area_ratio` | float | Crop area ratio |
| `pixel_ratio` | float | Pixels processed / max pixels |
| `cost(profile)` | float | Compute cost on given hardware |

### HardwareProfile

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Profile name |
| `latency_weight` | float | Weight for latency |
| `energy_weight` | float | Weight for energy |
| `memory_weight` | float | Weight for memory |
| `latency_scale` | float | Latency multiplier |
| `energy_scale` | float | Energy multiplier |
| `memory_scale` | float | Memory multiplier |

### Built-in Profiles

```python
HardwareProfile.default()        # Generic
HardwareProfile.jetson_orin()    # Jetson Orin Nano
HardwareProfile.raspberry_pi5()  # Raspberry Pi 5
HardwareProfile.desktop_gpu()    # RTX 4090
HardwareProfile.edge_tpu()       # Coral TPU
```
