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

#### Step API — deployment (no ground truth)

The controller decides; your perception pipeline produces observations.

```python
apc.begin()                                   # new task (resets belief, keeps learning)
while (action := apc.next_action()) is not None:
    obs = my_perception(action)               # 0/1 vote from the model run at `action`
    apc.observe(action, obs)                  # clarity defaults to the learned estimate
result = apc.finish()                         # result.decision in {0, 1, -1 (abstain)}
                                              # result.correct is None (truth unknown)
apc.feedback(action, correct=True)            # optional: delayed label updates the learner
```

| Method | Description |
|--------|-------------|
| `begin(max_steps=None)` | Start a task. Resets belief and conviction; keeps learned clarities. |
| `next_action() → Optional[Action]` | Next action, or `None` to stop (confident, max steps, cost budget, degradation). |
| `observe(action, obs, clarity=None, correct=None) → APCStep` | Record the 0/1 observation. Pass `clarity` only if truly known; pass `correct` if the label is already known. |
| `feedback(action, correct)` | Late label: updates the clarity estimate of `action`. |
| `finish(true_class=None) → APCResult` | End the task and return the result. |

#### `run(true_class, clarity_fn, max_steps=None, oracle_clarity=True) → APCResult`

**Simulation** of a full task against a synthetic environment (built on the step API).
`oracle_clarity=True` updates the belief with the simulator's true clarity (historical
behaviour); `False` uses the learned estimate, as in deployment.

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
| `oracle_clarity` | bool | Belief uses the simulator's true clarity (`True`) or the learned estimate (`False`) |

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

#### `select(belief, candidates, n_observations, clarity_estimates=None, emergency_clarity=None) → Optional[Action]`

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


---

## Optimal policy (`acies.optimal`)

Exact baseline for the model ACIES assumes (symmetric channel, i.i.d. observations given
the class): finite-horizon optimal stopping solved by dynamic programming over the belief.
No controller that follows this model can beat it, so it is the yardstick for any heuristic.

```python
from acies import build_standard_actions, HardwareProfile
from acies.optimal import solve_optimal, frontier, min_cost_for_error

acts, hw = build_standard_actions(), HardwareProfile.default()
clarity = {"64p": 0.55, "128p": 0.65, "224p": 0.75, "320p": 0.82, "512p": 0.88,
           "1024p": 0.93, "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92}

policy = solve_optimal(acts, clarity, hw, error_cost=1000, horizon=6)
policy.evaluate()                 # OptimalEvaluation(cost=37.4, error=0.0158, steps=3.5)
policy.action(belief=0.5, step=0) # first Action, or None to stop
min_cost_for_error(acts, clarity, hw, target_error=0.02)   # cheapest cost at <= 2 % error
frontier(acts, clarity, hw, [200, 1000, 3000])              # (λ, evaluation) points
```

`error_cost` (λ) is the price of one wrong decision in cost units. `python3 examples/optimal_gap.py`
compares `APCController` with this frontier.
