<div align="center">

<img src="assets/logo.svg" alt="ACIES" width="380">

# ACIES

### Adaptive Perception Control

A decision-theoretic controller that decides **how much perception to spend** on each input —
which resolution to run, whether to look again — to reach a reliable decision at the lowest cost.

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Go 1.22+](https://img.shields.io/badge/go-1.22+-00ADD8.svg?style=for-the-badge&logo=go&logoColor=white)](https://go.dev/)
[![Tests](https://img.shields.io/badge/tests-86%20passed-brightgreen.svg?style=for-the-badge)](#testing)
[![Benchmark](https://img.shields.io/badge/benchmark-YOLOv8n%20on%20COCO-ff6f00.svg?style=for-the-badge)](benchmarks/README.md)

<br>

*Perception is not a fixed pipeline. It is a resource to control.*

</div>

---

## What it is

ACIES is not a vision model. It sits on top of one (YOLO, a classifier, …) and answers, per input:
*run this action next, or stop and decide?* An **action** is a way of perceiving (a resolution, a
crop) with a known cost. The controller keeps a belief about the decision, learns what each action
tells it, and spends the next unit of compute only where it is expected to be worth it.

## Does it work? Measured on real YOLO output

Task: *"is there a `<class>` in the image?"* — YOLOv8n on COCO val2017, 8 classes, cost = measured CPU
latency (640 px = 96 ms). The reference is the same decision at 640 px; **no labels are used**. Settings are
chosen on separate splits and every number is measured on inputs the method never saw
([protocol and tables](benchmarks/README.md)).

**What works: a cascade with a guarantee** (`acies.cascade`, built on `acies.risk`). Learn-then-Test
turns "the thresholds look fine on a validation set" into "with probability ≥ 90 %, the accelerated system
stays within ε of the full one" — checked on 32 random splits per row:

| Guarantee you ask for | Certified splits | Compute saved | Significant violations |
|---|:--:|:--:|:--:|
| disagreement with the 640 px decision ≤ 2 % | 97 % | **60 %** | 0 / 32 |
| disagreement ≤ 3 % | 100 % | **73 %** | 0 / 32 |
| misses ≤ 10 % **and** false alarms ≤ 3 % | 66 % | 31 % | 0 / 32 |
| misses ≤ 5 % **and** false alarms ≤ 2 % | 34 % | 12 % | 0 / 32 |

**Read this honestly.**

- The disagreement guarantee is true but **weak for alerts**: for rare classes the miss rate reached 17–31 %,
  because errors on rare positives hide behind the many negatives. Use the conditional guarantee (misses /
  false alarms) for alerting; it certifies less often and saves less.
- A guarantee needs data: certifying a 5 % miss rate needs ≥ 45 calibration positives with no miss. Rare
  classes with few examples are not certified, and the honest answer is then *"no saving proven"* — the reference
  keeps running.
- **A planner is not better than a cascade.** The channel planner (`acies.channel`) reaches the same guarantee
  but saves less (11 % / 1 % symmetric, 8 % / 4 % asymmetric, vs 31 % / 12 % for the cascade). Region zoom, learned
  scene context and mAP-oriented resolution selection **did not help** either (details in the benchmarks).
- The original controller (v1) was worse than fixed-resolution YOLO on every class.
- One detector, one CPU, binary decisions. GPU/edge cost ratios and other detectors are not measured.

---

## Quick start

```bash
git clone https://github.com/NICE-DEV226/ACIES.git && cd ACIES
pip install -e ".[dev]"        # the core is pure Python, no dependencies
python3 -m pytest -q test_apc.py test_control.py test_channel.py test_risk.py test_cascade.py
```

### Deployment with a guarantee — the certified cascade

```python
from acies.cascade import CascadeSample, Ladder, calibrate_ladders

# offline, on YOLO's own outputs (no labels): first-stage score = top confidence of the class at a cheap size;
# second = reference answer at 640 px. `train` and `cert` are disjoint sets of inputs.
lad = lambda r: Ladder(CascadeSample(score_train[r], ref_train, ref_train),
                       CascadeSample(score_cert[r],  ref_cert,  ref_cert), (latency[r], latency[640]), name=str(r))
cal = calibrate_ladders([lad(160), lad(224), lad(320)], risk="conditional", eps_miss=0.10, eps_fa=0.03, delta=0.1)

if cal.certified:                                    # else: nothing proven, keep running the reference
    answer, escalated = cal.run(first_score, second=lambda: run_640_and_decide())
```

`risk="disagree"` bounds the fraction of inputs whose answer differs from the reference; `"conditional"` bounds
the miss rate and the false-alarm rate separately. Requirements: calibration and deployment inputs exchangeable,
and `train` / `cert` disjoint. See [docs/python-api.md](docs/python-api.md) (`acies.risk`, `acies.cascade`).

### The planner over channels (`acies.channel`)

Each action is a *channel*: the model's output is quantised into a few outcomes (e.g. bins of the
top detection confidence) and ACIES learns `P(outcome | class, action)` from labelled examples.

```python
from acies import HardwareProfile
from acies.actions import Action, ActionType
from acies.channel import ChannelController, ChannelConfig, ChannelLearner

# cost = your measured latency per action, in ms
actions = [Action(id=i, name=f"{r}p", action_type=ActionType.RESOLUTION,
                  base_latency_ms=ms, pixel_ratio=(r / 1024) ** 2)
           for i, (r, ms) in enumerate([(224, 30.0), (320, 40.0), (640, 84.0)])]
hw = HardwareProfile(name="ms", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)

# offline, once: outcomes[i][a] = binned output of action a on labelled image i
learner = ChannelLearner(n_actions=len(actions), n_outcomes=9).fit(outcomes, labels)

ctl = ChannelController(actions, learner, ChannelConfig(error_cost=1000, carry=0.0, hardware=hw))
ctl.begin()
while (a := ctl.next_action()) is not None:
    ctl.observe(a, my_model_outcome(a))        # no ground truth at deployment
result = ctl.finish()                          # .decision  .confidence  .total_cost  .actions_taken
```

`error_cost` is the price of one wrong decision in cost units — the single knob that trades
accuracy for compute. `carry` (0…1) discounts evidence from earlier, correlated actions
(0 = the latest, most expensive observation replaces earlier ones; best for deterministic detectors).

### The classic controller (binary clarity model)

`APCController` models each action by one number (its "clarity") and is kept for simulation and
comparison. It has the same step API without ground truth — `begin() / next_action() / observe() /
finish()` — and `run()` is a simulator built on top. See [docs/python-api.md](docs/python-api.md).

### How far from optimal? — `acies.optimal`

For the binary-symmetric model, `acies.optimal` solves the exact optimal stopping problem (dynamic
programming), a lower bound on cost that any controller in that model must respect:

```bash
python3 examples/optimal_gap.py     # classic controller vs the optimal frontier
```

### Go CLI and C++ library

A Go port of the classic simulator (`go build -o acies-cli . && ./acies-cli bench`) and a small C++
library of the belief/learner primitives (`cd cpp && make`). Both are **simulators/primitives**, not
the deployment controller: the Go CLI resets its learner on every run, and the C++ library is not
used by the Python controller (measured 0.75–0.8× the speed of pure Python through `ctypes`).
Known C++ issues: no bounds checking on action ids, an exception can cross the C ABI on invalid
sizes, and `ClarityLearner` owns a raw pointer without a copy constructor.

---

## How it works

| Step | Component | What it does |
|:----:|-----------|--------------|
| 1 | **Channels** (`ChannelLearner`) | learns `P(outcome | class, action)` with a Dirichlet posterior, and the class prior |
| 2 | **Belief** | Bayes update in log-odds; `carry` discounts redundant, correlated evidence |
| 3 | **Plan** (`ChannelController`) | dynamic programming over (belief, lowest action still allowed): run the next action only if its expected error reduction, priced by `error_cost`, exceeds its cost |
| 4 | **Never repeat** | perception is deterministic: re-running an action adds no information |
| 5 | **Decision** | Bayes decision from the belief; optional abstention |

### What was wrong with the first version

1. **One number per action** treats every observation as an independent coin flip. A deterministic
   model run twice returns the same answer, but the controller counted it as new evidence and paid again.
2. **Hard 0/1 votes** discard what the detector says: `0.02` and `0.24` are both "no" but mean very different things.
3. **An emergency override triggered on the untouched prior** (risk 5.0 ≥ 4.0) and forced the most expensive action on every task. Fixed.
4. **Greedy value-of-information per cost** starts with the cheapest action even when starting higher would avoid a second pass. Replaced by a plan.

## Known limitations

- **Change-point detection (BOCPD) never fires.** With a constant hazard rate `P(r_t = 0)` equals the hazard rate identically, so the alarm cannot trigger. The module is disabled by default and should not be relied on.
- **Multi-class belief** (`acies.multiclass`) is a pseudo-count scheme that saturates at the per-observation clarity instead of converging like a Bayesian posterior.
- Repeated observations are modelled as independent given the class in the classic controller; real errors are correlated across resolutions.
- Simulated benchmarks of earlier versions (synthetic MNIST "clarity" tables) were removed: they did not measure a real model.

## Testing

```bash
python3 -m pytest -q test_apc.py test_control.py test_channel.py test_risk.py test_cascade.py     # 86 tests
```

Highlights: the risk-control guarantee is checked empirically (a naive "empirical risk ≤ ε" rule violates the bound in
39 % of simulated calibration sets, Learn-then-Test in 3 %); equivalence of the channel model with the binary Bayes update
when K = 2; a synthetic environment where planning beats greedy by ~25 %; the optimal policy cross-checked by Monte Carlo
against the real `BeliefState`; regression tests for the emergency-override defect.

## Reproducing the benchmarks

```bash
python3 benchmarks/coco_subset.py --n 3000 --seed 0
python3 benchmarks/measure.py --weights yolov8n.pt --subset ~/.cache/acies-bench/coco/subset_3000_0.json
python3 benchmarks/certified_cascade_demo.py --measure <measure_*.pkl> --subset <subset_*.json> --latency-json benchmarks/results/latency_single_session.json
python3 benchmarks/certified_eval.py --mode conditional ...   # all methods under the same certified protocol
```

Model weights are Ultralytics (AGPL-3.0) and are not distributed with this repository.

## Project structure

```
ACIES/
├── acies/                  # Python package
│   ├── risk.py             # Learn-then-Test risk control (the guarantee)
│   ├── cascade.py          # certified two-stage cascade            ← recommended
│   ├── channel.py          # planner over learned channels
│   ├── optimal.py          # exact optimal policy (binary model)
│   ├── selector.py         # contextual resolution selector (experimental)
│   ├── controller.py       # classic controller + simulator
│   ├── belief.py  clarity_learner.py  safety.py  conviction.py
│   ├── change_point.py     # BOCPD (does not fire, see limitations)
│   ├── multiclass.py  actions.py  accelerator.py
├── benchmarks/             # real benchmark on COCO + YOLO (protocol, results)
├── cpp/                    # C++ belief/learner primitives
├── core.go  main.go        # Go port of the classic simulator + CLI
├── test_apc.py  test_control.py  test_channel.py  test_risk.py  test_cascade.py
├── examples/  docs/  paper/  assets/
└── Dockerfile  Makefile
```

## Documentation

[Architecture](docs/architecture.md) · [Configuration](docs/configuration.md) · [Python API](docs/python-api.md)
· [C++ API](docs/cpp-api.md) · [CLI](docs/cli-reference.md) · [Benchmarks](docs/benchmarks.md)
· [Contributing](docs/contributing.md)

## License

MIT License — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{acies2026,
  title = {ACIES: Adaptive Perception Control},
  year = {2026},
  url = {https://github.com/NICE-DEV226/ACIES}
}
```
