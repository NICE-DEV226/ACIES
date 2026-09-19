# Changelog

All notable changes to ACIES will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- **Emergency override no longer hijacks the untouched prior.** At belief 0.5 the risk is
  already 10·0.5 = 5.0 ≥ `emergency_risk` (4.0), so every task started with a forced,
  most-expensive `1024p` step and the Thompson/UCB/conviction scoring was bypassed. The
  override now applies only after at least one observation, ranks actions by *learned*
  clarity (a resolution-monotone prior only for untried actions), and is counted once.
  Default config, 6000 runs: threshold 0.92 → cost 187 → 96 with error 7.1 % → 3.3 %;
  threshold 0.99 → cost 274 → 167 at equal error (1.3 %). Same fix in the Go core
  (bench cost 579 → 283, accuracy 92.6 % → 86 % on its 0.85-scaled clarity table).
- `APCResult.n_emergency` was a cumulative counter across runs; it is now per run.
- Conviction state was never reset between tasks (its history grew without bound).
- Degraded-image detection now accounts for the cost of the observation that triggered it.

### Added
- **Step API** for deployment without ground truth: `begin()`, `next_action()`,
  `observe(action, obs, clarity=None, correct=None)`, `feedback()`, `finish()`.
  `run()` is now a simulator built on top of it; `run(..., oracle_clarity=False)` updates
  the belief with the learned clarity instead of the simulator's true value.
  `APCResult.correct` is `None` when the true class is unknown.
- **`acies.optimal`**: exact optimal stopping policy (dynamic programming over the belief)
  for ACIES's own observation model — `solve_optimal`, `frontier`, `min_cost_for_error`.
  It is a true lower bound on cost at a given error; `examples/optimal_gap.py` compares
  the controller with it.
- `test_control.py` (24 tests): emergency regression, optimal policy (closed form, Monte
  Carlo cross-check, monotone frontier, lower-bound property), step API lifecycle.

### Added (certified acceleration)
- **`acies.risk`**: Learn-then-Test risk control, pure Python. Exact binomial p-values (Hoeffding-Bentkus for
  losses in [0, 1]), fixed-sequence and Bonferroni family-wise control, `certify_pvalues` for conjunctions
  (intersection-union). The guarantee is checked empirically in tests: a naive "empirical risk <= eps" rule violates the
  bound in 39 % of simulated calibration sets, Learn-then-Test in 3 % (target <= 10 %).
- **`acies.cascade`**: a two-stage cascade calibrated with a guarantee, no labels needed
  (`risk="disagree"` or `"conditional"` for misses / false alarms), several candidate first stages in one test sequence.
  On YOLOv8n / COCO (8 classes x 4 splits): 60 % of compute saved at a guaranteed disagreement <= 2 %,
  31 % at misses <= 10 % and false alarms <= 3 %; 0 significant violations in 32 splits per row.
- `ChannelConfig(miss_cost=rho)` (asymmetric costs), `first_action`, and an external belief on `observe(..., belief=)`.
- `benchmarks/certified_eval.py`, `certified_cascade_demo.py`, `measure_zoom*.py`; negative results (planner < cascade,
  zoom, context, mAP selection) documented in `benchmarks/README.md`.
- `test_risk.py` (20 tests) and `test_cascade.py` (9 tests).

### Added (real-perception work)
- **`acies.channel`**: `ChannelController` + `ChannelLearner`. Each action is a *channel*
  P(outcome | class, action) learned from labelled examples (quantised model output, not a 0/1
  vote); class prior learned from data; value-of-information / cost with a dynamic-programming
  plan over (belief, lowest action still allowed); never repeats an action (deterministic
  perception); `carry` discounts evidence from correlated actions. Same step API as the classic
  controller. On real YOLOv8n detections it replaces a controller that was dominated by
  fixed-resolution YOLO on every class (see `benchmarks/README.md`).
- **`acies.selector`**: `ResolutionSelector` (pure-Python ridge on degree-2 features) and
  `detection_features`. Experimental: it does not beat fixed resolution on mAP.
- **`benchmarks/`**: reproducible real benchmark on COCO val2017 (measurement of YOLO at 8
  resolutions with real latency, mAP50-95 evaluator matching Ultralytics, calibration/test split,
  paired bootstrap, cascade and oracle baselines).
- `test_channel.py` (15 tests) incl. equivalence with the binary model when K = 2 and a
  synthetic environment where planning beats greedy by ~25 %.

### Changed
- `make test`, CI and the Docker build now run pytest on `test_apc.py` and `test_control.py`
  (`python3 test_apc.py` executed no test and always exited 0; the Docker stage lacked pytest).

### Known limitations (not addressed here)
- Repeated observations are modelled as i.i.d. given the class and independent across
  actions; real perception errors are correlated (hard image ⇒ hard at every resolution).
- The UCB bonus still over-explores: the controller remains ~3× above the optimal cost.
- BOCPD never fires (P(r=0) ≡ hazard rate for a constant hazard); multi-class belief
  saturates at the per-observation clarity; the C++ accelerator is unused and slower than
  pure Python; README/paper benchmark figures do not measure the shipped controller.

### Planned
- CIFAR-10 benchmark (training too slow, pending)
- FGSM adversarial robustness testing
- Literature baseline comparison (cascade done, more needed)

## [0.2.0] — 2026-09-05

### Added
- **Abstention mechanism**: ACIES can now say "I don't know" when confidence < 0.6
- **Cost budget cap**: max cost per image (default 600, configurable via `max_cost_per_image`)
- **Degradation detector**: 3 consecutive low-clarity steps triggers early stop
- New result fields: `abstained`, `degraded`, `cost_budget_exceeded`, `avg_clarity`
- Stress test suite: `test/stress_test_robust.py`
- **YOLO + ACIES integration**: adaptive object detection (`demo_yolo.py`)
- YOLO benchmark: fixed vs adaptive resolution comparison (`benchmark_yolo.py`)

### Changed
- `APCResult.decision` returns `-1` when abstaining (was always 0 or 1)
- `APCConfig` now includes robustness parameters

### Performance
- Blur kernel=7: cost reduced from 1058 → 530 (-50%)
- Hard images: system abstains instead of wasting resources
- "Decided" accuracy: 83-100% (only speaks when confident)

## [0.1.4] — 2026-09-05

### Added
- CI/CD: GitHub Actions (test, build, benchmark, publish, release assets)
- Community: 6 GitHub issues, issue templates, PR template, CONTRIBUTING.md, SECURITY.md
- ArXiv paper: `paper/acies.tex` + `references.bib`
- Release v0.1.4 published to PyPI

## [0.1.3] — 2026-09-05

### Added
- Professional README with logo, badges, comparison tables
- Logo: concentric circles SVG (`assets/logo.svg`)
- Architecture diagram (`assets/architecture.svg`)
- Favicon assets

### Changed
- Package renamed to `acies` on PyPI

## [0.1.2] — 2026-09-05

### Added
- Real CNN benchmark: 1000 mixed-difficulty images with PyTorch inference
- Cascade classifier baseline for comparison
- Benchmark results: ACIES 97.3% vs Fixed 81.9% vs Cascade 80.0%

## [0.1.1] — 2026-09-04

### Added
- Simulated benchmark: MNIST with Thompson Sampling
- 6 hardware profiles: edge_tpu, rpi5, default, gpu, jetson, laptop
- Go CLI: 32,800 runs/sec

### Changed
- Conviction zone: renamed `force_commit_clarity` → `clarity_threshold`
- Removed dead code in controller.py (lines 301-304)

## [0.1.0] — 2026-09-04

### Added
- Initial release
- Core modules: controller, belief, clarity_learner, safety, conviction, actions, change_point
- Thompson Sampling with Beta(2,2) conservative prior
- Bayesian belief tracking
- Safety layer: 5 decision rules
- Conviction anti-oscillation mechanism
- Bayesian Online Change-Point Detection (disabled by default)
- C++ library: `libacies.so`
- Python package: `pip install acies`
- 13 pytest tests (all passing)
