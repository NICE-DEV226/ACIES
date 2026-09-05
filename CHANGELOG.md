# Changelog

All notable changes to ACIES will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

## [Unreleased]

### Planned
- CIFAR-10 benchmark (training too slow, pending)
- FGSM adversarial robustness testing
- Literature baseline comparison (cascade done, more needed)
