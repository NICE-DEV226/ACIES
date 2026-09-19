# Contributing to ACIES

Thanks for your interest. **Read [`docs/STATUS.md`](STATUS.md) first**: it says what is proven, what failed, what nobody
has measured yet, and where we want your ideas.

ACIES makes a black-box perception model cheaper to run **with a finite-sample guarantee** against the full-cost system,
and reports plainly when it cannot prove a saving. It is a small tested library plus an honest benchmark — not a
finished product.

## Quick start (5 minutes)

```bash
git clone https://github.com/NICE-DEV226/ACIES.git && cd ACIES
pip install -e ".[dev]"
python3 -m pytest -q test_apc.py test_control.py test_channel.py test_risk.py test_cascade.py   # 86 tests
```

Optional: `go build -o acies-cli .` (Go simulator CLI), `cd cpp && make` (C++ primitives; see known defects in STATUS.md).
Real benchmarks need Ultralytics and a dataset subset: see [`benchmarks/README.md`](../benchmarks/README.md).

## Ways to contribute

- **Ideas and experiments** — the most valuable contribution. Pick one of the "Ideas wanted" in [STATUS.md](STATUS.md) or
  open an issue with the **Research idea** template. A well-run experiment that fails is as welcome as one that works.
- **Reproduce on your hardware or runtime** (GPU, TensorRT, OpenVINO, Jetson, Raspberry Pi) — including the cases where
  ACIES saves nothing.
- **Fix known defects** (table in STATUS.md): C++ memory safety, BOCPD, multi-class belief, Go flag parser.
- **Good first issues**: tests for a module, docs, the Go/C++ defects above, a wrapper for a detector API.

## Code structure

```
acies/
  risk.py          Learn-then-Test risk control (the guarantee)          <- start here
  cascade.py       certified two-stage cascade                            <- and here
  channel.py       planner over learned perception channels
  optimal.py       exact optimal policy for the binary model (lower bound)
  selector.py      contextual resolution selector (experimental)
  controller.py    classic controller + simulator (legacy, worse on real data)
  belief.py  clarity_learner.py  safety.py  conviction.py  actions.py
  change_point.py  BOCPD (does not fire, see STATUS.md)      multiclass.py  accelerator.py
benchmarks/        real benchmark: measurement, evaluators, certified evaluation, results
cpp/  core.go  main.go     C++ primitives, Go simulator port
test_*.py          test_risk, test_cascade, test_channel, test_control, test_apc
```

## Ground rules (they exist because we were wrong once)

1. **Compare against the strong baseline**: fixed resolution *and* a tuned two-stage cascade.
2. **Keep calibration, certification and test disjoint.**
3. **Measure costs on the target runtime, in one session, on the same images.**
4. **Report negative results** with the evidence.
5. **No claim without an interval** (paired bootstrap, or an exact binomial test for guarantees).
6. **Never commit model weights or datasets** (Ultralytics weights are AGPL-3.0; this repo is MIT).
7. A PR that changes an algorithm needs a test that would fail without the change.
8. The core stays dependency-free (pure Python); benchmark scripts may use numpy / Ultralytics.

## Pull requests

Fork, branch, `python3 -m pytest -q test_apc.py test_control.py test_channel.py test_risk.py test_cascade.py`, then open a PR
using the template. Keep commits focused, with messages that say *why*.

## Questions

Open an issue. Maintainer: [@NICE-DEV226](https://github.com/NICE-DEV226).
