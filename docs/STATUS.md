# State of the project — and where we want your ideas

*Written September 2026. Read this before you contribute: it says what is proven, what failed, what nobody has measured yet, and where help matters most.*

## TL;DR

ACIES started as an "adaptive perception controller" that would beat fixed-resolution inference. **That claim was wrong** and has been withdrawn (the original "76 % savings" came from a synthetic simulator, not a real model). After rebuilding the evaluation on real detector output, what survives is narrower and more useful:

> **ACIES can accelerate a black-box perception model with a finite-sample, label-free guarantee** against the full-cost system — and it says plainly when it cannot prove any saving.

It is a small, tested library plus an honest benchmark. It is **not** a breakthrough and **not** validated outside one detector family, one dataset pair and one CPU. That is exactly why we need you.

## The three layers

| Layer | Modules | Status |
|---|---|---|
| **Certified cascade** — the part that works | `acies.risk`, `acies.cascade` | Tested, validated on real detections. **Start here.** |
| Planner over learned channels | `acies.channel`, `acies.optimal`, `acies.selector` | Works, but **does not beat a tuned cascade**. Research code. |
| Classic controller (binary "clarity" model) | `acies.controller`, `belief`, `clarity_learner`, `safety`, … | Kept for simulation and comparison. Was worse than fixed resolution on real data. |

## What is established (with where to check it)

| Claim | Evidence |
|---|---|
| Learn-then-Test on a cascade's thresholds holds its guarantee | 0 significant violations in 24–32 random splits per setting; a naive rule violates the bound in 39 % of simulated draws vs 3 % for the certified one — `test_risk.py`, `benchmarks/results/` |
| YOLOv8n / COCO (8 classes): 60 % compute saved at a guaranteed disagreement ≤ 2 %, 73 % at ≤ 3 %; 31 % / 12 % with miss ≤ 10 % / 5 % and false alarms ≤ 3 % / 2 % | `benchmarks/README.md` (Result 3), `certified_cascade_demo.py` |
| No cheaper *fixed* resolution can be certified where the cascade can (VisDrone, 2–3 % disagreement) | `benchmarks/results/visdrone_yolo26n_certified_eval.txt` |
| The original controller was worse than fixed-resolution YOLO on every class; the causes (emergency override on the prior, i.i.d. treatment of deterministic observations, hard 0/1 votes) are diagnosed and fixed | `CHANGELOG.md`, `paper/acies.tex` §5 |
| Certifying a miss rate ≤ 5 % needs ≥ 45 calibration positives with no miss (77 with one) | `acies.risk`, `benchmarks/README.md` |

## What did **not** work — please don't repeat it without a new angle

- **The channel planner vs a tuned cascade**: 8–11 % certified saving vs 31 %. Even with asymmetric costs.
- **Region-of-interest zoom** (crop around the most plausible box of a cheap pass): information per millisecond comparable to a 320 px full pass, no gain.
- **Scene context as the planner's initial belief** (boosted trees on all-class detection statistics + image sharpness): nothing end to end, although it predicts safe negatives of rare classes well in isolation.
- **Any per-image resolution selector for mAP50-95**: reaching Fixed-640's mAP costs more than Fixed-640 itself. Only an oracle that knows the ground truth wins.
- **"Large images save more"** (hypothesis): not supported — on drone images the savings were *lower* than on COCO because small targets are invisible at low resolution.

## What is unknown — nobody has measured this yet

1. **Other runtimes and hardware.** Costs so far come from one laptop CPU (i7-8565U). In PyTorch eager mode the cost barely grows with resolution (fixed overheads); in ONNX Runtime it follows the pixels. GPU, TensorRT, OpenVINO, Jetson, Raspberry Pi: **unmeasured**. Savings will not transfer.
2. **The current model generation.** COCO results are for YOLOv8n (2023) under PyTorch. YOLO26 (Jan 2026, NMS-free) was measured only on **1200 of 2158 VisDrone images**. YOLO27 is announced but unreleased. RT-DETR / RF-DETR: not tried.
3. **Tasks beyond a binary "is class X present?"**: counting thresholds, multi-class, segmentation, per-object recall.
4. **Distribution shift.** The guarantee assumes exchangeable inputs. Nothing in ACIES detects drift.
5. **Video.** Temporal reuse is where the strongest systems live (NoScope reports two to three orders of magnitude of speed-up). ACIES has none of it.
6. **Real deployments and sectors** (industrial inspection, pathology slides, satellite/onboard, edge cameras). All plausible, none validated. We also have **no evidence of demand**.

## Known defects (good places to start)

| Where | Problem | Difficulty |
|---|---|---|
| `acies/change_point.py` ([#15](https://github.com/NICE-DEV226/ACIES/issues/15)) | BOCPD never fires: with a constant hazard, `P(r_t = 0)` equals the hazard rate identically | medium — replace with run-length MAP / CUSUM |
| `acies/multiclass.py` ([#16](https://github.com/NICE-DEV226/ACIES/issues/16)) | Pseudo-count belief saturates at the per-observation clarity; not a Bayesian posterior | medium |
| `cpp/` ([#11](https://github.com/NICE-DEV226/ACIES/issues/11)) | No bounds check on action ids (heap overflow), exception can cross the C ABI on invalid sizes, `ClarityLearner` owns a raw pointer with no copy constructor (double free); the wrapper is also *slower* than pure Python and unused | good first issue |
| `main.go` ([#12](https://github.com/NICE-DEV226/ACIES/issues/12)) | Boolean flags given alone do nothing (`--verbose`) and swallow the next flag (`--verbose --hardware jetson`) | good first issue |
| `core.go` | The learner is recreated on every run, so the Go CLI is not the same algorithm as the Python controller | easy |
| `test/stress_test_robust.py` | Needs an external model file (`/tmp/mnist_cnn.pth`) that is not in the repo | easy |

## Ideas wanted

These are open questions, roughly ordered by how much they would change our conclusions. **Take one, or propose your own** (see the bottom).

1. **Reproduce on your hardware/runtime** (GPU, TensorRT, OpenVINO, Jetson, RPi) — [#8](https://github.com/NICE-DEV226/ACIES/issues/8). *Success: a cost-vs-resolution curve and the certified-saving table from `certified_cascade_demo.py`, including the cases where it saves nothing.*
2. **Close the headroom gap** — [#9](https://github.com/NICE-DEV226/ACIES/issues/9). An oracle that knows every resolution's answer would save ~85 %; certified cascades reach 31–73 %. The missing piece is predicting *"will the cheap pass agree with the reference?"*. *Success: higher certified saving at the same ε, under the same protocol.*
3. **More data-efficient certification** — [#10](https://github.com/NICE-DEV226/ACIES/issues/10). The 45-positives wall makes rare events uncertifiable. Tighter bounds, sharing strength across classes, sequential/anytime-valid tests (e-values). *Success: certify a rare class with fewer calibration examples, with an empirical coverage check.*
4. **Drift-aware guarantees** — [#14](https://github.com/NICE-DEV226/ACIES/issues/14). Detect when exchangeability breaks and fall back to the reference; conformal methods under covariate shift. Related: runtime monitoring and audit log, [#17](https://github.com/NICE-DEV226/ACIES/issues/17).
5. **Non-binary decisions**: counting, multi-class, detection-level guarantees — [#13](https://github.com/NICE-DEV226/ACIES/issues/13).
6. **Video**: temporal reuse combined with a certified guarantee.
7. **Other detectors**: complete YOLO26 on COCO, RT-DETR/RF-DETR, YOLO27 when released.
8. **Sector validation with real data and a frozen model.** If you work in one, tell us what the decision, the cost and the tolerable error actually are.
9. **Packaging**: an Ultralytics/ONNX Runtime wrapper (`certified_predict`), `pip install acies[bench]`, a CLI that calibrates from a folder of images.
10. **Repair the defects** in the table above.

## Ground rules (they exist because we were wrong once)

- **Compare against the strong baseline**: fixed resolution *and* a tuned two-stage cascade. Beating only a weak baseline proves nothing.
- **Keep calibration, certification and test disjoint.** Fit on one, certify on another, report on a third.
- **Measure costs on the target runtime, in one session**, on the same images. Costs measured in different sessions differed by up to 40 % here.
- **Report negative results**, with the evidence. They are contributions.
- **No claim without an interval** (paired bootstrap, or an exact binomial test for guarantees).
- **Never commit model weights or datasets** (licenses: Ultralytics weights are AGPL-3.0 and this repo is MIT).
- Every PR that touches an algorithm needs a test that would fail without the change.

## Run it in five minutes

```bash
git clone https://github.com/NICE-DEV226/ACIES.git && cd ACIES
pip install -e ".[dev]"
python3 -m pytest -q test_apc.py test_control.py test_channel.py test_risk.py test_cascade.py   # 86 tests
python3 examples/optimal_gap.py                # classic controller vs the exact optimum (synthetic)
```

Real benchmarks need Ultralytics and a COCO / VisDrone subset; commands are in [`benchmarks/README.md`](../benchmarks/README.md).

## Propose an idea

Open an issue with the **"Research idea"** template. A good proposal says: the question, why you think it could help, the baseline it must beat, the data and runtime, and what result would make you drop it. A well-run experiment that fails is as welcome as one that works.

Maintainer: [@NICE-DEV226](https://github.com/NICE-DEV226).
