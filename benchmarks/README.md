# Real benchmarks

Everything here runs on **real detector output and real measured latency** — no simulated
observations. Reproduce (CPU, ~40 min for the measurement, minutes for each evaluation):

```bash
python3 benchmarks/coco_subset.py --n 3000 --seed 0        # COCO val2017 subset -> ~/.cache/acies-bench
python3 benchmarks/measure.py --weights yolov8n.pt --subset ~/.cache/acies-bench/coco/subset_3000_0.json
python3 benchmarks/final_eval.py   --measure <measure_*.pkl> --subset <subset_*.json> --clean-ms-from <idle-machine measure_*.pkl>
python3 benchmarks/track_p_v2.py   ...   # full policy curves per class
python3 benchmarks/run_benchmark.py ...  # mAP track + original ACIES
python3 benchmarks/selector_lab.py ...   # which features/models predict the resolution a frame needs
```

Protocol: images split by index parity (even = calibration, odd = test). Every knob is chosen
on calibration only; the test half is measured once. Cost = latency of YOLOv8n on this CPU
(4 threads), taken per resolution from an idle-machine run. Weights are Ultralytics (AGPL-3.0)
and are not redistributed here.

## Result 1 — decisions ("is there a <class> in the image?")

Operating point of each method chosen as "cheapest setting within 0.3 pt of Fixed 640 on
calibration". Test half, 1500 images. `Δ` = accuracy difference vs Fixed 640, paired-bootstrap 95 % CI.

| class | Fixed 640 | ACIES v2 (channel controller) | Δ vs Fixed 640 | 2-stage cascade |
|---|---|---|---|---|
| person | 94.7 % · 84 ms | 93.4 % · 51 ms (−40 %) | −1.33 [−2.33, −0.40] | 93.9 % · 45 ms |
| car | 96.7 % · 84 ms | 96.5 % · 63 ms (−26 %) | −0.20 [−0.93, +0.47] | 95.6 % · 44 ms |
| chair | 94.3 % · 84 ms | 94.4 % · 46 ms (−45 %) | +0.14 [−0.73, +1.07] | 93.7 % · 36 ms |
| bottle | 95.6 % · 84 ms | 95.9 % · 86 ms (−2 %) | +0.27 [−0.47, +1.00] | 95.3 % · 46 ms |
| cup | 95.1 % · 84 ms | 94.2 % · 32 ms (−62 %) | −0.95 [−1.93, 0.00] | 94.8 % · 36 ms |
| dog | 97.6 % · 84 ms | 98.0 % · 33 ms (−60 %) | +0.41 [−0.27, +1.07] | 97.2 % · 31 ms |
| tv | 98.4 % · 84 ms | 98.5 % · 53 ms (−37 %) | +0.15 [−0.40, +0.67] | 98.3 % · 28 ms |
| bench | 96.7 % · 84 ms | 96.2 % · 41 ms (−51 %) | −0.53 [−1.27, +0.20] | 96.1 % · 24 ms |
| **mean** | **96.1 % · 84 ms** | **95.9 % · 51 ms (−40 %)** | | 95.6 % · 36 ms |

Reading it honestly:
- The original controller (v1) was **dominated by fixed-resolution YOLO on every class**
  (e.g. person 89.2 % at 69 ms, car 92.1 % at 67 ms). v2 is ≈ +4 points and ≈ −28 % cost against it.
- v2 saves ~40 % of the compute of Fixed 640 for −0.2 point on average, but loses significantly on
  `person` and is borderline on `cup`; elsewhere the differences are within noise.
- **A plain two-stage confidence cascade is as good or better** on the frontier (−57 % cost, −0.5 pt).
  v2's advantage is generality (one knob, no hand-set thresholds, soft evidence, a plan) not a win
  over a well-tuned cascade.

## Result 2 — detection quality (mAP50-95): no win

A selector that spends one cheap pass then predicts the resolution a frame needs (ridge or boosted
trees on label-free detection statistics and image sharpness) does **not** beat fixed-resolution
YOLOv8n: reaching Fixed 640's mAP costs 101–124 ms vs 84 ms. YOLOv8n's mAP saturates at 512 px,
so the first pass (22–40 ms) eats the saving. An oracle that knows the ground truth could save
~60 % — the gap is a prediction problem, not a lack of headroom.

## Result 3 — certified operation (the guarantee)

`certified_eval.py` / `certified_cascade_demo.py`: every method is calibrated on a *certification* split with
Learn-then-Test (`acies.risk`) against the decision of the 640 px system (no labels). 8 classes x 4 random splits
of the 3000 images (train / cert / test thirds), delta = 10 %. A violation counts only when the test risk is
*significantly* above the tolerance (exact binomial test, 5 %). Costs: full-frame and crop passes timed in
one session (640 px = 96 ms), because costs measured in different sessions differed by up to 40 %.

Certified cascade (first stage among 160/224/320/416 px, reference 640 px):

| Guarantee | Certified splits | Saving | Test miss / false alarm | Violations |
|---|:--:|:--:|:--:|:--:|
| disagreement <= 2 % | 97 % | 60 % | 16.9 % / 0.25 % | 0 / 32 |
| disagreement <= 3 % | 100 % | 73 % | 30.6 % / 0.28 % | 0 / 32 |
| misses <= 10 %, false alarms <= 3 % | 66 % | 31 % | 2.9 % / 0.34 % | 0 / 32 |
| misses <= 5 %, false alarms <= 2 % | 34 % | 12 % | 0.7 % / 0.10 % | 0 / 32 |

The disagreement guarantee is true but weak for alerts (miss rate up to 31 % on rare classes). Same conditional
guarantees, other methods (saving at 10 %/3 % and 5 %/2 %): cascade 31 % / 12 %; channel planner 11 % / 1 %;
planner with asymmetric loss 8 % / 4 % (certified in 56 % / 22 % of splits); planner with a learned context belief
10 % / 2 %; Dynamic-Resolution-style predictor (no retraining) 12 % / 0 %. Zero significant violations for every
method. Certifying a 5 % miss rate needs >= 45 calibration positives with no miss (`acies.risk`).

## Result 4 — what did not help

- **Headroom is large but is a prediction problem.** An oracle that knows each resolution's answer and picks the
  cheapest that reproduces the reference would save 85 %; the best certified controllers reach 31-73 %.
- **Scene context** (boosted trees on all-class detection statistics and image sharpness) predicts safe negatives of
  rare classes at 99 % precision on 30-60 % of images in isolation, but as the initial belief of the planner it added
  nothing end to end.
- **Region-of-interest zoom** (crop around the most plausible box of a 224 px pass, 26-32 ms vs 96 ms): information
  per millisecond comparable to a 320 px full pass (0.20 vs 0.21 accuracy points per 100 ms on `person`); a
  hand-designed zoom cascade could not be certified in a preliminary two-class run.
- **mAP50-95:** on 3000 images, matching Fixed 640's mAP (34.87) costs 114 ms with a cascade and 99 ms with the best
  selector vs 84 ms for Fixed 640; only an oracle does better (36.4 at 77 ms).
- **Outcome quantisation must contain the reference's decision boundary** (a bin edge at 0.25); otherwise no
  channel can reproduce the reference and the planner cannot be certified at 1 %.

## Result 5 — current model and runtime (YOLO26n, ONNX Runtime), high-resolution drone images — PARTIAL

The COCO results above use **YOLOv8n in PyTorch** (2023 generation). YOLO26 (released 14 Jan 2026) is the current
Ultralytics model; YOLO27 is announced but not released (no weights). Two facts matter for this project:

- **Runtime changes the cost model.** In PyTorch eager mode the cost of YOLOv8n/YOLO26n varies little with resolution
  (fixed overheads dominate; 1024 px costs 5-8x a 160 px pass). With ONNX Runtime the cost follows the pixels
  (YOLO26n: 6.6 ms at 160 px, 61 ms at 640 px, 115 ms at 1024 px; 640 vs 320 px = 4.75x for 4x the pixels).
  Savings measured under PyTorch costs therefore do not transfer to an optimised runtime.
- YOLO26n is 5.5 GFLOPs vs 8.7 for YOLOv8n and has no NMS (output `(1, 300, 6)`); the "up to 43 % faster on CPU"
  claim is for ONNX export and was **not** observed in PyTorch on this machine (i7-8565U).

VisDrone2019-DET (val + test-dev, 1080x1920 drone images, many small objects), YOLO26n exported to ONNX, resolutions
320-1920 px, reference = 1920 px (median cost 569 ms; 640 px = 47 ms; 320 px = 15 ms). **Only 1200 of 2158 images were
measured** (the run was stopped because the host machine ran out of memory); the run is resumable
(`benchmarks/measure.py` continues from its pickle). Costs are medians of a run that was perturbed by other activity
(+/- 15 %). Classes: bicycle, bus, truck, motorcycle (balanced presence 34-59 %), plus person and car (present in 83 % and
95 % of images, so uninformative).

| Guarantee (delta = 10 %) | Certified splits | Saving | Violations |
|---|:--:|:--:|:--:|
| disagreement <= 2 % | 96 % | 31 % | 0 / 24 |
| disagreement <= 3 % | 100 % | 38 % | 0 / 24 |
| misses <= 10 %, false alarms <= 3 % | 62 % | 15 % | 0 / 24 |
| misses <= 5 %, false alarms <= 2 % | 42 % | 7 % | 0 / 24 |

Reading: (i) **the hypothesis that high-resolution images give larger savings was not supported** — the savings are lower
than on COCO although cost follows the pixels, because small aerial targets are invisible at low resolution and the
cheap pass rarely discriminates; (ii) no cheaper *fixed* resolution could be certified at 2-3 % disagreement, so the
adaptive cascade is the only way to save anything there; (iii) with 4 classes x 2 splits the cascade saved 25 % / 31 %
and the channel planner 6 % / 22 %; (iv) the COCO-trained detector is only ~60 % accurate against VisDrone ground truth
for these classes, and the guarantee is *relative to the reference*, so a fine-tuned aerial model would be needed for a
meaningful deployment.

Commands: pass `--ref-res 640` to `certified_eval.py` / `certified_cascade_demo.py` for the COCO runs (the default reference
is the largest resolution).

## Limits
One detector (YOLOv8n), one machine (CPU), 1500 test images per class (accuracy s.e. ≈ 0.6 pt),
binary presence decisions only, COCO val2017. YOLO26 and GPU/edge cost ratios are not measured.
