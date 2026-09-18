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

## Limits
One detector (YOLOv8n), one machine (CPU), 1500 test images per class (accuracy s.e. ≈ 0.6 pt),
binary presence decisions only, COCO val2017. YOLO26 and GPU/edge cost ratios are not measured.
