# Benchmarks

The authoritative results are in [`benchmarks/README.md`](../benchmarks/README.md): real YOLOv8n
detections on COCO val2017 with measured latency, a calibration/test split, and paired-bootstrap
confidence intervals. Summary (decision "is there a `<class>` in the image?", mean of 8 classes,
1500 test images each):

| Policy | Accuracy | Cost (ms/img, CPU) |
|--------|:--------:|:------------------:|
| Fixed 640 px | 96.1 % | 84 |
| **ACIES v2** (`acies.channel`) | 95.9 % | **51 (−40 %)** |
| Two-stage confidence cascade | 95.6 % | 36 (−57 %) |
| ACIES v1 (classic controller) | ~92 % | ~71 |

- v1 was dominated by fixed-resolution YOLO on every class; v2 recovers ≈ +4 accuracy points at ≈ −28 % cost over v1.
- v2 loses significantly on `person` (−1.3 pt) and is within noise on most other classes.
- A plain cascade is as good on the frontier; v2's advantage is generality (one knob, no hand-set thresholds).
- No gain on detection quality (mAP50-95): a per-image resolution selector does not beat fixed-resolution YOLOv8n.

## Removed results

Earlier versions of this page reported figures from a synthetic simulator (MNIST "clarity" tables
and a Bernoulli observation model, not a real model), a throughput comparison between different
algorithms (Python 476 img/s, Go 32,800 runs/s, C++ 1,500–2,000 img/s) and a "76 % savings" headline.
They did not measure the shipped controller or a real perception model and have been withdrawn.
Measured for reference: the classic controller runs ≈ 5,000 simulated tasks/s in Python and the Go
simulator ≈ 49,000/s (they run different algorithms); the C++ library through `ctypes` is 0.75–0.8× the
speed of pure Python.

## Reproduce

```bash
python3 benchmarks/coco_subset.py --n 3000 --seed 0
python3 benchmarks/measure.py --weights yolov8n.pt --subset ~/.cache/acies-bench/coco/subset_3000_0.json
python3 benchmarks/final_eval.py --measure <measure_*.pkl> --subset <subset_*.json>
python3 examples/optimal_gap.py      # classic controller vs the exact optimal frontier (synthetic model)
```
