# Launch notes

Draft text kept deliberately factual. All numbers come from `benchmarks/README.md`.

**One-paragraph pitch.** ACIES is a controller that decides how much perception to spend per input.
It sits on top of a vision model and chooses which resolution to run next, or to stop and decide.
On YOLOv8n / COCO val2017, for the decision "is there a `<class>` in the image?", it uses ~40 % less
compute than running at a fixed 640 px for ~0.2 point less accuracy on average (8 classes, 1500
held-out images each, settings chosen on a separate calibration split).

**What to say plainly.**
- It helps decisions (presence, alerts), not full detection quality (mAP).
- A well-tuned two-stage confidence cascade reaches a similar cost/accuracy frontier; ACIES's
  contribution is a principled version with one knob and no hand-set thresholds.
- The first version was worse than fixed-resolution YOLO; the fixes and the evidence are in the repo.
- One detector, one CPU, binary decisions.

**Good first issues.** Multi-class decisions on the channel model; a GPU/edge cost model; YOLO26 and
other detectors; replace BOCPD (a constant hazard makes `P(r=0)` constant, so the alarm never fires);
fix the C++ library's bounds checking.
