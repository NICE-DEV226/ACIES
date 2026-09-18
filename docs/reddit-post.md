# Post draft

**Title:** ACIES — a controller that decides how much perception to spend per image (YOLOv8n/COCO: −40 % compute for −0.2 pt on presence decisions)

I built a controller that sits on top of a vision model and decides, per image, which resolution to
run next or whether to stop. Each action is modelled as a channel `P(outcome | class, action)` learned
from labelled examples; a dynamic-programming plan runs the next action only if its expected error
reduction (priced by one `error_cost` knob) exceeds its compute cost.

Measured on real YOLOv8n detections (COCO val2017, 8 classes, 1500 held-out images each, everything
tuned on a separate split, CPU latency): −40 % compute vs fixed 640 px for −0.2 accuracy points on
average. It loses significantly on `person` (−1.3 pt). A plain two-stage confidence cascade reaches a
similar frontier, so the value is generality rather than a win over a tuned cascade. It does not
improve full-detection mAP. My first version was worse than fixed-resolution YOLO on every class;
the write-up explains why (repeated deterministic observations counted as independent, hard 0/1
votes, an override that always forced the most expensive action).

Code, protocol and per-class tables with confidence intervals: https://github.com/NICE-DEV226/ACIES
