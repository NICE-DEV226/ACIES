#!/usr/bin/env python3
"""
Measure a real YOLO model on every image of a COCO subset at every resolution.

    python3 benchmarks/measure.py --weights yolov8n.pt --subset ~/.cache/acies-bench/coco/subset_1000_0.json

For each (image, imgsz) it stores the raw detections (conf >= 0.001, NMS iou 0.7 — the
Ultralytics validation protocol) and the measured latency (preprocess + inference +
postprocess, batch 1). Policies are then evaluated offline on this cache: every number
in a policy comparison is a real detector output and a real measured cost, none is
simulated. Resumable: re-running continues where it stopped.
"""
import argparse
import json
import os
import pickle
import time

import cv2
import numpy as np

RESOLUTIONS = [160, 224, 320, 416, 512, 640, 800, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    if a.threads:
        torch.set_num_threads(a.threads)

    subset = json.load(open(a.subset))
    images = subset["images"][: a.limit]
    tag = os.path.splitext(os.path.basename(a.weights))[0]
    out = a.out or os.path.join(os.path.dirname(a.subset), f"measure_{tag}_{len(images)}.pkl")

    state = {"weights": a.weights, "resolutions": RESOLUTIONS, "threads": torch.get_num_threads(),
             "dets": {}, "ms": {}}
    if os.path.exists(out):
        state = pickle.load(open(out, "rb"))
        print(f"resuming: {len(state['ms'])}/{len(images)} images done")

    model = YOLO(a.weights)
    dummy = np.zeros((480, 640, 3), np.uint8)
    for r in RESOLUTIONS:                       # warm-up (allocations, oneDNN kernels)
        for _ in range(3):
            model.predict(dummy, imgsz=r, verbose=False, device="cpu")

    t0 = time.time()
    for k, im in enumerate(images):
        if im["id"] in state["ms"]:
            continue
        img = cv2.imread(im["file"])
        ms = []
        for r in RESOLUTIONS:
            res = model.predict(img, imgsz=r, conf=0.001, iou=0.7, max_det=300,
                                verbose=False, device="cpu")[0]
            b = res.boxes
            state["dets"][(im["id"], r)] = (
                b.xyxy.numpy().astype(np.float32), b.conf.numpy().astype(np.float32),
                b.cls.numpy().astype(np.int16))
            ms.append(float(sum(res.speed.values())))
        state["ms"][im["id"]] = ms
        if (k + 1) % 50 == 0:
            pickle.dump(state, open(out, "wb"))
            done = len(state["ms"])
            print(f"{done}/{len(images)}  {time.time() - t0:.0f}s  "
                  f"mean ms/res={[round(float(np.mean([v[i] for v in state['ms'].values()])), 1) for i in range(len(RESOLUTIONS))]}",
                  flush=True)
    pickle.dump(state, open(out, "wb"))
    print("saved", out)


if __name__ == "__main__":
    main()
