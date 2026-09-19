#!/usr/bin/env python3
"""
Measure real region-of-interest ("zoom") actions.

    python3 benchmarks/measure_zoom.py --measure <pkl> --subset <json> [--shard 0 --nshards 2]

After a cheap full-frame pass (src resolution, default 224), the most plausible box of a class
is picked from that pass (highest confidence >= cand_min) and a square window around it
(side = max(expand * box side, min_side), clipped to the image) is cropped and run through YOLO
at input size S. Detections are mapped back to image coordinates and stored, together with the
measured latency of that crop pass. The window depends only on the cheap pass, so the action is
deployable: no ground truth is involved.

Output: per shard a pickle {(image_id, cls, S): (boxes_img, conf, cls_ids, ms, window)}.
"""
import argparse
import json
import os
import pickle
import time

import cv2
cv2.setNumThreads(1)
import numpy as np


def window(box, W, H, expand, min_side):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(expand * max(x2 - x1, y2 - y1), min_side)
    side = min(side, W, H)
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return int(round(x0)), int(round(y0)), int(round(side))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--src", type=int, default=224)
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 320])
    ap.add_argument("--classes", type=int, nargs="+", default=[0, 2, 56, 39, 41, 16, 62, 13])
    ap.add_argument("--cand-min", type=float, default=0.01)
    ap.add_argument("--expand", type=float, default=2.5)
    ap.add_argument("--min-side", type=int, default=96)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from ultralytics import YOLO
    torch.set_num_threads(a.threads)

    S = pickle.load(open(a.measure, "rb"))
    images = json.load(open(a.subset))["images"][: a.limit]
    images = [im for k, im in enumerate(images) if k % a.nshards == a.shard and im["id"] in S["ms"]]
    model = YOLO(a.weights)
    dummy = np.zeros((300, 300, 3), np.uint8)
    for s in a.sizes:
        for _ in range(3):
            model.predict(dummy, imgsz=s, verbose=False, device="cpu")

    out, t0 = {}, time.time()
    for n, im in enumerate(images):
        bx, cf, cl = S["dets"][(im["id"], a.src)]
        img = None
        for c in a.classes:
            m = (cl == c) & (cf >= a.cand_min)
            if not m.any():
                continue
            j = np.flatnonzero(m)[int(np.argmax(cf[m]))]
            if img is None:
                img = cv2.imread(im["file"])
            H, W = img.shape[:2]
            x0, y0, side = window(bx[j], W, H, a.expand, a.min_side)
            crop = img[y0:y0 + side, x0:x0 + side]
            for s in a.sizes:
                r = model.predict(crop, imgsz=s, conf=0.001, iou=0.7, max_det=100, verbose=False, device="cpu")[0]
                b = r.boxes
                xy = b.xyxy.numpy().astype(np.float32)
                xy[:, [0, 2]] += x0
                xy[:, [1, 3]] += y0
                out[(im["id"], c, s)] = (xy, b.conf.numpy().astype(np.float32), b.cls.numpy().astype(np.int16),
                                         float(sum(r.speed.values())), (x0, y0, side))
        if (n + 1) % 25 == 0:
            print(f"shard {a.shard}: {n + 1}/{len(images)}  {len(out)} crops  {time.time() - t0:.0f}s", flush=True)
    pickle.dump({"args": vars(a), "crops": out}, open(a.out, "wb"))
    print("saved", a.out, len(out), "crops")


if __name__ == "__main__":
    main()
