#!/usr/bin/env python3
"""Clean latency of a region-of-interest crop pass (batch 1, idle machine): median per crop of 3 timed runs."""
import argparse
import json
import pickle
import random

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--zoom", nargs="+", required=True)
ap.add_argument("--subset", required=True)
ap.add_argument("--weights", default="yolov8n.pt")
ap.add_argument("--n", type=int, default=150)
a = ap.parse_args()

import torch
from ultralytics import YOLO
torch.set_num_threads(4)                      # same as the full-frame latency measurement
crops = {}
for p in a.zoom:
    crops.update(pickle.load(open(p, "rb"))["crops"])
files = {im["id"]: im["file"] for im in json.load(open(a.subset))["images"]}
keys = sorted({(k[0], k[1]) for k in crops})
random.Random(0).shuffle(keys)
model = YOLO(a.weights)
sample = []
for iid, c in keys[: a.n]:
    x0, y0, side = crops[(iid, c, 224)][4]
    img = cv2.imread(files[iid])
    sample.append(np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side]))
out = {}
for S in (224, 320):
    for x in sample[:20]:
        model.predict(x, imgsz=S, verbose=False, device="cpu")
    per = []
    for x in sample:
        t = [sum(model.predict(x, imgsz=S, conf=0.001, iou=0.7, max_det=100, verbose=False, device="cpu")[0].speed.values())
             for _ in range(3)]
        per.append(float(np.median(t)))
    out[S] = {"mean_of_medians": float(np.mean(per)), "median": float(np.median(per)), "p90": float(np.percentile(per, 90))}
    print(S, out[S], flush=True)
# full-frame passes on the SAME images, same session, same threads: costs are only comparable this way
full = {}
imgs = [cv2.imread(files[iid]) for iid, _ in keys[: a.n]]
for r in (160, 224, 320, 416, 512, 640, 800, 1024):
    for x in imgs[:10]:
        model.predict(x, imgsz=r, verbose=False, device="cpu")
    per = [float(np.median([sum(model.predict(x, imgsz=r, conf=0.001, iou=0.7, max_det=300, verbose=False,
                                             device="cpu")[0].speed.values()) for _ in range(3)])) for x in imgs]
    full[r] = float(np.mean(per))
    print("full", r, round(full[r], 1), flush=True)
out["full"] = full
json.dump(out, open("benchmarks/results/latency_single_session.json", "w"), indent=1)
