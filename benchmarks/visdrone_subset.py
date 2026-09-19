#!/usr/bin/env python3
"""
VisDrone2019-DET (val + test-dev, 2158 HD drone images, many small objects) as a subset file in the same
format as coco_subset.py, with ground truth mapped to COCO class ids (the detector is COCO-trained):
pedestrian/people -> person(0), bicycle -> 1, car/van -> car(2), motor -> motorcycle(3), bus -> 5, truck -> 7.

    python3 benchmarks/visdrone_subset.py
Downloads: https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{val,test-dev}.zip
"""
import glob
import json
import os

import cv2

ROOT = os.path.expanduser("~/.cache/acies-bench/visdrone")
MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 7, 9: 5, 10: 3}      # VisDrone category -> COCO id (others dropped)
NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

images = []
for split in ("VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"):
    for f in sorted(glob.glob(f"{ROOT}/{split}/images/*.jpg")):
        h, w = cv2.imread(f).shape[:2]
        ann = f.replace("/images/", "/annotations/").replace(".jpg", ".txt")
        boxes, cls = [], []
        for line in open(ann):
            v = line.strip().split(",")
            if len(v) < 6 or int(v[5]) not in MAP or int(v[4]) == 0:
                continue
            x, y, bw, bh = (float(t) for t in v[:4])
            if bw > 1 and bh > 1:
                boxes.append([x, y, x + bw, y + bh])
                cls.append(MAP[int(v[5])])
        images.append({"id": len(images), "file": f, "w": w, "h": h, "boxes": boxes, "cls": cls})
out = f"{ROOT}/subset_visdrone.json"
json.dump({"names": [NAMES.get(i, str(i)) for i in range(80)], "images": images}, open(out, "w"))
print(out, len(images), "images,", sum(len(i["cls"]) for i in images), "objects")
for k, n in NAMES.items():
    print(f"  {n:11s} present in {sum(k in i['cls'] for i in images) / len(images):5.1%} of images")
