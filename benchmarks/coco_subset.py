#!/usr/bin/env python3
"""
Fetch a reproducible COCO val2017 subset (images + ground truth) into a cache dir.

    python3 benchmarks/coco_subset.py --n 1000 --seed 0

val2017 is disjoint from train2017, on which the Ultralytics COCO weights are trained,
so there is no train/test contamination. Crowd annotations are dropped, as Ultralytics
does when it converts COCO for validation.

Output (in $ACIES_BENCH_CACHE, default ~/.cache/acies-bench/coco):
    images/<id>.jpg
    subset_<n>_<seed>.json   {"images": [{"id","file","w","h","boxes":[[x1,y1,x2,y2]],"cls":[k]}],
                              "names": [80 class names]}
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

CACHE = os.environ.get("ACIES_BENCH_CACHE", os.path.expanduser("~/.cache/acies-bench/coco"))
ANN = os.path.join(CACHE, "annotations", "instances_val2017.json")
URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"   # plain http: the https cert is broken


def fetch(image_id: int) -> str:
    path = os.path.join(CACHE, "images", f"{image_id}.jpg")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urlopen(URL.format(image_id), timeout=60) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
    return path


def build(n: int, seed: int) -> str:
    out = os.path.join(CACHE, f"subset_{n}_{seed}.json")
    if os.path.exists(out):
        return out
    coco = json.load(open(ANN))
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_idx = {c["id"]: i for i, c in enumerate(cats)}
    imgs = {im["id"]: im for im in coco["images"]}
    per_image = {i: ([], []) for i in imgs}
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        per_image[a["image_id"]][0].append([x, y, x + w, y + h])
        per_image[a["image_id"]][1].append(cat_idx[a["category_id"]])
    ids = sorted(imgs)
    random.Random(seed).shuffle(ids)
    ids = sorted(ids[:n])
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(fetch, ids))
    subset = {
        "names": [c["name"] for c in cats],
        "images": [{
            "id": i, "file": os.path.join(CACHE, "images", f"{i}.jpg"),
            "w": imgs[i]["width"], "h": imgs[i]["height"],
            "boxes": per_image[i][0], "cls": per_image[i][1],
        } for i in ids],
    }
    json.dump(subset, open(out, "w"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = build(a.n, a.seed)
    s = json.load(open(p))
    print(p, len(s["images"]), "images,", sum(len(i["cls"]) for i in s["images"]), "objects")
