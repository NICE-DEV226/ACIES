#!/usr/bin/env python3
"""Per-resolution cost model from a measurement run (single session): mean latency of every resolution."""
import argparse
import json
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--measure", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stat", choices=["mean", "median"], default="mean", help="statistic used for the cost model")
a = ap.parse_args()
S = pickle.load(open(a.measure, "rb"))
ms = np.array(list(S["ms"].values()))
agg = np.mean if a.stat == "mean" else np.median
out = {"weights": S["weights"], "n_images": int(len(ms)), "stat": a.stat,
       "full": {str(r): float(agg(ms[:, i])) for i, r in enumerate(S["resolutions"])},
       "median": {str(r): float(np.median(ms[:, i])) for i, r in enumerate(S["resolutions"])}}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out["full"]))
