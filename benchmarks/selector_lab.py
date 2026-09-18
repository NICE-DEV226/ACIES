#!/usr/bin/env python3
"""
Selector lab: which cheap signals + which model best predict "how much resolution does
this image need?" — evaluated on the real cached detections (train on the calibration
half, report on the test half; nothing here sees test labels before scoring).

    python3 benchmarks/selector_lab.py --measure <pkl> --subset <json>

Metric per configuration: the (ms, mAP) frontier over the cost-penalty mu, summarised by
  - ms to reach the mAP of Fixed 640      (lower is better; Fixed 640 itself costs ~87 ms)
  - mAP at a fixed ~cost budget            (higher is better)
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit import image_stats  # noqa: E402
from run_benchmark import Data, pareto  # noqa: E402


# ----------------------------------------------------------------------------- features

def det_features(det, wh):
    """Label-free features of one detection set (conf, geometry, uncertainty mass)."""
    bx, cf, cl = det
    W, H = wh
    img_area = float(W * H)
    hi = cf >= 0.25
    low = (cf >= 0.05) & (cf < 0.25)
    rel = np.sqrt(np.clip((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]), 1, None) / img_area)
    top = np.sort(cf)[::-1]
    n = int(hi.sum())
    f = [
        np.log1p(n), np.log1p(int((cf >= 0.5).sum())), np.log1p(int((cf >= 0.75).sum())),
        np.log1p(int(low.sum())), float(cf[low].sum()) / 5.0,
        float(top[0]) if len(top) else 0.0,
        float(top[:3].mean()) if len(top) else 0.0,
        float(cf[hi].mean()) if n else 0.0,
        float((rel[hi] < 0.08).sum()) / 5.0 if n else 0.0,            # small (rel. side < 8 %)
        float(((rel[hi] >= 0.08) & (rel[hi] < 0.25)).sum()) / 5.0 if n else 0.0,
        float((rel[hi] >= 0.25).sum()) / 5.0 if n else 0.0,
        float(rel[hi].min()) if n else 0.0,
        float(rel[hi].mean()) if n else 0.0,
        len(set(cl[hi].tolist())) / 5.0,
        float(((cf >= 0.15) & (cf < 0.5)).sum()) / 5.0,               # borderline detections
    ]
    return np.array(f, np.float64)


def image_features(path):
    """Cheap pixel statistics from a 96 px thumbnail (sharpness/contrast: what a low pass misses)."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = g.shape
    t = cv2.resize(g, (96, max(1, int(96 * h / w))), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(t, cv2.CV_32F)
    edges = cv2.Canny(t, 50, 150)
    return np.array([t.mean() / 255.0, t.std() / 128.0, float(lap.var()) / 1000.0,
                     float(edges.mean()) / 255.0, np.log(w * h) / 14.0], np.float64), (w, h)


def poly2(v):
    return np.concatenate([v, np.outer(v, v)[np.triu_indices(len(v))]])


# ----------------------------------------------------------------------------- models

def fit_ridge(Z, Y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ Y)


class Stump3:
    """Depth-3 regression tree on quantile bins (numpy), for gradient boosting."""

    def fit(self, X, r, depth=3, min_leaf=12, nbins=12):
        self.node = self._grow(X, r, depth, min_leaf, nbins)
        return self

    def _grow(self, X, r, depth, min_leaf, nbins):
        if depth == 0 or len(r) < 2 * min_leaf:
            return r.mean()
        best = None
        tot, n = r.sum(), len(r)
        for j in range(X.shape[1]):
            qs = np.unique(np.quantile(X[:, j], np.linspace(0.08, 0.92, nbins)))
            for q in qs:
                m = X[:, j] <= q
                nl = m.sum()
                if nl < min_leaf or n - nl < min_leaf:
                    continue
                sl = r[m].sum()
                gain = sl * sl / nl + (tot - sl) ** 2 / (n - nl)
                if best is None or gain > best[0]:
                    best = (gain, j, q)
        if best is None:
            return r.mean()
        _, j, q = best
        m = X[:, j] <= q
        return (j, q, self._grow(X[m], r[m], depth - 1, min_leaf, nbins),
                self._grow(X[~m], r[~m], depth - 1, min_leaf, nbins))

    def predict(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            nd = self.node
            while isinstance(nd, tuple):
                nd = nd[2] if x[nd[0]] <= nd[1] else nd[3]
            out[i] = nd
        return out


def fit_gbm(X, y, rounds=60, lr=0.12):
    base = y.mean()
    F = np.full(len(y), base)
    trees = []
    for _ in range(rounds):
        t = Stump3().fit(X, y - F)
        F = F + lr * t.predict(X)
        trees.append(t)
    return base, lr, trees


def predict_gbm(m, X):
    base, lr, trees = m
    F = np.full(len(X), base)
    for t in trees:
        F = F + lr * t.predict(X)
    return F


# ----------------------------------------------------------------------------- evaluation

MUS = (0.0, 0.0003, 0.0006, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.02)


def frontier_for(D, pred, k0, extra_ms=0.5):
    T, R = D.test, D.R
    pts = []
    for mu in MUS:
        choice, cost = [], []
        for i in T:
            extra = np.array([0.0 if k == k0 else D.ms[i, k] for k in range(len(R))])
            k = int(np.argmax(pred[i] - mu * extra))
            choice.append(k)
            cost.append(D.ms[i, k0] + extra[k] + extra_ms)
        m, m50 = D.map(T, choice)
        pts.append({"mu": mu, "map": m, "ms": float(np.mean(cost))})
    return pareto(pts)


def ms_to_reach(fr, target):
    pts = sorted(fr, key=lambda p: p["map"])
    for a, b in zip(pts, pts[1:]):
        if a["map"] <= target <= b["map"]:
            t = (target - a["map"]) / max(b["map"] - a["map"], 1e-12)
            return a["ms"] + t * (b["ms"] - a["ms"])
    return float("nan") if target > pts[-1]["map"] else pts[0]["ms"]


def map_at_ms(fr, budget):
    pts = sorted(fr, key=lambda p: p["ms"])
    best = float("nan")
    for a, b in zip(pts, pts[1:]):
        if a["ms"] <= budget <= b["ms"]:
            t = (budget - a["ms"]) / max(b["ms"] - a["ms"], 1e-12)
            return a["map"] + t * (b["map"] - a["map"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--r0", type=int, nargs="+", default=[160, 224, 320])
    ap.add_argument("--gbm", action="store_true")
    ap.add_argument("--clean-ms-from", default=None)
    a = ap.parse_args()

    import json
    D = Data(a.measure, a.subset, a.clean_ms_from)
    sub = {im["id"]: im for im in json.load(open(a.subset))["images"]}
    order = [im for im in json.load(open(a.subset))["images"] if any(True for _ in [0])]
    ids = [im["id"] for im in order][: D.n]
    img_feats, whs = zip(*[image_features(sub[i]["file"]) for i in ids])
    img_feats = np.array(img_feats)
    T, R = D.test, D.R
    fx640 = next(p for p in [{"map": D.map(T, [R.index(640)] * len(T))[0]}])["map"]
    fx_ms = float(D.ms[T, R.index(640)].mean())
    print(f"n={D.n}  test={len(T)}  Fixed 640: mAP {fx640*100:.2f} @ {fx_ms:.1f} ms\n")
    print(f"{'config':44s} {'ms to reach F640 mAP':>21s} {'mAP @ 60 ms':>12s} {'mAP @ 100 ms':>13s}")

    for r0 in a.r0:
        k0 = R.index(r0)
        base = np.array([det_features(D.dets[i][k0], whs[i]) for i in range(D.n)])
        sets = {
            "A  det-stats, ridge deg2": (base, "ridge"),
            "B  + image stats, ridge deg2": (np.hstack([base, img_feats]), "ridge"),
        }
        if a.gbm:
            sets["C  det+image, GBM"] = (np.hstack([base, img_feats]), "gbm")
        for name, (X, kind) in sets.items():
            if kind == "ridge":
                P = np.array([poly2(x) for x in X])
                mu_, sd_ = P[D.cal].mean(0), P[D.cal].std(0) + 1e-6
                Z = np.hstack([(P - mu_) / sd_, np.ones((D.n, 1))])
                best = None
                for lam in (3.0, 10.0, 30.0, 100.0, 300.0):     # choose lam on calibration only (2-fold)
                    h = len(D.cal) // 2
                    tr, va = D.cal[:h], D.cal[h:]
                    err = ((Z[va] @ fit_ridge(Z[tr], D.f1[tr], lam) - D.f1[va]) ** 2).mean()
                    if best is None or err < best[0]:
                        best = (err, lam)
                pred = Z @ fit_ridge(Z[D.cal], D.f1[D.cal], best[1])
            else:
                pred = np.zeros((D.n, len(R)))
                for k in range(len(R)):
                    m = fit_gbm(X[D.cal], D.f1[D.cal, k])
                    pred[:, k] = predict_gbm(m, X)
            fr = frontier_for(D, pred, k0)
            e = ms_to_reach(fr, fx640)
            print(f"r0={r0:<4d} {name:36s} {e:14.1f} ms ({(1 - e / fx_ms) * 100:+4.0f}%) "
                  f"{map_at_ms(fr, 60) * 100:11.2f} {map_at_ms(fr, 100) * 100:13.2f}")


if __name__ == "__main__":
    main()
