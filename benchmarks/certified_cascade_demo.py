#!/usr/bin/env python3
"""
Plug-and-play certified cascade on real YOLO output: acies.cascade.calibrate_cascade, 224 px -> 640 px.

    python3 benchmarks/certified_cascade_demo.py --measure <pkl> --subset <json> --latency-json benchmarks/results/latency_single_session.json

Same random TRAIN / CERT / TEST splits as certified_eval.py. Nothing is tuned by hand.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acies.cascade import CascadeSample, Ladder, calibrate_ladders  # noqa: E402
from acies.risk import binom_cdf  # noqa: E402


def significant(k, n, eps):
    return n > 0 and (1.0 - binom_cdf(k - 1, n, eps)) < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--latency-json", required=True)
    ap.add_argument("--classes", default="person:0,car:2,chair:56,bottle:39,cup:41,dog:16,tv:62,bench:13")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--first-stages", type=int, nargs="+", default=[160, 224, 320, 416])
    a = ap.parse_args()

    S = pickle.load(open(a.measure, "rb"))
    R = S["resolutions"]
    images = [im for im in json.load(open(a.subset))["images"] if im["id"] in S["ms"]]
    lat = json.load(open(a.latency_json))["full"]
    c1 = lat["640"]
    classes = {s.split(":")[0]: int(s.split(":")[1]) for s in a.classes.split(",")}
    print(f"candidate first stages {a.first_stages} px -> reference 640 px ({c1:.1f} ms); delta = 10%\n")
    settings = [("disagree", dict(eps=0.02), "disagreement <= 2%"),
                ("disagree", dict(eps=0.03), "disagreement <= 3%"),
                ("conditional", dict(eps_miss=0.10, eps_fa=0.03), "misses <= 10% and false alarms <= 3%"),
                ("conditional", dict(eps_miss=0.05, eps_fa=0.02), "misses <= 5% and false alarms <= 2%")]
    def conf_of(im, r, cls):
        _, cf, cl = S["dets"][(im["id"], r)]
        m = cl == cls
        return float(cf[m].max()) if m.any() else 0.0
    data = {name: ({r0: np.array([conf_of(im, r0, cls) for im in images]) for r0 in a.first_stages},
                   np.array([int(conf_of(im, 640, cls) >= 0.25) for im in images])) for name, cls in classes.items()}
    for risk, kw, label in settings:
        rows = []
        for rep in range(a.reps):
            for name in classes:
                sc, ref = data[name]
                perm = np.random.default_rng(1000 * rep + 7).permutation(len(ref))
                t = len(ref) // 3
                tr, ce, te = perm[:t], perm[t:2 * t], perm[2 * t:3 * t]
                mk = lambda r0, idx: CascadeSample(sc[r0][idx].tolist(), ref[idx].tolist(), ref[idx].tolist())
                ladders = [Ladder(mk(r0, tr), mk(r0, ce), (lat[str(r0)], c1), str(r0)) for r0 in a.first_stages]
                cal = calibrate_ladders(ladders, risk=risk, **kw)
                if cal.certified:
                    s0 = sc[int(cal.ladder_name)][te]
                    dec = [cal.run(s, second=lambda r=r: r) for s, r in zip(s0, ref[te])]
                    d = np.array([x[0] for x in dec]); esc = np.array([x[1] for x in dec])
                    cost = cal.costs[0] + c1 * esc.mean()
                else:
                    d, cost = ref[te].copy(), c1
                pos = ref[te] == 1
                if risk == "disagree":
                    viol = significant(int((d != ref[te]).sum()), len(te), kw["eps"])
                else:
                    viol = (significant(int((d[pos] == 0).sum()), int(pos.sum()), kw["eps_miss"])
                            or significant(int((d[~pos] == 1).sum()), int((~pos).sum()), kw["eps_fa"]))
                rows.append((cal.certified, cost, viol, float((d != ref[te]).mean()),
                             float((d[pos] == 0).mean()) if pos.any() else 0.0, float((d[~pos] == 1).mean())))
        r = np.array(rows, float)
        print(f"{label:42s} certified {r[:, 0].mean()*100:4.0f}% | cost {r[:, 1].mean():5.1f} ms "
              f"(saving {(1 - r[:, 1].mean() / c1) * 100:3.0f}% vs {c1:.0f}) | test disagreement {r[:, 3].mean()*100:4.2f}% "
              f"miss {r[:, 4].mean()*100:4.1f}% FA {r[:, 5].mean()*100:4.2f}% | significant violations {int(r[:, 2].sum())}/{len(r)}")


if __name__ == "__main__":
    main()
