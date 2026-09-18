#!/usr/bin/env python3
"""
Final evaluation with an honest protocol.

For each class ("is there a <class>?"), on real YOLO detections:
  * every knob (ACIES v2: carry and the error price lambda; cascade: r0/r1/thresholds) is
    chosen on the CALIBRATION half only, by the same rule:
        "cheapest setting whose cross-fitted calibration accuracy is within `--slack` of Fixed 640"
  * the chosen setting is then frozen and measured once on the TEST half;
  * the accuracy difference against Fixed 640 comes with a paired-bootstrap 95 % interval
    (same test images resampled for both policies).

    python3 benchmarks/final_eval.py --measure <pkl> --subset <json> --clean-ms-from <pkl>
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from track_p_v2 import EDGES, calibrate, make_actions, run_v2  # noqa: E402
from run_benchmark import Data  # noqa: E402

from acies import HardwareProfile  # noqa: E402
from acies.channel import ChannelConfig, ChannelLearner  # noqa: E402

LAMBDAS = (400, 600, 800, 1000, 1300, 1700, 2200, 2800, 3500, 4500, 6000, 8000)


def paired_bootstrap(correct_a, correct_b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    n = len(a)
    d = [(a[j] - b[j]).mean() for j in (rng.integers(0, n, n) for _ in range(B))]
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--clean-ms-from", default=None)
    ap.add_argument("--classes", default="person:0,car:2,chair:56,bottle:39,cup:41,dog:16,tv:62,bench:13")
    ap.add_argument("--slack", type=float, default=0.003)
    a = ap.parse_args()

    D = Data(a.measure, a.subset, a.clean_ms_from)
    R, T, C = D.R, D.test, D.cal
    k640 = R.index(640)
    hw = HardwareProfile(name="ms", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)
    actions = make_actions(D, D.ms[C].mean(0))
    K = len(EDGES) + 1
    h = len(C) // 2
    folds = [(C[:h], C[h:]), (C[h:], C[:h])]
    print(f"images={D.n} cal={len(C)} test={len(T)}  slack={a.slack*100:.1f} pt  cost: {D.cost_model}\n")
    print(f"{'class':7s} {'prev':>5s} | {'Fixed 640':>12s} | {'ACIES v2 (chosen on cal)':>34s} | "
          f"{'acc diff vs F640 [95% CI]':>27s} | {'cascade (chosen on cal)':>24s}")
    tot = {"f": [], "v2": [], "c": []}

    for spec in a.classes.split(","):
        name, cls = spec.split(":")
        cls = int(cls)
        Y = np.array([int((g[1] == cls).any()) for g in D.gts])
        conf = np.array([[float(d[1][d[2] == cls].max()) if (d[2] == cls).any() else 0.0 for d in row]
                         for row in D.dets])
        out = np.digitize(conf, EDGES)
        vote = (conf >= 0.25).astype(int)
        if Y[T].sum() < 30 or Y[C].sum() < 30:
            continue
        target = float((vote[C, k640] == Y[C]).mean()) - a.slack

        # ---- ACIES v2: carry on cal, then cheapest lambda meeting the target (cross-fitted on cal)
        tau, carry = calibrate(actions, hw, out, Y, C)
        chosen = None
        for lam in LAMBDAS:
            acc, ms = [], []
            for tr, va in folds:
                ln = ChannelLearner(len(actions), K).fit(out[tr], Y[tr])
                res = run_v2(actions, ln, ChannelConfig(error_cost=lam, tau=tau, carry=carry, hardware=hw),
                             out, va)
                acc += [d == Y[i] for (d, _, _, _), i in zip(res, va)]
                ms += [c for _, _, c, _ in res]
            if np.mean(acc) >= target:
                chosen = lam
                break
        chosen = chosen or LAMBDAS[-1]
        ln = ChannelLearner(len(actions), K).fit(out[C], Y[C])
        res = run_v2(actions, ln, ChannelConfig(error_cost=chosen, tau=tau, carry=carry, hardware=hw), out, T)
        ok_v2 = np.array([d == Y[i] for (d, _, _, _), i in zip(res, T)])
        ms_v2 = float(np.mean([c for _, _, c, _ in res]))

        # ---- cascade: same rule on cal (cheapest parameters meeting the target), frozen for test
        def casc(idx, r0, r1, hi, lo):
            k0, k1 = R.index(r0), R.index(r1)
            d = np.where(conf[idx, k0] >= hi, 1, np.where(conf[idx, k0] < lo, 0, (conf[idx, k1] >= 0.25).astype(int)))
            esc = (conf[idx, k0] < hi) & (conf[idx, k0] >= lo)
            cost = D.ms[idx, k0] + np.where(esc, D.ms[idx, k1], 0.0)
            return (d == Y[idx]), cost
        best = None
        for r0, r1 in ((160, 640), (224, 640), (320, 640), (224, 800), (416, 640)):
            for hi in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                for lo in (0.005, 0.01, 0.02, 0.05, 0.08, 0.12, 0.2):
                    ok, cost = casc(C, r0, r1, hi, lo)
                    if ok.mean() >= target and (best is None or cost.mean() < best[0]):
                        best = (cost.mean(), (r0, r1, hi, lo))
        if best is None:
            ok_c, ms_c, lab_c = np.zeros(len(T), bool), float("nan"), "n/a"
        else:
            ok_c, cost_c = casc(T, *best[1])
            ms_c, lab_c = float(cost_c.mean()), f"{best[1][0]}->{best[1][1]}"

        ok_f = (vote[T, k640] == Y[T])
        ms_f = float(D.ms[T, k640].mean())
        dm, lo_, hi_ = paired_bootstrap(ok_v2, ok_f)
        sav = (1 - ms_v2 / ms_f) * 100
        savc = (1 - ms_c / ms_f) * 100
        print(f"{name:7s} {Y[T].mean():5.2f} | {ok_f.mean()*100:5.1f}% {ms_f:5.0f}ms | "
              f"{ok_v2.mean()*100:5.1f}% {ms_v2:5.0f}ms ({sav:+4.0f}%) lam={chosen:<5d} | "
              f"{dm*100:+5.2f} [{lo_*100:+5.2f}, {hi_*100:+5.2f}] | "
              f"{ok_c.mean()*100:5.1f}% {ms_c:4.0f}ms ({savc:+4.0f}%) {lab_c}")
        tot["f"].append((ok_f.mean(), ms_f))
        tot["v2"].append((ok_v2.mean(), ms_v2))
        tot["c"].append((ok_c.mean(), ms_c))

    m = lambda k: (np.mean([x[0] for x in tot[k]]) * 100, np.mean([x[1] for x in tot[k]]))
    print("\nmean over classes:  Fixed 640 %.1f%% @ %.0f ms | ACIES v2 %.1f%% @ %.0f ms | cascade %.1f%% @ %.0f ms"
          % (*m("f"), *m("v2"), *m("c")))
    print("saving of ACIES v2 vs Fixed 640: %.0f%%" % ((1 - m("v2")[1] / m("f")[1]) * 100))


if __name__ == "__main__":
    main()
