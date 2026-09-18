#!/usr/bin/env python3
"""
Track P with the channel controller: "is there a <class> in the image?" on real YOLO output.

    python3 benchmarks/track_p_v2.py --measure <pkl> --subset <json> [--clean-ms-from <pkl>]

Compared on the TEST half (channels and tau are learned on the calibration half only):
    Fixed(r)        YOLO at one resolution, decision = top confidence >= 0.25
    Cascade         hand-tuned 2-stage confidence cascade (best of a sweep — optimistic)
    ACIES (v1)      the original APCController (binary clarity model), thr 0.95
    ACIES (v2)      ChannelController: multi-outcome channels + value of information / cost
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_benchmark import Data  # noqa: E402

from acies import APCConfig, APCController, HardwareProfile  # noqa: E402
from acies.actions import Action, ActionType  # noqa: E402
from acies.channel import ChannelConfig, ChannelController, ChannelLearner  # noqa: E402

EDGES = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.85])   # -> 9 outcomes
LAMBDAS = (20, 40, 80, 150, 300, 600, 1200, 2500, 5000)


def make_actions(D, ms):
    return [Action(id=k, name=f"{r}p", action_type=ActionType.RESOLUTION,
                   params={"resolution": float(r), "crop_area_ratio": 1.0},
                   base_latency_ms=float(ms[k]), pixel_ratio=(r / 1024) ** 2)
            for k, r in enumerate(D.R)]


def run_v2(actions, learner, cfg, outcomes, idx):
    """Run the controller on images `idx`; returns per-image (decision, belief, cost, steps)."""
    ctl = ChannelController(actions, learner, cfg)
    out = []
    for i in idx:
        ctl.begin()
        while (a := ctl.next_action()) is not None:
            ctl.observe(a, int(outcomes[i, a.id]))
        r = ctl.finish()
        out.append((r.decision, r.belief, r.total_cost, r.n_steps))
    return out


def logloss(res, y):
    eps = 1e-4
    return float(np.mean([-math.log(min(max(b if yy == 1 else 1 - b, eps), 1 - eps))
                          for (_, b, _, _), yy in zip(res, y)]))


def calibrate(actions, hw, outcomes, Y, cal, lam=600.0):
    """
    `carry` minimising held-out log-loss inside the calibration half (2-fold, learner refit).
    tau stays 1: with carry < 1 the redundancy is already handled, and letting log-loss also
    shrink tau made rare classes collapse to "always no" (measured), so it is not calibrated.
    """
    h = len(cal) // 2
    folds = [(cal[:h], cal[h:]), (cal[h:], cal[:h])]
    best = None
    for tau in (1.0,):
        for carry in (0.0, 0.25, 0.5, 0.75, 1.0):
            ll = 0.0
            for tr, va in folds:
                ln = ChannelLearner(len(actions), len(EDGES) + 1).fit(outcomes[tr], Y[tr])
                res = run_v2(actions, ln, ChannelConfig(error_cost=lam, tau=tau, carry=carry, hardware=hw),
                             outcomes, va)
                ll += logloss(res, Y[va])
            if best is None or ll < best[0]:
                best = (ll, tau, carry)
    return best[1], best[2]


def evaluate_class(D, cls, name, verbose=True):
    R, T, C = D.R, D.test, D.cal
    Y = np.array([int((g[1] == cls).any()) for g in D.gts])
    conf = np.array([[float(d[1][d[2] == cls].max()) if (d[2] == cls).any() else 0.0 for d in row]
                     for row in D.dets])
    outcomes = np.digitize(conf, EDGES)                                # [n, |R|] in 0..8
    vote = (conf >= 0.25).astype(int)
    hw = HardwareProfile(name="ms", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)
    ms_cal = D.ms[C].mean(0)
    actions = make_actions(D, ms_cal)

    rows = []
    for k, r in enumerate(R):
        rows.append(("Fixed %d" % r, float((vote[T, k] == Y[T]).mean()), float(D.ms[T, k].mean()), 1.0))

    # 2-stage confidence cascade: parameters swept and Pareto-selected on CAL, then frozen for TEST
    def casc(idx, r0, r1, hi, lo):
        k0, k1 = R.index(r0), R.index(r1)
        d, c = [], []
        for i in idx:
            cost = D.ms[i, k0]
            if conf[i, k0] >= hi:
                d.append(1)
            elif conf[i, k0] < lo:
                d.append(0)
            else:
                cost += D.ms[i, k1]
                d.append(int(conf[i, k1] >= 0.25))
            c.append(cost)
        return float((np.array(d) == Y[idx]).mean()), float(np.mean(c))

    cal_pts = []
    for r0, r1 in ((160, 640), (224, 640), (320, 640), (224, 800)):
        for hi in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for lo in (0.01, 0.02, 0.05, 0.08, 0.12, 0.2):
                acc, ms = casc(C, r0, r1, hi, lo)
                cal_pts.append((ms, acc, (r0, r1, hi, lo)))
    keep, best_acc = [], -1
    for ms, acc, prm in sorted(cal_pts):
        if acc > best_acc + 1e-9:
            acct, mst = casc(T, *prm)
            keep.append((f"Cascade {prm[0]}->{prm[1]} hi={prm[2]} lo={prm[3]}", acct, mst, 0))
            best_acc = acc

    # ACIES v1 (original controller, binary clarity, learner frozen after calibration)
    v1 = []
    for thr in (0.90, 0.95, 0.99):
        apc = APCController(APCConfig(confidence_threshold=thr, max_steps=6, hardware=hw,
                                      max_cost_per_image=1e9), actions=actions)
        for i in C:
            for k in range(len(R)):
                apc.feedback(actions[k], bool(vote[i, k] == Y[i]))
        ok = cost = 0
        for i in T:
            apc.begin()
            while (a := apc.next_action()) is not None:
                apc.observe(a, int(vote[i, a.id]))
            res = apc.finish()
            ok += res.decision == Y[i]
            cost += res.total_cost
        v1.append((f"ACIES v1 thr={thr}", ok / len(T), cost / len(T), 0))

    # ACIES v2
    tau, carry = calibrate(actions, hw, outcomes, Y, C)
    learner = ChannelLearner(len(actions), len(EDGES) + 1).fit(outcomes[C], Y[C])
    v2 = []
    for lam in LAMBDAS:
        res = run_v2(actions, learner, ChannelConfig(error_cost=lam, tau=tau, carry=carry, hardware=hw), outcomes, T)
        acc = float(np.mean([d == Y[i] for (d, _, _, _), i in zip(res, T)]))
        v2.append((f"ACIES v2 lambda={lam}", acc, float(np.mean([c for _, _, c, _ in res])),
                   float(np.mean([s for _, _, _, s in res]))))

    if verbose:
        print(f"\n===== '{name}' present?  prevalence(test)={Y[T].mean():.2f}  tau={tau} carry={carry}  "
              f"(n_test={len(T)}) =====")
        print(f"{'policy':38s} {'acc %':>6s} {'ms/img':>7s} {'steps':>6s}")
        for grp in (rows, v1, keep, v2):
            for n, acc, ms, st in grp:
                print(f"{n[:38]:38s} {acc*100:6.1f} {ms:7.1f} {st if st else float('nan'):6.2f}")
            print()
    return {"Y": Y, "outcomes": outcomes, "fixed": rows, "v1": v1, "cascade": keep, "v2": v2, "tau": tau,
            "learner": learner, "actions": actions, "hw": hw, "vote": vote}


def cost_at_accuracy(points, target):
    """Cheapest cost at which a policy family reaches `target` accuracy (linear interpolation)."""
    pts = sorted((p[2], p[1]) for p in points)
    pareto, best = [], -1
    for ms, acc in pts:
        if acc > best + 1e-9:
            pareto.append((ms, acc))
            best = acc
    for (m0, a0), (m1, a1) in zip(pareto, pareto[1:]):
        if a0 <= target <= a1:
            return m0 + (target - a0) / max(a1 - a0, 1e-12) * (m1 - m0)
    return pareto[0][0] if target <= pareto[0][1] else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--clean-ms-from", default=None)
    ap.add_argument("--classes", default="person:0,car:2,chair:56,bottle:39")
    a = ap.parse_args()
    D = Data(a.measure, a.subset, a.clean_ms_from)
    print(f"images={D.n} cal={len(D.cal)} test={len(D.test)}  cost model: {D.cost_model}")
    summary = []
    for spec in a.classes.split(","):
        name, c = spec.split(":")
        r = evaluate_class(D, int(c), name)
        f640 = next(x for x in r["fixed"] if x[0] == "Fixed 640")
        for lbl, fam in (("v1", r["v1"]), ("cascade", r["cascade"]), ("v2", r["v2"])):
            summary.append((name, lbl, cost_at_accuracy(fam, f640[1]), f640[2], f640[1]))
    print("\n===== cost to reach the accuracy of Fixed 640 (lower is better) =====")
    print(f"{'class':8s} {'Fixed 640':>12s} {'ACIES v1':>10s} {'Cascade*':>10s} {'ACIES v2':>10s}   (cascade tuned on cal)")
    for name in dict.fromkeys(s[0] for s in summary):
        rows = {s[1]: s for s in summary if s[0] == name}
        f = rows["v2"]
        fmt = lambda k: (f"{rows[k][2]:6.1f} ms" if not math.isnan(rows[k][2]) else "  n/a    ")
        print(f"{name:8s} {f[3]:6.1f} ms/{f[4]*100:.0f}% {fmt('v1'):>10s} {fmt('cascade'):>10s} {fmt('v2'):>10s}")


if __name__ == "__main__":
    main()
