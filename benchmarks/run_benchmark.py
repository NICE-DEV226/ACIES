#!/usr/bin/env python3
"""
Real benchmark: YOLO at fixed resolution vs adaptive-resolution policies.

    python3 benchmarks/run_benchmark.py --measure <measure_*.pkl> --subset <subset_*.json>

Everything is evaluated on the cached real detections and measured latencies from
benchmarks/measure.py. Images are split by index parity: even = calibration (anything that
learns, learns here), odd = test (all reported numbers). No policy sees test labels.

TRACK D — detection quality (mAP50-95), the metric YOLO is judged on:
    Fixed(r)          YOLO at one resolution                        (the baseline frontier)
    Oracle            picks the resolution per image knowing the ground truth (upper bound)
    Cascade           deployable early-exit on detection confidence (no ACIES, no learning)
    Selector          deployable: one cheap pass -> predict quality per resolution -> pick

TRACK P — ACIES's native task (a binary decision): "is there a person in the image?"
    Fixed(r)          YOLO at one resolution votes
    ACIES (shipped)   APCController, clarities learned on the calibration half, then frozen
    DP-optimal        acies.optimal under ACIES's own i.i.d. model (its lower-bound cost)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evalkit import image_f1, image_stats, mean_ap  # noqa: E402


# ----------------------------------------------------------------------------- data

class Data:
    def __init__(self, measure_path, subset_path, clean_ms_from=None):
        """clean_ms_from: measure pickle taken on an idle machine. When given, cost is the
        per-resolution mean latency of that run (YOLO compute is ~content independent), so
        images measured under CPU contention do not distort costs."""
        S = pickle.load(open(measure_path, "rb"))
        sub = json.load(open(subset_path))
        self.R = S["resolutions"]
        self.threads = S.get("threads")
        self.weights = S.get("weights")
        images = [im for im in sub["images"] if im["id"] in S["ms"]]
        self.n = len(images)
        self.ms = np.array([S["ms"][im["id"]] for im in images])                   # [n, |R|]
        if clean_ms_from:
            clean = np.array(list(pickle.load(open(clean_ms_from, "rb"))["ms"].values()))
            self.ms = np.tile(clean.mean(0), (len(images), 1))
            self.cost_model = "per-resolution mean of an idle-machine run"
        else:
            self.cost_model = "per-image measured latency"
        self.dets = [[S["dets"][(im["id"], r)] for r in self.R] for im in images]  # [n][|R|]
        self.gts = [(np.array(im["boxes"], np.float32).reshape(-1, 4),
                     np.array(im["cls"], np.int64)) for im in images]
        idx = np.arange(self.n)
        self.cal, self.test = idx[idx % 2 == 0], idx[idx % 2 == 1]
        self.f1 = np.array([[image_f1(self.dets[i][k], self.gts[i]) for k in range(len(self.R))]
                            for i in range(self.n)])

    def map(self, ids, choice):
        """mAP50-95 / mAP50 when image ids[j] is answered with resolution index choice[j]."""
        return mean_ap([self.dets[i][c] for i, c in zip(ids, choice)], [self.gts[i] for i in ids])


def pareto(points):
    """Keep non-dominated (cost, quality) points: lower cost, higher quality."""
    out, best = [], -1.0
    for p in sorted(points, key=lambda p: (p["ms"], -p["map"])):
        if p["map"] > best + 1e-9:
            out.append(p)
            best = p["map"]
    return out


def interp_ms_at(frontier, quality):
    """Cost of the fixed-resolution frontier (linear interpolation) at a given quality."""
    pts = pareto(frontier)                      # dominated fixed points (e.g. 1024) must not count
    if quality <= pts[0]["map"]:
        return pts[0]["ms"]
    for a, b in zip(pts, pts[1:]):
        if a["map"] <= quality <= b["map"]:
            t = (quality - a["map"]) / max(b["map"] - a["map"], 1e-12)
            return a["ms"] + t * (b["ms"] - a["ms"])
    return float("inf")


def bootstrap_delta(D: Data, ids, choice_a, choice_b, B=300, seed=0):
    """Paired bootstrap over images of mAP50-95(A) - mAP50-95(B): (mean, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    ca, cb = np.asarray(choice_a), np.asarray(choice_b)
    deltas = []
    for _ in range(B):
        j = rng.integers(0, len(ids), len(ids))
        a, _ = D.map(ids[j], ca[j])
        b, _ = D.map(ids[j], cb[j])
        deltas.append(a - b)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def bootstrap_delta(D: Data, ids, choice_a, choice_b, B=300, seed=0):
    """Paired bootstrap over images of mAP50-95(A) - mAP50-95(B): (mean, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(ids)
    ca, cb = np.asarray(choice_a), np.asarray(choice_b)
    deltas = []
    for _ in range(B):
        j = rng.integers(0, len(ids), len(ids))
        a, _ = D.map(ids[j], ca[j])
        b, _ = D.map(ids[j], cb[j])
        deltas.append(a - b)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


# ----------------------------------------------------------------------------- track D

def track_d(D: Data, ref_res=640):
    T, R = D.test, D.R
    ref = R.index(ref_res)
    out = {"fixed": [], "oracle": [], "cascade": [], "selector": []}

    for k, r in enumerate(R):
        m50_95, m50 = D.map(T, [k] * len(T))
        out["fixed"].append({"name": f"Fixed {r}", "res": r, "map": m50_95, "map50": m50,
                             "ms": float(D.ms[T, k].mean())})

    # Oracles (know ground truth; no probing cost) — upper bounds, not deployable
    cheapest = []
    for i in T:
        ok = [k for k in range(len(R)) if D.f1[i, k] >= D.f1[i, ref] - 1e-9]
        cheapest.append(min(ok, key=lambda k: D.ms[i, k]))
    m, m50 = D.map(T, cheapest)
    out["oracle"].append({"name": f"Oracle-cheapest (>= Fixed {ref_res} per image)", "map": m, "map50": m50,
                          "ms": float(np.mean([D.ms[i, c] for i, c in zip(T, cheapest)])),
                          "res_mix": {R[k]: int(cheapest.count(k)) for k in range(len(R))}})
    best = [max(range(len(R)), key=lambda k: (round(D.f1[i, k], 9), -D.ms[i, k])) for i in T]
    m, m50 = D.map(T, best)
    out["oracle"].append({"name": "Oracle-best (max F1 per image)", "map": m, "map50": m50,
                          "ms": float(np.mean([D.ms[i, c] for i, c in zip(T, best)])),
                          "res_mix": {R[k]: int(best.count(k)) for k in range(len(R))}})

    # Cascades on confidence — deployable baseline without ACIES or learning
    ladders = [(320, 640), (224, 640), (320, 800), (416, 800), (224, 416, 640), (160, 320, 640)]
    for lad in ladders:
        ks = [R.index(r) for r in lad]
        for t in np.linspace(0.3, 0.95, 14):
            choice, cost = [], []
            for i in T:
                c = 0.0
                for s, k in enumerate(ks):
                    c += D.ms[i, k]
                    if s == len(ks) - 1 or image_stats(D.dets[i][k])["top3"] >= t:
                        choice.append(k)
                        break
                cost.append(c)
            m, m50 = D.map(T, choice)
            out["cascade"].append({"name": f"Cascade {lad} t={t:.2f}", "ladder": lad, "t": float(t),
                                   "map": m, "map50": m50, "ms": float(np.mean(cost))})

    # Selector: one cheap pass -> ridge-predict F1 per resolution -> pick by utility
    def feats(i, k0):
        s = image_stats(D.dets[i][k0])
        v = np.array([np.log1p(s["n"]), np.log1p(s["n50"]), s["max"], s["mean"], s["top3"],
                      np.log1p(s["small"]), np.log1p(s["area"])])
        return np.concatenate([v, np.outer(v, v)[np.triu_indices(len(v))]])

    for r0 in (160, 224, 320):
        k0 = R.index(r0)
        X = np.array([feats(i, k0) for i in range(D.n)])
        mu_, sd_ = X[D.cal].mean(0), X[D.cal].std(0) + 1e-6
        Z = np.hstack([(X - mu_) / sd_, np.ones((D.n, 1))])
        lam = 10.0
        A = Z[D.cal].T @ Z[D.cal] + lam * np.eye(Z.shape[1])
        W = np.linalg.solve(A, Z[D.cal].T @ D.f1[D.cal])                    # [d, |R|]
        pred = Z @ W                                                         # predicted F1 per res
        for mu in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03):
            choice, cost = [], []
            for i in T:
                extra = np.array([0.0 if k == k0 else D.ms[i, k] for k in range(len(R))])
                k = int(np.argmax(pred[i] - mu * extra))
                choice.append(k)
                cost.append(D.ms[i, k0] + extra[k])
            m, m50 = D.map(T, choice)
            out["selector"].append({"name": f"Selector r0={r0} mu={mu}", "r0": r0, "mu": mu,
                                    "map": m, "map50": m50, "ms": float(np.mean(cost)),
                                    "choice": [int(c) for c in choice]})
    return out


# ----------------------------------------------------------------------------- track P

def track_p(D: Data, person=0, conf_thr=0.25):
    from acies import APCConfig, APCController, BeliefState, HardwareProfile
    from acies.actions import Action, ActionType
    from acies.optimal import solve_optimal

    n, R = D.n, D.R
    Y = np.array([int((g[1] == person).any()) for g in D.gts])
    conf = np.array([[float(d[1][d[2] == person].max()) if (d[2] == person).any() else 0.0
                      for d in row] for row in D.dets])
    obs = (conf >= conf_thr).astype(int)                                     # [n, |R|] the vote

    T, C = D.test, D.cal
    out = {"prevalence_test": float(Y[T].mean()), "fixed": [], "acies": [], "dp": [], "cascade": []}
    acc_cal = [float((obs[C, k] == Y[C]).mean()) for k in range(len(R))]
    ms_cal = D.ms[C].mean(0)

    for k, r in enumerate(R):
        out["fixed"].append({"name": f"Fixed {r}", "res": r, "acc": float((obs[T, k] == Y[T]).mean()),
                             "ms": float(D.ms[T, k].mean())})

    hw = HardwareProfile(name="measured-ms", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)
    actions = [Action(id=k, name=f"{r}p", action_type=ActionType.RESOLUTION,
                      params={"resolution": float(r), "crop_area_ratio": 1.0},
                      base_latency_ms=float(ms_cal[k]), pixel_ratio=(r / 1024) ** 2)
               for k, r in enumerate(R)]

    def fresh_controller(thr, max_steps):
        apc = APCController(APCConfig(confidence_threshold=thr, max_steps=max_steps, hardware=hw,
                                      max_cost_per_image=1e9), actions=actions)
        for i in C:                                     # calibration: labels allowed, every action
            for k in range(len(R)):
                apc.feedback(actions[k], bool(obs[i, k] == Y[i]))
        return apc

    for thr in (0.80, 0.90, 0.95, 0.99):
        apc = fresh_controller(thr, max_steps=6)
        correct = cost = abst = steps = 0
        mix = {}
        for i in T:
            apc.begin()
            while (a := apc.next_action()) is not None:
                apc.observe(a, int(obs[i, a.id]))       # learner frozen: no label on test images
                mix[a.name] = mix.get(a.name, 0) + 1
            r = apc.finish()
            cost += r.total_cost
            steps += r.n_steps
            abst += r.abstained
            correct += (r.decision == Y[i])
        out["acies"].append({"name": f"ACIES thr={thr}", "thr": thr, "acc": correct / len(T),
                             "ms": cost / len(T), "steps": steps / len(T), "abstain": abst / len(T),
                             "mix": dict(sorted(mix.items(), key=lambda kv: -kv[1])[:4])})

    clar = {a.name: min(max(acc_cal[k], 0.51), 0.99) for k, a in enumerate(actions)}
    for lam in (5, 20, 50, 100, 300, 1000):
        pol = solve_optimal(actions, clar, hw, error_cost=lam, horizon=6)
        correct = cost = steps = 0
        for i in T:
            b, t = BeliefState(prior=0.5), 0
            while (a := pol.action(b.belief, t)) is not None:
                b.update(int(obs[i, a.id]), clar[a.name])
                cost += a.base_latency_ms
                t += 1
            steps += t
            correct += (b.decision == Y[i])
        out["dp"].append({"name": f"DP-optimal lambda={lam}", "acc": correct / len(T),
                          "ms": cost / len(T), "steps": steps / len(T)})

    # simple 2-stage confidence cascade on the same task (no ACIES)
    for r0, r1 in ((224, 640), (320, 640), (224, 1024)):
        k0, k1 = R.index(r0), R.index(r1)
        for hi in (0.5, 0.7, 0.9):
            for lo in (0.02, 0.05, 0.1):
                dec, cost = [], []
                for i in T:
                    c = D.ms[i, k0]
                    if conf[i, k0] >= hi:
                        dec.append(1)
                    elif conf[i, k0] < lo:
                        dec.append(0)
                    else:
                        c += D.ms[i, k1]
                        dec.append(int(conf[i, k1] >= conf_thr))
                    cost.append(c)
                out["cascade"].append({"name": f"Cascade {r0}->{r1} hi={hi} lo={lo}",
                                       "acc": float((np.array(dec) == Y[T]).mean()),
                                       "ms": float(np.mean(cost))})
    return out


# ----------------------------------------------------------------------------- report

def fmt_d(rows, frontier):
    lines = [f"{'policy':52s} {'mAP50-95':>8s} {'mAP50':>7s} {'ms/img':>7s} {'ms @ same mAP*':>15s}"]
    for p in rows:
        eq = interp_ms_at(frontier, p["map"])
        eqs = f"{eq:6.1f} ({(1 - p['ms'] / eq) * 100:+.0f}%)" if np.isfinite(eq) else "   > Fixed max"
        lines.append(f"{p['name'][:52]:52s} {p['map']*100:8.2f} {p['map50']*100:7.2f} {p['ms']:7.1f} {eqs:>15s}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--clean-ms-from", default=None)
    a = ap.parse_args()

    D = Data(a.measure, a.subset, a.clean_ms_from)
    print(f"model={D.weights}  images={D.n} (cal {len(D.cal)} / test {len(D.test)})  "
          f"torch threads={D.threads}  resolutions={D.R}\ncost model: {D.cost_model}\n")

    d = track_d(D)
    fixed = d["fixed"]
    print("== TRACK D — detection (mAP50-95, test half) ==")
    print("*ms at which the FIXED-resolution curve reaches the same mAP (linear interpolation);"
          " % = saving of the policy vs that\n")
    print(fmt_d(fixed, fixed), "\n")
    print(fmt_d(d["oracle"], fixed), "\n")
    for key in ("cascade", "selector"):
        pf = pareto(d[key])
        print(f"-- {key}: Pareto-optimal points --")
        print(fmt_d(pf, fixed), "\n")

    print("-- paired bootstrap (300 resamples of the test images): delta mAP50-95, 95% CI --")
    T = D.test
    fx = {r: [D.R.index(r)] * len(T) for r in (640, 800)}
    for sel in d["selector"]:
        if sel["mu"] == 0.0:
            for r in (640, 800):
                m, lo, hi = bootstrap_delta(D, T, sel["choice"], fx[r])
                print(f"{sel['name']:32s} ({sel['ms']:.0f} ms) minus Fixed {r} ({D.ms[T, D.R.index(r)].mean():.0f} ms): "
                      f"{m*100:+.2f} mAP  [{lo*100:+.2f}, {hi*100:+.2f}]")
    print()

    print("-- paired bootstrap (300 resamples of the test images): delta mAP50-95, 95% CI --")
    T = D.test
    for sel in d["selector"]:
        if sel["mu"] == 0.0:
            for r in (640, 800):
                m, lo, hi = bootstrap_delta(D, T, sel["choice"], [D.R.index(r)] * len(T))
                print(f"{sel['name']:26s} ({sel['ms']:.0f} ms) minus Fixed {r} ({D.ms[T, D.R.index(r)].mean():.0f} ms): "
                      f"{m*100:+.2f} mAP  [{lo*100:+.2f}, {hi*100:+.2f}]")
    print()

    p = track_p(D)
    print(f"== TRACK P — 'person present?' (test half, prevalence {p['prevalence_test']:.2f}) ==")
    print(f"{'policy':40s} {'acc':>6s} {'ms/img':>7s} {'steps':>6s} {'abst':>5s}  top actions")
    for key in ("fixed", "acies", "dp", "cascade"):
        rows = p[key]
        if key == "cascade":
            rows = sorted(rows, key=lambda r: r["ms"])
            keep, best = [], -1
            for r_ in rows:
                if r_["acc"] > best:
                    keep.append(r_)
                    best = r_["acc"]
            rows = keep
        for r_ in rows:
            print(f"{r_['name'][:40]:40s} {r_['acc']*100:6.1f} {r_['ms']:7.1f} "
                  f"{r_.get('steps', float('nan')):6.2f} {r_.get('abstain', 0)*100:4.0f}%  {r_.get('mix', '')}")
        print()

    if a.out:
        json.dump({"track_d": d, "track_p": p, "model": D.weights, "n": D.n}, open(a.out, "w"),
                  indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print("saved", a.out)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].plot([f["ms"] for f in fixed], [f["map"] * 100 for f in fixed], "ko-", label="Fixed YOLO")
        for f in fixed:
            ax[0].annotate(str(f["res"]), (f["ms"], f["map"] * 100), fontsize=7, xytext=(3, -9),
                           textcoords="offset points")
        for key, sty, lab in (("cascade", "s--", "Cascade (deployable)"), ("selector", "^-", "Selector (deployable)")):
            pf = pareto(d[key])
            ax[0].plot([q["ms"] for q in pf], [q["map"] * 100 for q in pf], sty, label=lab)
        for o in d["oracle"]:
            ax[0].plot(o["ms"], o["map"] * 100, "r*", ms=12, label=o["name"][:28])
        ax[0].set_xlabel("latency (ms/image, CPU)")
        ax[0].set_ylabel("mAP50-95 (%)")
        ax[0].set_title("Track D — YOLO detection: quality vs cost")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=.3)
        ax[1].plot([f["ms"] for f in p["fixed"]], [f["acc"] * 100 for f in p["fixed"]], "ko-", label="Fixed YOLO")
        ax[1].plot([f["ms"] for f in p["acies"]], [f["acc"] * 100 for f in p["acies"]], "rs-", label="ACIES")
        ax[1].plot([f["ms"] for f in p["dp"]], [f["acc"] * 100 for f in p["dp"]], "g^-", label="DP-optimal (model)")
        cs = sorted(p["cascade"], key=lambda r: r["ms"])
        ax[1].scatter([c["ms"] for c in cs], [c["acc"] * 100 for c in cs], s=8, c="gray", alpha=.5, label="Cascade")
        ax[1].set_xlabel("latency (ms/image, CPU)")
        ax[1].set_ylabel("accuracy (%)")
        ax[1].set_title("Track P — 'person present?'")
        ax[1].legend(fontsize=7)
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print("saved", a.plot)


if __name__ == "__main__":
    main()
