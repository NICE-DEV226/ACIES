#!/usr/bin/env python3
"""
Certified evaluation: cheapest setting that is GUARANTEED to stay within epsilon of the
full-resolution system, for ACIES and for the existing alternatives, on real YOLO output.

    python3 benchmarks/certified_eval.py --measure <pkl> --subset <json> --clean-ms-from <pkl>

Decision (per class c): "is there a c in the image?" = top confidence of c >= 0.25.
Reference system: the same decision at the reference (largest) resolution. No labels are used anywhere in the pipeline:
channels are learned from the reference's decisions and settings are certified against them.

Each repetition draws a random split of the images into TRAIN / CERT / TEST (1000 each):
  TRAIN  fits whatever must be fitted (channels, predictors) and fixes the testing order
  CERT   certifies settings with Learn-then-Test (acies.risk)  -> P(risk <= eps) >= 1 - delta
  TEST   measures the chosen setting once (never seen before)
For every method the settings are tested from the safest to the riskiest on TRAIN, the first
failure stops the sequence, and the cheapest certified setting is deployed. Because the
guarantee is a probability, we count how often the TEST disagreement exceeds eps (target <= delta).
"""
import argparse
import multiprocessing as mp
import os
import pickle
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acies import HardwareProfile  # noqa: E402
from acies.actions import Action, ActionType  # noqa: E402
from acies.channel import ChannelConfig, ChannelController, ChannelLearner  # noqa: E402
from acies.risk import certify, certify_pvalues, risk_pvalue  # noqa: E402

# bin edge at 0.25 = the reference system's own decision boundary, otherwise no channel can reproduce it
EDGES = np.array([0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50, 0.70, 0.85])
K = len(EDGES) + 1
HW = HardwareProfile(name="ms", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)
G = {}          # shared read-only data (inherited by workers through fork)


def ref_index():
    return G["R"].index(G["ref_res"])


def lambdas(lo=0.6, hi=312.0, n=14):
    """Error prices proportional to the reference cost (the original 60..30000 were for a 96 ms reference)."""
    c = float(G["ms"][ref_index()])
    return [round(x) for x in np.geomspace(lo * c, hi * c, n)]


def load(measure, subset, clean, classes):
    """clean: measurement pickle to take per-resolution mean costs from (None: use `measure` itself)."""
    S = pickle.load(open(measure, "rb"))
    sub = json.load(open(subset))
    R = S["resolutions"]
    images = [im for im in sub["images"] if im["id"] in S["ms"]]
    ms = np.array(list((pickle.load(open(clean, "rb")) if clean else S)["ms"].values())).mean(0)
    conf = {}
    for name, cls in classes.items():
        conf[name] = np.array([[float(S["dets"][(im["id"], r)][1][S["dets"][(im["id"], r)][2] == cls].max())
                                if (S["dets"][(im["id"], r)][2] == cls).any() else 0.0 for r in R]
                               for im in images])
    Y = {name: np.array([int(cls in im["cls"]) for im in images]) for name, cls in classes.items()}
    return R, ms, conf, Y


def load_zoom(paths, subset, classes, zms):
    """Zoom outcomes per class: top class-c confidence inside the crop (0 if no crop was made)."""
    crops = {}
    for p in paths:
        crops.update(pickle.load(open(p, "rb"))["crops"])
    ids = [im["id"] for im in json.load(open(subset))["images"]]
    z = {}
    for name, cls in classes.items():
        d = {}
        for S in (224, 320):
            conf = np.zeros(len(ids))
            has = np.zeros(len(ids), bool)
            for i, iid in enumerate(ids):
                v = crops.get((iid, cls, S))
                if v is not None:
                    has[i] = True
                    m = v[2] == cls
                    conf[i] = float(v[1][m].max()) if m.any() else 0.0
            d[S] = (conf, has)
        z[name] = d
    return z


def load_ctx(measure, subset, classes, r0s):
    """Label-free context of the cheap pass r0: own-class stats + all-class detection stats + image stats."""
    import selector_lab as sl
    S = pickle.load(open(measure, "rb"))
    images = [im for im in json.load(open(subset))["images"] if im["id"] in S["ms"]]
    img = np.array([sl.image_features(im["file"])[0] for im in images])
    ctx = {name: {} for name in classes}
    for r0 in r0s:
        glob = np.array([sl.det_features(S["dets"][(im["id"], r0)], (im["w"], im["h"])) for im in images])
        for name, cls in classes.items():
            own = []
            for im in images:
                bx, cf, cl = S["dets"][(im["id"], r0)]
                c = cf[cl == cls]
                own.append([c.max() if len(c) else 0.0, np.log1p((c >= 0.05).sum()), np.log1p((c >= 0.25).sum()),
                            c[c < 0.25].sum() / 3 if len(c) else 0.0])
            ctx[name][r0] = np.hstack([np.array(own), glob, img])
    return ctx


def actions_for(R, ms):
    return [Action(id=k, name=f"{r}p", action_type=ActionType.RESOLUTION,
                   params={"resolution": float(r), "crop_area_ratio": 1.0},
                   base_latency_ms=float(ms[k]), pixel_ratio=(r / 1024) ** 2) for k, r in enumerate(R)]


# ----------------------------------------------------------------------------- method families
# Each family builds, from TRAIN only, a list of settings; each setting is a function
# idx -> (decision[idx], cost[idx]) that never looks at CERT/TEST labels.

def fam_fixed(conf, ref, ms, R, train, name=None):
    return [(f"Fixed {r}", (lambda idx, k=k: ((conf[idx, k] >= 0.25).astype(int), np.full(len(idx), ms[k]))))
            for k, r in enumerate(R)]


def fam_cascade(conf, ref, ms, R, train, name=None):
    out = []
    k1 = ref_index()
    for k0, r0 in enumerate(R):
        if k0 >= k1:
            continue
        for hi in (0.3, 0.4, 0.5, 0.6, 0.8):
            for lo in (0.005, 0.01, 0.02, 0.05, 0.1):
                def f(idx, k0=k0, k1=k1, hi=hi, lo=lo):
                    c0 = conf[idx, k0]
                    esc = (c0 < hi) & (c0 >= lo)
                    d = np.where(c0 >= hi, 1, np.where(c0 < lo, 0, (conf[idx, k1] >= 0.25).astype(int)))
                    return d, ms[k0] + np.where(esc, ms[k1], 0.0)
                out.append((f"Cascade {r0}->{R[k1]} hi={hi} lo={lo}", f))
    return out


def fam_predictor(conf, ref, ms, R, train, name=None):
    """DRNet-style (no retraining): from a cheap pass, predict the smallest resolution whose decision
    matches the reference; ridge on the pass's confidence, chosen by a threshold tau."""
    out = []
    k_ref = ref_index()
    cand = [k for k in range(1, k_ref + 1)]
    for k0 in (0, 1):
        r0 = R[k0]

        def feats(idx, k0=k0):
            c = conf[idx, k0]
            return np.stack([np.ones(len(idx)), c, c * c, np.log(c + 1e-3), (c >= 0.25).astype(float),
                             (c >= 0.05).astype(float)], 1)
        X = feats(train)
        A = X.T @ X + 1.0 * np.eye(X.shape[1])
        W = {}
        for k in cand:
            agree = ((conf[train, k] >= 0.25).astype(int) == ref[train]).astype(float)
            W[k] = np.linalg.solve(A, X.T @ agree)
        for tau in (0.90, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999):
            def f(idx, k0=k0, tau=tau, W=W, feats=feats):
                F = feats(idx)
                chosen = np.full(len(idx), k_ref)
                for k in reversed(cand):                       # smallest resolution predicted good enough
                    ok = (F @ W[k]) >= tau
                    chosen = np.where(ok, k, chosen)
                d = np.array([int(conf[i, c] >= 0.25) for i, c in zip(idx, chosen)])
                cost = ms[k0] + np.where(chosen == k0, 0.0, ms[chosen])
                return d, cost
            out.append((f"Predictor r0={r0} tau={tau}", f))
    return out


def fam_acies(conf, ref, ms, R, train, name=None):
    """ACIES v2: channels learned from the reference's decisions on TRAIN (label-free), planned escalation."""
    acts = actions_for(R, ms)
    outc = np.digitize(conf, EDGES)
    learner = ChannelLearner(len(acts), K).fit(outc[train], ref[train])
    out = []
    for lam in lambdas():
        ctl = ChannelController(acts, learner, ChannelConfig(error_cost=lam, carry=0.0, hardware=HW, grid=101))

        def f(idx, ctl=ctl):
            dec, cost = [], []
            for i in idx:
                ctl.begin()
                while (a := ctl.next_action()) is not None:
                    ctl.observe(a, int(outc[i, a.id]))
                r = ctl.finish()
                dec.append(r.decision)
                cost.append(r.total_cost)
            return np.array(dec), np.array(cost)
        out.append((f"ACIES v2 lambda={lam}", f))
    return out


def _belief_model(X, ref, train):
    """Calibrated P(ref decision = 1 | context of the cheap pass), boosted trees, fitted on TRAIN only."""
    import selector_lab as sl
    m = sl.fit_gbm(X[train], ref[train].astype(float), rounds=80)
    return np.clip(sl.predict_gbm(m, X), 0.002, 0.998)


def fam_acies_asym(conf, ref, ms, R, train, name=None):
    """ACIES v2 with an asymmetric loss: a miss costs rho x a false alarm; grid over (lambda, rho)."""
    acts = actions_for(R, ms)
    outc = np.digitize(conf, EDGES)
    learner = ChannelLearner(len(acts), K).fit(outc[train], ref[train])
    out = []
    for rho in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        for lam in lambdas(1.0, 208.0, 7):
            ctl = ChannelController(acts, learner, ChannelConfig(error_cost=lam, miss_cost=rho, carry=0.0,
                                                                 hardware=HW, grid=101))

            def f(idx, ctl=ctl):
                dec, cost = [], []
                for i in idx:
                    ctl.begin()
                    while (a := ctl.next_action()) is not None:
                        ctl.observe(a, int(outc[i, a.id]))
                    r = ctl.finish()
                    dec.append(r.decision)
                    cost.append(r.total_cost)
                return np.array(dec), np.array(cost)
            out.append((f"ACIES v2 asym rho={rho} lambda={lam}", f))
    return out


def fam_acies_ctx(conf, ref, ms, R, train, name=None):
    """ACIES v3: the first cheap pass yields a belief from a context model; ACIES plans the rest with channels."""
    acts = actions_for(R, ms)
    outc = np.digitize(conf, EDGES)
    learner = ChannelLearner(len(acts), K).fit(outc[train], ref[train])
    out = []
    for r0 in (160, 224):
        k0 = R.index(r0)
        b1 = _belief_model(G["ctx"][name][r0], ref, train)
        for lam in LAMBDAS:
            ctl = ChannelController(acts, learner, ChannelConfig(error_cost=lam, carry=0.0, hardware=HW, grid=101,
                                                                 first_action=k0))

            def f(idx, ctl=ctl, k0=k0, b1=b1):
                dec, cost = [], []
                for i in idx:
                    ctl.begin()
                    a = ctl.next_action()
                    ctl.observe(a, int(outc[i, k0]), belief=float(b1[i]))
                    while (a := ctl.next_action()) is not None:
                        ctl.observe(a, int(outc[i, a.id]))
                    r = ctl.finish()
                    dec.append(r.decision)
                    cost.append(r.total_cost)
                return np.array(dec), np.array(cost)
            out.append((f"ACIES v3 r0={r0} lambda={lam}", f))
    return out


def fam_gate(conf, ref, ms, R, train, name=None):
    """Same context model, no planning: accept the cheap pass if confident enough, else run r1."""
    out = []
    for r0 in (160, 224):
        k0 = R.index(r0)
        b1 = _belief_model(G["ctx"][name][r0], ref, train)
        for r1 in (416, 512, 640):
            k1 = R.index(r1)
            for tau in (0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998):
                def f(idx, k0=k0, k1=k1, tau=tau, b1=b1):
                    sure = np.maximum(b1[idx], 1 - b1[idx]) >= tau
                    d = np.where(sure, (b1[idx] >= 0.5).astype(int), (conf[idx, k1] >= 0.25).astype(int))
                    return d, ms[k0] + np.where(sure, 0.0, ms[k1])
                out.append((f"Gate r0={r0} r1={r1} tau={tau}", f))
    return out


ZOOM_MS = {224: 43.0, 320: 55.0}      # replaced by measured medians via --zoom-ms


def zoom_actions(R, ms):
    """Full-frame actions plus two zoom actions, ordered by cost; 224p (index 0) always runs first."""
    def act(i, name, ms_, pr):
        return Action(id=i, name=name, action_type=ActionType.CROP if name.startswith("Z") else ActionType.RESOLUTION,
                      params={"resolution": 0.0, "crop_area_ratio": 1.0}, base_latency_ms=float(ms_), pixel_ratio=pr)
    keep = [(224, None), (320, None), ("Z224", None), (416, None), ("Z320", None), (512, None), (640, None), (800, None)]
    acts, col = [], []
    for i, (r, _) in enumerate(keep):
        if isinstance(r, str):
            acts.append(act(i, r, ZOOM_MS[int(r[1:])], 0.05))
            col.append(r)
        else:
            k = R.index(r)
            acts.append(act(i, f"{r}p", ms[k], (r / 1024) ** 2))
            col.append(k)
    return acts, col


def fam_acies_zoom(conf, ref, ms, R, train, name=None):
    """ACIES v2 with heterogeneous actions: zoom on the most plausible region of the 224p pass."""
    acts, col = zoom_actions(R, ms)
    zc = G["zoom"][name]
    cols, has = [], []
    for c in col:
        if isinstance(c, str):
            conf_c, has_c = zc[int(c[1:])]
            cols.append(conf_c)
            has.append(has_c)
        else:
            cols.append(conf[:, c])
            has.append(np.ones(len(conf), bool))
    M = np.stack(cols, 1)
    H = np.stack(has, 1)
    outc = np.digitize(M, EDGES)
    learner = ChannelLearner(len(acts), K).fit(outc[train], ref[train])
    out = []
    for lam in LAMBDAS:
        ctl = ChannelController(acts, learner, ChannelConfig(error_cost=lam, carry=0.0, hardware=HW, grid=101,
                                                             first_action=0))

        def f(idx, ctl=ctl):
            dec, cost = [], []
            for i in idx:
                ctl.begin()
                c = 0.0
                while (a := ctl.next_action()) is not None:
                    ctl.observe(a, int(outc[i, a.id]))
                    c += a.base_latency_ms if H[i, a.id] else 0.0      # no candidate region: nothing to run
                dec.append(ctl.finish().decision)
                cost.append(c)
            return np.array(dec), np.array(cost)
        out.append((f"ACIES v2+zoom lambda={lam}", f))
    return out


def fam_zoom_cascade(conf, ref, ms, R, train, name=None):
    """Hand-designed zoom cascade: 224p, then zoom on the uncertain region, optionally 640p if still unsure."""
    c224 = conf[:, R.index(224)]
    k640 = R.index(640)
    out = []
    for S in (224, 320):
        zc, has = G["zoom"][name][S]
        zms = ZOOM_MS[S]
        for hi in (0.3, 0.4, 0.5, 0.6, 0.8):
            for lo in (0.01, 0.02, 0.05, 0.1):
                for tz in (0.15, 0.25, 0.35):
                    for band in (None, (0.1, 0.5)):
                        def f(idx, S=S, zc=zc, has=has, zms=zms, hi=hi, lo=lo, tz=tz, band=band):
                            c0 = c224[idx]
                            esc = (c0 < hi) & (c0 >= lo)
                            z = zc[idx]
                            d = np.where(c0 >= hi, 1, np.where(c0 < lo, 0, (z >= tz).astype(int)))
                            cost = ms[R.index(224)] + np.where(esc & has[idx], zms, 0.0)
                            if band is not None:
                                again = esc & (z >= band[0]) & (z < band[1])
                                d = np.where(again, (conf[idx, k640] >= 0.25).astype(int), d)
                                cost = cost + np.where(again, ms[k640], 0.0)
                            return d, cost
                        out.append((f"ZoomCascade S={S} hi={hi} lo={lo} tz={tz} band={band}", f))
    return out


FAMILIES = {"Fixed": fam_fixed, "Cascade": fam_cascade, "Predictor (DRNet-style)": fam_predictor,
            "ACIES v2": fam_acies, "ACIES v2 (asym)": fam_acies_asym, "Context gate": fam_gate, "ACIES v3 (ctx)": fam_acies_ctx, "ACIES v2 + zoom": fam_acies_zoom, "Zoom cascade (hand)": fam_zoom_cascade}


def sig_violation(k, n, eps, level=0.05):
    """Is k bad events out of n significantly more than eps*n? (exact binomial, not sampling noise)"""
    from acies.risk import binom_cdf
    return n > 0 and (1.0 - binom_cdf(k - 1, n, eps)) < level


def one_trial(args):
    name, rep, targets, delta, mode = args
    R, ms, conf, Y = G["R"], G["ms"], G["conf"][name], G["Y"][name]
    n = len(Y)
    perm = np.random.default_rng(1000 * rep + 7).permutation(n)
    third = n // 3
    train, cert, test = perm[:third], perm[third:2 * third], perm[2 * third:3 * third]
    k640 = ref_index()
    ref = (conf[:, k640] >= 0.25).astype(int)
    ref_acc = float(((conf[test, k640] >= .25) == Y[test]).mean())
    res = []
    for fam, build in FAMILIES.items():
        settings = build(conf, ref, ms, R, train, name)
        dtr = [f(train) for _, f in settings]
        cost_tr = [float(c.mean()) for _, c in dtr]
        dce = [f(cert) for _, f in settings]
        cache = {}
        pos_tr, pos_ce, pos_te = ref[train] == 1, ref[cert] == 1, ref[test] == 1
        if mode == "disagree":
            risk_tr = [float((d != ref[train]).mean()) for d, _ in dtr]
            L = np.stack([(d != ref[cert]).astype(float) for d, _ in dce], 1).tolist()
        for tgt in targets:
            if mode == "disagree":
                eps = tgt
                order = sorted(range(len(settings)), key=lambda j: (risk_tr[j], cost_tr[j]))     # safest first
                cr = certify(L, alpha=eps, delta=delta, order=order)
            else:
                em, ef = tgt
                if pos_ce.sum() < 20 or (~pos_ce).sum() < 20:
                    cr = None
                else:
                    sc = [float((d[pos_tr] == 0).mean()) / em + float((d[~pos_tr] == 1).mean()) / ef for d, _ in dtr]
                    order = sorted(range(len(settings)), key=lambda j: (sc[j], cost_tr[j]))
                    pv = [max(risk_pvalue((d[pos_ce] == 0).astype(float).tolist(), em),
                              risk_pvalue((d[~pos_ce] == 1).astype(float).tolist(), ef)) for d, _ in dce]
                    cr = certify_pvalues(pv, delta=delta, order=order)
            best = cr.best(cost_tr) if cr is not None and cr.certified else None
            if best is None:                                  # nothing certified: fall back to the reference
                cost, dis, miss, fa, label, viol = float(ms[k640]), 0.0, 0.0, 0.0, "Reference (fallback)", False
                acc = ref_acc
            else:
                if best not in cache:
                    cache[best] = settings[best][1](test)
                d, c = cache[best]
                cost, dis = float(c.mean()), float((d != ref[test]).mean())
                miss = float((d[pos_te] == 0).mean()) if pos_te.any() else 0.0
                fa = float((d[~pos_te] == 1).mean())
                acc, label = float((d == Y[test]).mean()), settings[best][0]
                if mode == "disagree":
                    viol = sig_violation(int((d != ref[test]).sum()), len(test), tgt)
                else:
                    viol = (sig_violation(int((d[pos_te] == 0).sum()), int(pos_te.sum()), tgt[0])
                            or sig_violation(int((d[~pos_te] == 1).sum()), int((~pos_te).sum()), tgt[1]))
            res.append(dict(cls=name, rep=rep, family=fam, target=tgt, cost=cost, test_dis=dis, test_miss=miss,
                            test_fa=fa, test_acc=acc, ref_acc=ref_acc, violated=bool(viol),
                            certified=best is not None, setting=label, n_pos=int(pos_ce.sum())))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--clean-ms-from", default=None)
    ap.add_argument("--ref-res", type=int, default=None, help="reference resolution (default: the largest)")
    ap.add_argument("--classes", default="person:0,car:2,chair:56,bottle:39,cup:41,dog:16,tv:62,bench:13")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--mode", choices=["disagree", "conditional"], default="disagree")
    ap.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.02, 0.03],
                    help="disagree mode: tolerated disagreement with the reference")
    ap.add_argument("--miss-fa", type=float, nargs="+", default=[0.10, 0.03, 0.05, 0.02],
                    help="conditional mode: pairs (max miss rate, max false-alarm rate)")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--zoom", nargs="*", default=[], help="zoom shards (measure_zoom.py output)")
    ap.add_argument("--latency-json", default=None,
                    help="output of measure_zoom_latency.py: full-frame and crop costs measured in ONE session")
    ap.add_argument("--families", default=None, help="comma list; default all available")
    a = ap.parse_args()

    classes = {s.split(":")[0]: int(s.split(":")[1]) for s in a.classes.split(",")}
    R, ms, conf, Y = load(a.measure, a.subset, a.clean_ms_from, classes)
    if a.latency_json:
        lat = json.load(open(a.latency_json))
        ms = np.array([lat["full"][str(r)] for r in R])
        if "224" in lat:
            ZOOM_MS.update({224: lat["224"]["mean_of_medians"], 320: lat["320"]["mean_of_medians"]})
        print("costs (single session, ms): full-frame", {r: round(float(v), 1) for r, v in zip(R, ms)}, flush=True)
    G.update(R=R, ms=ms, conf=conf, Y=Y, ref_res=a.ref_res or max(R))
    if a.zoom:
        G["zoom"] = load_zoom(a.zoom, a.subset, classes, ZOOM_MS)
    else:
        for k in ("ACIES v2 + zoom", "Zoom cascade (hand)"):
            FAMILIES.pop(k, None)
    if a.families:
        keep = set(a.families.split(","))
        for k in list(FAMILIES):
            if k not in keep:
                FAMILIES.pop(k)
    G["ctx"] = load_ctx(a.measure, a.subset, classes, (160, 224)) if any(
        k in FAMILIES for k in ("ACIES v3 (ctx)", "Context gate")) else {}

    if a.mode == "disagree":
        targets = list(a.eps)
    else:
        targets = list(zip(a.miss_fa[0::2], a.miss_fa[1::2]))
    tasks = [(c, r, targets, a.delta, a.mode) for r in range(a.reps) for c in classes]
    print(f"{len(tasks)} trials ({len(classes)} classes x {a.reps} random splits), delta={a.delta}, mode={a.mode}", flush=True)
    with mp.get_context("fork").Pool(a.procs) as pool:
        rows = [r for chunk in pool.imap_unordered(one_trial, tasks) for r in chunk]
    if a.out:
        json.dump(rows, open(a.out, "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))

    ref_ms = ms[ref_index()]
    print(f"\nReference = full decision at {G['ref_res']} px: {ref_ms:.1f} ms | cost = mean latency on TEST | violation = significantly above the "
          f"tolerance (exact binomial test, 5%)\n")
    for tgt in targets:
        if a.mode == "disagree":
            head = f"epsilon = {tgt:.0%}: at most {tgt:.0%} of inputs may differ from the 640 px decision"
        else:
            head = f"miss rate <= {tgt[0]:.0%} of what the 640 px system flags AND false alarms <= {tgt[1]:.0%} of the rest"
        print(f"===== {head}   (delta = {a.delta:.0%}) =====")
        print(f"{'method':22s} {'cost ms':>8s} {'saving':>7s} {'miss':>6s} {'false-al.':>9s} {'disagree':>8s} {'violations':>12s} {'certified':>10s} {'acc vs GT':>9s}")
        for fam in FAMILIES:
            rs = [r for r in rows if r["family"] == fam and r["target"] == tgt]
            if not rs:
                continue
            cost = np.mean([r["cost"] for r in rs])
            print(f"{fam:22s} {cost:8.1f} {(1 - cost / ref_ms) * 100:6.0f}% {np.mean([r['test_miss'] for r in rs])*100:5.1f}% "
                  f"{np.mean([r['test_fa'] for r in rs])*100:8.2f}% {np.mean([r['test_dis'] for r in rs])*100:7.2f}% "
                  f"{np.mean([r['violated'] for r in rs])*100:5.1f}% ({sum(r['violated'] for r in rs):>2d}/{len(rs)}) "
                  f"{np.mean([r['certified'] for r in rs])*100:8.0f}%  {np.mean([r['test_acc'] for r in rs])*100:8.2f}%")
        print()


if __name__ == "__main__":
    main()
