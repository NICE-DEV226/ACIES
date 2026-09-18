#!/usr/bin/env python3
"""
ACIES vs the exact optimum

Compares APCController with the optimal cost/error frontier (acies.optimal) under the
same generative model, so "how good is the controller?" has a number, not an opinion.

    python3 examples/optimal_gap.py [--runs 4000] [--seed 1]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acies import APCConfig, APCController, HardwareProfile, build_standard_actions
from acies.optimal import min_cost_for_error

CLARITIES = {
    "64p": 0.55, "128p": 0.65, "224p": 0.75, "320p": 0.82, "512p": 0.88,
    "1024p": 0.93, "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()

    hw = HardwareProfile.default()
    actions = build_standard_actions()
    fixed_1024 = next(a for a in actions if a.name == "1024p").cost(hw)

    print(f"{'threshold':>9} | {'ACIES err':>9} {'cost':>7} | {'optimal cost @ same err':>24} | {'gap':>6}")
    print("-" * 70)
    for thr in (0.90, 0.95, 0.99):
        random.seed(args.seed)
        apc = APCController(APCConfig(confidence_threshold=thr, max_steps=args.horizon, hardware=hw))
        res = [apc.run(random.randint(0, 1), lambda a: CLARITIES[a.name]) for _ in range(args.runs)]
        err = 1 - sum(r.correct for r in res) / len(res)
        cost = sum(r.total_cost for r in res) / len(res)
        opt = min_cost_for_error(actions, CLARITIES, hw, target_error=max(err, 1e-3), horizon=args.horizon)
        print(f"{thr:>9.2f} | {err:>9.4f} {cost:>7.1f} | {opt.cost:>24.1f} | {cost / max(opt.cost, 1e-9):>5.1f}x")
    print(f"\n(one 1024p observation costs {fixed_1024:.1f}; gap = ACIES cost / optimal cost at equal error)")


if __name__ == "__main__":
    main()
