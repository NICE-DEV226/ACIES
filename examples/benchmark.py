#!/usr/bin/env python3
"""
ACIES Benchmark — APC vs Baselines

Compares:
  1. APC (adaptive perception control)
  2. Fixed 1024p (always maximum resolution)
  3. Fixed 224p (always medium resolution)
  4. Random action selection
  5. Information-gain greedy (no Thompson Sampling)

Generates a text report with metrics.
"""

import sys
import os
import time
import random
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acies import (
    APCController, APCConfig, HardwareProfile,
    BeliefState, ClarityLearner, SafetyLayer, SafetyConfig,
    build_standard_actions,
)


# ============================================================
# Baselines
# ============================================================

def baseline_fixed(actions, hardware, true_class, clarity_fn, resolution_name):
    """Always uses the same resolution."""
    action = next(a for a in actions if a.name == resolution_name)
    belief = BeliefState(prior=0.5)
    total_cost = 0.0
    steps = 0

    for _ in range(6):
        clarity = clarity_fn(action)
        cost = action.cost(hardware)

        if true_class == 1:
            obs = 1 if random.random() < clarity else 0
        else:
            obs = 0 if random.random() < clarity else 1

        belief.update(obs, clarity)
        total_cost += cost
        steps += 1

        if belief.confidence >= 0.95:
            break

    return {
        "decision": belief.decision,
        "correct": belief.decision == true_class,
        "cost": total_cost,
        "steps": steps,
    }


def baseline_random(actions, hardware, true_class, clarity_fn):
    """Random action selection."""
    belief = BeliefState(prior=0.5)
    total_cost = 0.0
    steps = 0

    for _ in range(6):
        action = random.choice(actions)
        clarity = clarity_fn(action)
        cost = action.cost(hardware)

        if true_class == 1:
            obs = 1 if random.random() < clarity else 0
        else:
            obs = 0 if random.random() < clarity else 1

        belief.update(obs, clarity)
        total_cost += cost
        steps += 1

        if belief.confidence >= 0.95:
            break

    return {
        "decision": belief.decision,
        "correct": belief.decision == true_class,
        "cost": total_cost,
        "steps": steps,
    }


def baseline_infogain(actions, hardware, true_class, clarity_fn):
    """Greedy information gain (no Thompson Sampling)."""
    belief = BeliefState(prior=0.5)
    total_cost = 0.0
    steps = 0

    for _ in range(6):
        best_action = None
        best_score = -1

        for action in actions:
            clarity = 0.5 + 0.49 * action.pixel_ratio  # fixed estimate
            cost = action.cost(hardware)
            score = belief.delta_risk_efficiency(clarity, cost)
            if score > best_score:
                best_score = score
                best_action = action

        clarity = clarity_fn(best_action)
        cost = best_action.cost(hardware)

        if true_class == 1:
            obs = 1 if random.random() < clarity else 0
        else:
            obs = 0 if random.random() < clarity else 1

        belief.update(obs, clarity)
        total_cost += cost
        steps += 1

        if belief.confidence >= 0.95:
            break

    return {
        "decision": belief.decision,
        "correct": belief.decision == true_class,
        "cost": total_cost,
        "steps": steps,
    }


# ============================================================
# Benchmark
# ============================================================

def make_clarities(difficulty=0.1):
    base = {
        "64p": 0.55, "128p": 0.65, "224p": 0.75,
        "320p": 0.82, "512p": 0.88, "1024p": 0.93,
        "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
    }
    def clarity_fn(action):
        b = base.get(action.name, 0.5) * (1.0 - difficulty * 0.4)
        b += random.gauss(0, 0.03)
        return max(0.01, min(0.99, b))
    return clarity_fn


def run_benchmark(n=1000, hardware_name="default", difficulty=0.1):
    hardware = HardwareProfile.__dict__[hardware_name]()
    actions = build_standard_actions()
    clarity_fn = make_clarities(difficulty)

    apc = APCController(APCConfig(
        confidence_threshold=0.95,
        max_steps=6,
        hardware=hardware,
    ))

    def apc_fn(tc, cf):
        r = apc.run(tc, cf)
        return {"correct": r.correct, "cost": r.total_cost, "steps": r.n_steps}

    methods = {
        "APC": apc_fn,
        "Fixed 1024p": lambda tc, cf: baseline_fixed(actions, hardware, tc, cf, "1024p"),
        "Fixed 224p": lambda tc, cf: baseline_fixed(actions, hardware, tc, cf, "224p"),
        "Random": lambda tc, cf: baseline_random(actions, hardware, tc, cf),
        "Info-Gain": lambda tc, cf: baseline_infogain(actions, hardware, tc, cf),
    }

    results = {name: [] for name in methods}

    start = time.time()
    for _ in range(n):
        true_class = random.randint(0, 1)
        cf = make_clarities(difficulty)
        for name, fn in methods.items():
            r = fn(true_class, cf)
            results[name].append(r)
    elapsed = time.time() - start

    return results, elapsed, hardware_name


def format_table(results, hardware_name):
    lines = []
    lines.append(f"{'Method':<15} {'Accuracy':>8} {'Avg Cost':>9} {'Avg Steps':>10} {'EPC':>8} {'Violations':>11}")
    lines.append("─" * 65)

    for name, runs in results.items():
        correct = sum(1 for r in runs if r.get("correct", False))
        acc = correct / len(runs)
        cost = sum(r.get("cost", r.get("total_cost", 0)) for r in runs) / len(runs)
        n_steps = sum(r.get("steps", r.get("n_steps", 0)) if isinstance(r.get("steps", r.get("n_steps", 0)), int) else len(r.get("steps", [])) for r in runs)
        steps = n_steps / len(runs)
        epc = cost / max(acc, 1e-10)

        lines.append(f"{name:<15} {acc*100:>7.1f}% {cost:>9.1f} {steps:>10.1f} {epc:>8.1f}")

    return "\n".join(lines)


def main():
    print()
    print("╔" + "═" * 63 + "╗")
    print("║  ACIES Benchmark — APC vs Baselines                            ║")
    print("╚" + "═" * 63 + "╝")
    print()

    n = 2000
    profiles = ["default", "jetson_orin", "raspberry_pi5", "desktop_gpu", "edge_tpu"]

    all_reports = []

    for profile in profiles:
        print(f"Running benchmark on {profile}... ", end="", flush=True)
        results, elapsed, _ = run_benchmark(n=n, hardware_name=profile)
        report = format_table(results, profile)
        all_reports.append((profile, report, elapsed))
        print(f"{elapsed:.1f}s")

    # Print all reports
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    for profile, report, elapsed in all_reports:
        print(f"\n┌─ Hardware: {profile.upper()} ─ ({elapsed:.1f}s for {n} iterations)")
        for line in report.split("\n"):
            print(f"│ {line}")
        print("└" + "─" * 60)

    # Savings summary
    print()
    print("=" * 70)
    print("SAVINGS vs Fixed 1024p")
    print("=" * 70)
    print(f"{'Profile':<12} {'APC Acc':>8} {'1024p Acc':>9} {'APC Cost':>9} {'1024p Cost':>10} {'Savings':>8}")
    print("─" * 60)

    for profile, report, _ in all_reports:
        lines = report.split("\n")
        apc_line = [l for l in lines if l.startswith("APC")]
        fixed_line = [l for l in lines if l.startswith("Fixed 1024p")]
        if apc_line and fixed_line:
            apc_parts = apc_line[0].split()
            fixed_parts = fixed_line[0].split()
            apc_acc = apc_parts[1]
            apc_cost = apc_parts[2]
            fixed_acc = fixed_parts[1]
            fixed_cost = fixed_parts[2]
            try:
                savings = (1 - float(apc_cost) / float(fixed_cost)) * 100
                print(f"{profile:<12} {apc_acc:>8} {fixed_acc:>9} {apc_cost:>9} {fixed_cost:>10} {savings:>+7.0f}%")
            except:
                print(f"{profile:<12} {apc_acc:>8} {fixed_acc:>9} {apc_cost:>9} {fixed_cost:>10}")

    print()


if __name__ == "__main__":
    random.seed(42)
    main()
