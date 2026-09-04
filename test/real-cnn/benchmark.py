"""ACIES benchmark — compare with fixed resolution baselines."""

import random
import time
from acies import APCController, APCConfig, HardwareProfile

def make_clarity_fn(difficulty):
    """Higher difficulty = need higher resolution to get good clarity."""
    def clarity_fn(action):
        name = action.name
        if name == "1024p":
            return max(0.5, 1.0 - difficulty * 0.3)
        elif name == "512p":
            return max(0.4, 0.85 - difficulty * 0.2)
        elif name == "320p":
            return max(0.3, 0.75 - difficulty * 0.25)
        elif name == "224p":
            return max(0.2, 0.65 - difficulty * 0.3)
        elif name == "128p":
            return max(0.15, 0.55 - difficulty * 0.35)
        elif name == "64p":
            return max(0.1, 0.45 - difficulty * 0.4)
        elif "crop" in name:
            return max(0.3, 0.7 - difficulty * 0.2)
        return 0.5
    return clarity_fn

def run_apc(n=1000):
    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)
    total_cost = 0
    correct = 0
    for _ in range(n):
        true_class = random.choice([0, 1])
        difficulty = random.random()
        clarity_fn = make_clarity_fn(difficulty)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
        total_cost += result.total_cost
        if result.decision == true_class:
            correct += 1
    return correct / n, total_cost / n

def run_fixed(n=1000, clarity_level=0.93):
    correct = 0
    total_cost = 0
    for _ in range(n):
        true_class = random.choice([0, 1])
        if random.random() < clarity_level:
            predicted = true_class
        else:
            predicted = 1 - true_class
        if predicted == true_class:
            correct += 1
        total_cost += 100.0
    return correct / n, total_cost / n

def main():
    n = 2000
    print(f"ACIES Benchmark ({n} images)")
    print("-" * 50)

    start = time.time()
    ac_acc, ac_cost = run_apc(n)
    ac_time = time.time() - start

    start = time.time()
    f1_acc, f1_cost = run_fixed(n, 0.93)
    f1_time = time.time() - start

    start = time.time()
    f2_acc, f2_cost = run_fixed(n, 0.75)
    f2_time = time.time() - start

    print(f"{'Method':<15} {'Accuracy':>8} {'Cost':>8} {'Time':>8}")
    print("-" * 50)
    print(f"{'Fixed 1024p':<15} {f1_acc*100:>7.1f}% {f1_cost:>7.1f} {f1_time:>7.2f}s")
    print(f"{'Fixed 224p':<15} {f2_acc*100:>7.1f}% {f2_cost:>7.1f} {f2_time:>7.2f}s")
    print(f"{'ACIES':<15} {ac_acc*100:>7.1f}% {ac_cost:>7.1f} {ac_time:>7.2f}s")
    print("-" * 50)
    savings = (1 - ac_cost / f1_cost) * 100
    print(f"Savings vs 1024p: {savings:.0f}%")

if __name__ == "__main__":
    main()
