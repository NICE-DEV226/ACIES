#!/usr/bin/env python3
"""
ACIES Example — Simulate perception on a folder of images

Usage:
    python3 examples/image_folder.py

This demonstrates how ACIES would work with a real perception pipeline.
In production, you'd replace the clarity_fn with actual model inference.
"""

import sys
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acies import APCController, APCConfig, HardwareProfile, build_standard_actions


def simulate_image_classification(image_path: str, true_class: int) -> dict:
    """
    Simulate image classification with ACIES.

    In production, this would:
    1. Load the image
    2. Define clarity_fn based on your model's actual performance
    3. Run APC to decide which resolution/crop to use
    4. Return the decision
    """
    # Simulated clarity values (in production, these come from your model)
    clarities = {
        "64p": 0.55, "128p": 0.65, "224p": 0.75,
        "320p": 0.82, "512p": 0.88, "1024p": 0.93,
        "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
    }

    def clarity_fn(action):
        base = clarities.get(action.name, 0.5)
        noise = random.gauss(0, 0.03)
        return max(0.01, min(0.99, base + noise))

    # Run APC
    apc = APCController(APCConfig(
        confidence_threshold=0.92,
        max_steps=6,
        hardware=HardwareProfile.jetson_orin(),
    ))
    result = apc.run(true_class=true_class, clarity_fn=clarity_fn)

    return {
        "image": os.path.basename(image_path),
        "decision": result.decision,
        "correct": result.correct,
        "cost": result.total_cost,
        "steps": result.n_steps,
        "actions": result.actions_taken,
        "confidence": round(max(result.final_belief, 1 - result.final_belief), 3),
    }


def main():
    print("=" * 60)
    print("ACIES — Image Classification Simulation")
    print("=" * 60)
    print()

    # Simulate 10 images
    n_images = 10
    results = []

    start = time.time()
    for i in range(n_images):
        image_path = f"image_{i:04d}.jpg"
        true_class = random.randint(0, 1)
        result = simulate_image_classification(image_path, true_class)
        results.append(result)

        status = "✓" if result["correct"] else "✗"
        print(f"  {status} {result['image']}: "
              f"class={result['decision']} (true={true_class}) | "
              f"cost={result['cost']:.1f} | "
              f"steps={result['steps']} | "
              f"actions={' → '.join(result['actions'])}")

    elapsed = time.time() - start

    # Summary
    correct = sum(1 for r in results if r["correct"])
    avg_cost = sum(r["cost"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    print()
    print("─" * 60)
    print(f"Results: {correct}/{n_images} correct ({correct/n_images*100:.0f}%)")
    print(f"Avg cost: {avg_cost:.1f} | Avg steps: {avg_steps:.1f}")
    print(f"Total time: {elapsed*1000:.0f}ms ({n_images/elapsed:.0f} images/sec)")
    print()

    # Show action distribution
    all_actions = []
    for r in results:
        all_actions.extend(r["actions"])
    action_counts = {}
    for a in all_actions:
        action_counts[a] = action_counts.get(a, 0) + 1

    print("Action distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / len(results) * 20)
        print(f"  {action:<12} {count:>3} ({count/len(results)*100:4.0f}%) {bar}")


if __name__ == "__main__":
    main()
