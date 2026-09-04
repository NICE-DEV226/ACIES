"""ACIES demo — adaptive perception on a simulated image stream."""

import random
from acies import APCController, APCConfig, HardwareProfile

CLARITIES = {
    "64p": 0.55, "128p": 0.65, "224p": 0.75,
    "320p": 0.82, "512p": 0.88, "1024p": 0.93,
    "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
}

def clarity_fn(action):
    return CLARITIES.get(action.name, 0.5)

def main():
    print("=" * 60)
    print("ACIES Demo — Adaptive Perception Control")
    print("=" * 60)

    config = APCConfig(
        confidence_threshold=0.92,
        hardware=HardwareProfile.jetson_orin(),
    )
    apc = APCController(config)

    total_cost = 0
    correct = 0
    n_images = 10

    for i in range(n_images):
        true_class = random.choice([0, 1])
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)

        total_cost += result.total_cost
        if result.decision == true_class:
            correct += 1

        status = "OK" if result.decision == true_class else "MISS"
        print(f"  Image {i+1:2d} | {status} | "
              f"decision={result.decision} | "
              f"belief={result.final_belief:.3f} | "
              f"cost={result.total_cost:7.1f} | "
              f"steps={result.n_steps}")

    accuracy = correct / n_images * 100
    avg_cost = total_cost / n_images

    print("=" * 60)
    print(f"  Accuracy: {accuracy:.0f}% | Avg cost: {avg_cost:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
