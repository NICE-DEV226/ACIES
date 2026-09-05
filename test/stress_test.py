"""
ACIES Stress Test — Find the limits.

Systematically tests:
1. Clean MNIST (baseline)
2. Progressive noise (find breaking point)
3. Ambiguous images
4. Scale (10k+ images)
5. Confidence threshold sweep
6. Hardware profile extremes
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random
import json

from acies import APCController, APCConfig, HardwareProfile


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_model():
    model = SmallCNN()
    model.load_state_dict(torch.load('/tmp/mnist_cnn.pth', weights_only=True))
    model.eval()
    return model


def add_noise(image, level):
    noise = torch.randn_like(image) * level
    return torch.clamp(image + noise, 0, 1)


def cnn_confidence(model, image, resolution=None):
    transform_norm = transforms.Normalize((0.1307,), (0.3081,))
    if resolution and resolution != 28:
        img = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(resolution, resolution), mode='bilinear'
        )
        img = torch.nn.functional.interpolate(img, size=(28, 28), mode='bilinear').squeeze(0)
    else:
        img = image
    img = transform_norm(img)
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        return probs.argmax(1).item(), probs.max().item()


def make_clarity_fn(model, image_tensor, noise_level):
    resolutions = {"64p": 8, "128p": 14, "224p": 28, "320p": 32, "512p": 56, "1024p": 28}
    def clarity_fn(action):
        noisy = add_noise(image_tensor, noise_level)
        if action.name in resolutions:
            _, conf = cnn_confidence(model, noisy, resolution=resolutions[action.name])
        else:
            _, conf = cnn_confidence(model, noisy)
        return conf
    return clarity_fn


# ============================================================
# Test 1: Clean MNIST (no noise)
# ============================================================
def test_clean(model, test_loader, n=500):
    print("=" * 60)
    print("TEST 1: Clean MNIST (no noise)")
    print("=" * 60)

    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)

    correct = 0
    total_cost = 0
    total_steps = 0
    actions_used = set()

    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n: break
        true_class = 1 if target.item() != 0 else 0
        clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level=0.0)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
        if result.decision == true_class:
            correct += 1
        total_cost += result.total_cost
        total_steps += result.n_steps
        for s in result.steps:
            actions_used.add(s.action.name)

    elapsed = time.time() - start
    acc = correct / n * 100
    cost = total_cost / n
    steps = total_steps / n

    print(f"  Accuracy:  {acc:.1f}%")
    print(f"  Avg cost:  {cost:.1f}")
    print(f"  Avg steps: {steps:.1f}")
    print(f"  Actions:   {len(actions_used)}/9 ({', '.join(sorted(actions_used))})")
    print(f"  Time:      {elapsed:.1f}s ({n/elapsed:.0f} img/s)")
    print()

    return {"acc": acc, "cost": cost, "steps": steps}


# ============================================================
# Test 2: Progressive noise (find breaking point)
# ============================================================
def test_noise_sweep(model, test_loader, n_per_level=200):
    print("=" * 60)
    print("TEST 2: Progressive noise (find breaking point)")
    print("=" * 60)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []

    for noise in noise_levels:
        config = APCConfig(confidence_threshold=0.92)
        apc = APCController(config)

        correct = 0
        total_cost = 0

        for i, (data, target) in enumerate(test_loader):
            if i >= n_per_level: break
            true_class = 1 if target.item() != 0 else 0
            clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level=noise)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost

        acc = correct / n_per_level * 100
        cost = total_cost / n_per_level
        results.append({"noise": noise, "acc": acc, "cost": cost})
        print(f"  noise={noise:.1f}  acc={acc:5.1f}%  cost={cost:7.1f}")

    # Find breaking point
    print()
    for r in results:
        if r["acc"] < 90:
            print(f"  ⚠ BREAKING POINT: noise={r['noise']:.1f} → {r['acc']:.1f}% accuracy")
            break
    else:
        print(f"  ✓ No breaking point found (all above 90%)")

    print()
    return results


# ============================================================
# Test 3: Confidence threshold sweep
# ============================================================
def test_threshold_sweep(model, test_loader, n=300):
    print("=" * 60)
    print("TEST 3: Confidence threshold sweep")
    print("=" * 60)

    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
    results = []

    for thr in thresholds:
        config = APCConfig(confidence_threshold=thr)
        apc = APCController(config)

        correct = 0
        total_cost = 0
        total_steps = 0

        for i, (data, target) in enumerate(test_loader):
            if i >= n: break
            true_class = 1 if target.item() != 0 else 0
            noise = random.uniform(0, 0.5)
            clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level=noise)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost
            total_steps += result.n_steps

        acc = correct / n * 100
        cost = total_cost / n
        steps = total_steps / n
        results.append({"thr": thr, "acc": acc, "cost": cost, "steps": steps})
        print(f"  thr={thr:.2f}  acc={acc:5.1f}%  cost={cost:7.1f}  steps={steps:.1f}")

    print()
    return results


# ============================================================
# Test 4: Scale test (10k images)
# ============================================================
def test_scale(model, test_loader, n=5000):
    print("=" * 60)
    print(f"TEST 4: Scale test ({n} images)")
    print("=" * 60)

    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)

    correct = 0
    total_cost = 0
    action_counts = {}
    batch_times = []

    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n: break
        true_class = 1 if target.item() != 0 else 0
        noise = random.uniform(0, 0.5)
        clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level=noise)

        t0 = time.time()
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
        batch_times.append(time.time() - t0)

        if result.decision == true_class:
            correct += 1
        total_cost += result.total_cost
        for s in result.steps:
            action_counts[s.action.name] = action_counts.get(s.action.name, 0) + 1

    elapsed = time.time() - start
    acc = correct / n * 100
    cost = total_cost / n

    print(f"  Accuracy:  {acc:.1f}%")
    print(f"  Avg cost:  {cost:.1f}")
    print(f"  Total:     {elapsed:.1f}s ({n/elapsed:.0f} img/s)")
    print(f"  Per-image: {sum(batch_times)/len(batch_times)*1000:.1f}ms avg")
    print(f"  Actions:   {dict(sorted(action_counts.items(), key=lambda x: -x[1])[:5])}")
    print()

    return {"acc": acc, "cost": cost, "speed": n/elapsed}


# ============================================================
# Test 5: Hardware profile extremes
# ============================================================
def test_hardware_extremes(model, test_loader, n=200):
    print("=" * 60)
    print("TEST 5: Hardware profile extremes")
    print("=" * 60)

    profiles = {
        "edge_tpu": HardwareProfile.edge_tpu(),
        "rpi5": HardwareProfile.raspberry_pi5(),
        "default": HardwareProfile.default(),
        "gpu": HardwareProfile.desktop_gpu(),
        "jetson": HardwareProfile.jetson_orin(),
    }

    for name, profile in profiles.items():
        config = APCConfig(confidence_threshold=0.92, hardware=profile)
        apc = APCController(config)

        correct = 0
        total_cost = 0
        for i, (data, target) in enumerate(test_loader):
            if i >= n: break
            true_class = 1 if target.item() != 0 else 0
            noise = random.uniform(0, 0.5)
            clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level=noise)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost

        acc = correct / n * 100
        cost = total_cost / n
        print(f"  {name:<12}  acc={acc:5.1f}%  cost={cost:7.1f}")

    print()


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  ACIES STRESS TEST — Find the limits                    ║")
    print("╚" + "═" * 58 + "╝")
    print()

    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    random.seed(42)

    # Run all tests
    test_clean(model, test_loader, n=500)
    test_noise_sweep(model, test_loader, n_per_level=200)
    test_threshold_sweep(model, test_loader, n=300)
    test_scale(model, test_loader, n=5000)
    test_hardware_extremes(model, test_loader, n=200)

    print("=" * 60)
    print("DONE — All stress tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
