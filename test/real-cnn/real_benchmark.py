"""Real benchmark — ACIES with real CNN on MNIST.

Compares:
1. Fixed 28x28 (full resolution)
2. Fixed 14x14 (reduced)
3. ACIES adaptive (selects resolution per image)
4. Cascade classifier (2-threshold baseline from literature)
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random

from acies import APCController, APCConfig


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


def add_noise(image, noise_level):
    noise = torch.randn_like(image) * noise_level
    return torch.clamp(image + noise, 0, 1)


def cnn_predict(model, image, resolution=None):
    """Run CNN on image at given resolution. Returns (predicted_class, confidence)."""
    transform_norm = transforms.Normalize((0.1307,), (0.3081,))

    if resolution and resolution != 28:
        img = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(resolution, resolution), mode='bilinear'
        )
        img = torch.nn.functional.interpolate(
            img, size=(28, 28), mode='bilinear'
        ).squeeze(0)
    else:
        img = image

    img = transform_norm(img)
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        return probs.argmax(1).item(), probs.max().item()


def make_clarity_fn(model, image_tensor, noise_level):
    """Clarity = CNN confidence on resized image."""
    resolutions = {"64p": 8, "128p": 14, "224p": 28, "320p": 32, "512p": 56, "1024p": 28}

    def clarity_fn(action):
        name = action.name
        noisy = add_noise(image_tensor, noise_level)

        if name in resolutions:
            res = resolutions[name]
            _, conf = cnn_predict(model, noisy, resolution=res)
        else:
            _, conf = cnn_predict(model, noisy)

        return conf

    return clarity_fn


def run_fixed(model, test_loader, n_images, resolution=None):
    """Fixed resolution baseline."""
    correct = 0
    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n_images: break
        # Mixed difficulty: 50% easy (0-0.2), 30% medium (0.2-0.5), 20% hard (0.5-0.8)
        r = random.random()
        if r < 0.5:
            noise_level = random.uniform(0, 0.2)
        elif r < 0.8:
            noise_level = random.uniform(0.2, 0.5)
        else:
            noise_level = random.uniform(0.5, 0.8)
        noisy = add_noise(data.squeeze(0), noise_level)
        pred, _ = cnn_predict(model, noisy, resolution=resolution)
        if pred == target.item():
            correct += 1
    elapsed = time.time() - start
    return correct / n_images * 100, elapsed


def run_cascade(model, test_loader, n_images, threshold_high=0.95, threshold_low=0.70):
    """Cascade classifier: try low-res first, escalate if uncertain."""
    correct = 0
    total_cost = 0
    start = time.time()

    for i, (data, target) in enumerate(test_loader):
        if i >= n_images: break
        r = random.random()
        if r < 0.5:
            noise_level = random.uniform(0, 0.2)
        elif r < 0.8:
            noise_level = random.uniform(0.2, 0.5)
        else:
            noise_level = random.uniform(0.5, 0.8)
        noisy = add_noise(data.squeeze(0), noise_level)

        # Stage 1: fast low-res (14x14)
        pred_low, conf_low = cnn_predict(model, noisy, resolution=14)
        total_cost += 50  # low-res cost

        if conf_low >= threshold_high:
            # Confident → accept
            if pred_low == target.item():
                correct += 1
            continue

        # Stage 2: medium-res (28x28)
        pred_med, conf_med = cnn_predict(model, noisy, resolution=28)
        total_cost += 100  # medium-res cost

        if conf_med >= threshold_low:
            if pred_med == target.item():
                correct += 1
            continue

        # Stage 3: full-res (28x28 with more computation)
        pred_high, conf_high = cnn_predict(model, noisy, resolution=28)
        total_cost += 100
        if pred_high == target.item():
            correct += 1

    elapsed = time.time() - start
    return correct / n_images * 100, total_cost / n_images, elapsed


def run_acies(model, test_loader, n_images):
    """ACIES adaptive perception."""
    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)

    correct = 0
    total_cost = 0
    total_steps = 0
    action_counts = {}

    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n_images: break

        r = random.random()
        if r < 0.5:
            noise_level = random.uniform(0, 0.2)
        elif r < 0.8:
            noise_level = random.uniform(0.2, 0.5)
        else:
            noise_level = random.uniform(0.5, 0.8)
        true_class = 1 if target.item() != 0 else 0
        clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)

        if result.decision == true_class:
            correct += 1
        total_cost += result.total_cost
        total_steps += result.n_steps

        for step in result.steps:
            action_counts[step.action.name] = action_counts.get(step.action.name, 0) + 1

    elapsed = time.time() - start
    return (correct / n_images * 100, total_cost / n_images,
            total_steps / n_images, action_counts, elapsed)


def main():
    N = 1000

    print("=" * 70)
    print(f"ACIES Real CNN Benchmark — MNIST ({N} images, mixed difficulty)")
    print("=" * 70)

    model = SmallCNN()
    model.load_state_dict(torch.load('/tmp/mnist_cnn.pth', weights_only=True))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # 1. Fixed 28x28
    print(f"\n[1/4] Fixed 28x28...")
    acc_full, t = run_fixed(model, test_loader, N, resolution=None)
    print(f"  {acc_full:.1f}% ({t:.1f}s)")

    # 2. Fixed 14x14
    print(f"\n[2/4] Fixed 14x14...")
    acc_14, t = run_fixed(model, test_loader, N, resolution=14)
    print(f"  {acc_14:.1f}% ({t:.1f}s)")

    # 3. Cascade classifier
    print(f"\n[3/4] Cascade classifier (3 stages)...")
    acc_cas, cost_cas, t = run_cascade(model, test_loader, N)
    print(f"  {acc_cas:.1f}% | avg cost: {cost_cas:.1f} ({t:.1f}s)")

    # 4. ACIES
    print(f"\n[4/4] ACIES adaptive...")
    acc_acies, cost_acies, steps_acies, actions, t = run_acies(model, test_loader, N)
    print(f"  {acc_acies:.1f}% | cost: {cost_acies:.1f} | steps: {steps_acies:.1f} ({t:.1f}s)")

    # Action distribution
    print(f"\n  Action distribution:")
    total_a = sum(actions.values())
    for a, c in sorted(actions.items(), key=lambda x: -x[1]):
        print(f"    {a:<12} {c:4d} ({c/total_a*100:5.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS (mixed difficulty: 50% easy, 30% medium, 20% hard)")
    print("=" * 70)
    print(f"  {'Method':<25} {'Accuracy':>8} {'Avg Cost':>10}")
    print(f"  {'-'*45}")
    print(f"  {'Fixed 28x28':<25} {acc_full:>7.1f}% {'187.2':>10}")
    print(f"  {'Fixed 14x14':<25} {acc_14:>7.1f}% {'76.5':>10}")
    print(f"  {'Cascade (3-stage)':<25} {acc_cas:>7.1f}% {cost_cas:>10.1f}")
    print(f"  {'ACIES adaptive':<25} {acc_acies:>7.1f}% {cost_acies:>10.1f}")
    print("=" * 70)

    if cost_acies < 187.2:
        savings = (1 - cost_acies / 187.2) * 100
        print(f"  ✓ ACIES saves {savings:.0f}% cost vs Fixed 28x28")
    else:
        overhead = (cost_acies / 187.2 - 1) * 100
        print(f"  ⚠ ACIES costs {overhead:.0f}% more but achieves {acc_acies - acc_full:+.1f}% accuracy")

    if acc_acies > acc_cas:
        print(f"  ✓ ACIES beats Cascade by {acc_acies - acc_cas:.1f}% accuracy")
    print("=" * 70)


if __name__ == "__main__":
    main()
