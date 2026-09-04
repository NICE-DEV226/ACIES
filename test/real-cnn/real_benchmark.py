"""Real benchmark — ACIES with noisy images to force adaptation."""

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
    """Add Gaussian noise to simulate difficult images."""
    noise = torch.randn_like(image) * noise_level
    return torch.clamp(image + noise, 0, 1)


def make_clarity_fn(model, image_tensor, noise_level):
    """Real clarity = CNN confidence on resized image."""
    transform_norm = transforms.Normalize((0.1307,), (0.3081,))
    resolutions = {"64p": 8, "128p": 14, "224p": 28, "320p": 32, "512p": 56, "1024p": 28}

    def clarity_fn(action):
        name = action.name
        noisy = add_noise(image_tensor, noise_level)

        if name in resolutions:
            res = resolutions[name]
            img = torch.nn.functional.interpolate(
                noisy.unsqueeze(0), size=(res, res), mode='bilinear'
            )
            img = torch.nn.functional.interpolate(
                img, size=(28, 28), mode='bilinear'
            ).squeeze(0)
        else:
            img = noisy

        img = transform_norm(img)
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
            return probs.max().item()

    return clarity_fn


def main():
    print("=" * 65)
    print("ACIES Real Benchmark — MNIST + Noisy Images (Real CNN)")
    print("=" * 65)

    model = SmallCNN()
    model.load_state_dict(torch.load('/tmp/mnist_cnn.pth', weights_only=True))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    n_images = 300

    # ── Fixed 28x28 on noisy images ──
    print("\n[1/3] Fixed 28x28 (noisy images)...")
    correct = 0
    start = time.time()
    transform_norm = transforms.Normalize((0.1307,), (0.3081,))
    for i, (data, target) in enumerate(test_loader):
        if i >= n_images: break
        noise_level = random.uniform(0, 0.8)
        noisy = add_noise(data, noise_level)
        with torch.no_grad():
            output = model(transform_norm(noisy.squeeze(0)).unsqueeze(0))
            if output.argmax(1).item() == target.item():
                correct += 1
    fixed_acc = correct / n_images * 100
    fixed_time = time.time() - start
    print(f"  Accuracy: {fixed_acc:.1f}% | Time: {fixed_time:.2f}s")

    # ── ACIES adaptive ──
    print(f"\n[2/3] ACIES adaptive (real CNN, noisy images)...")
    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)

    correct = 0
    total_cost = 0
    total_steps = 0
    action_counts = {}

    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n_images: break

        noise_level = random.uniform(0, 0.8)
        true_class = 1 if target.item() != 0 else 0
        clarity_fn = make_clarity_fn(model, data.squeeze(0), noise_level)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)

        if result.decision == true_class:
            correct += 1
        total_cost += result.total_cost
        total_steps += result.n_steps

        # Track which actions were chosen
        for step in result.steps:
            action_name = step.action.name
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

    acies_acc = correct / n_images * 100
    acies_cost = total_cost / n_images
    acies_steps = total_steps / n_images
    acies_time = time.time() - start

    print(f"  Accuracy: {acies_acc:.1f}% | Cost: {acies_cost:.1f} | Steps: {acies_steps:.1f}")
    print(f"  Time: {acies_time:.2f}s")

    # ── Action distribution ──
    print(f"\n[3/3] Action distribution:")
    total_actions = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_actions * 100
        print(f"  {action:<12} {count:4d} ({pct:5.1f}%)")

    # ── Summary ──
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Fixed 28x28:    {fixed_acc:.1f}%")
    print(f"  ACIES adaptive: {acies_acc:.1f}%")
    print(f"  Accuracy diff:  {fixed_acc - acies_acc:+.1f}%")
    print(f"  Avg cost:       {acies_cost:.1f} / 187.2 (full)")
    savings = (1 - acies_cost / 187.2) * 100
    print(f"  Cost savings:   {savings:.0f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
