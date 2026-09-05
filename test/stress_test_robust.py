"""
ACIES Stress Test 4 — With Robustness Metrics

Tests the new mechanisms: abstention, cost budget, degradation detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_model():
    model = SmallCNN()
    model.load_state_dict(torch.load('/tmp/mnist_cnn.pth', weights_only=True))
    model.eval()
    return model


# ============================================================
# Distortion functions
# ============================================================
def add_gaussian(image, sigma):
    return torch.clamp(image + torch.randn_like(image) * sigma, 0, 1)

def add_blur(image, kernel_size):
    if kernel_size < 3: return image
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    blurred = F.conv2d(image.unsqueeze(0), kernel, padding=padding)
    return blurred.squeeze(0)

def add_rotation(image, angle_deg):
    angle = torch.tensor([angle_deg])
    cos_a = torch.cos(angle * 3.14159 / 180)
    sin_a = torch.sin(angle * 3.14159 / 180)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]]).unsqueeze(0)
    grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
    return F.grid_sample(image.unsqueeze(0), grid, align_corners=False).squeeze(0)

def add_occlusion(image, ratio):
    img = image.clone()
    h, w = img.shape[1], img.shape[2]
    occ_h = int(h * ratio)
    occ_w = int(w * ratio)
    y = random.randint(0, h - occ_h)
    x = random.randint(0, w - occ_w)
    img[:, y:y+occ_h, x:x+occ_w] = 0
    return img


def cnn_confidence(model, image):
    transform_norm = transforms.Normalize((0.1307,), (0.3081,))
    img = transform_norm(image)
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        return probs.argmax(1).item(), probs.max().item()


def make_clarity_fn(model, image, distortion_fn):
    def clarity_fn(action):
        distorted = distortion_fn(image.clone())
        _, conf = cnn_confidence(model, distorted)
        return conf
    return clarity_fn


# ============================================================
# Test runner with new metrics
# ============================================================
def run_test(name, model, test_loader, distortion_fn, n=300, config=None):
    if config is None:
        config = APCConfig(
            confidence_threshold=0.92,
            max_cost_per_image=600.0,
            degradation_patience=3,
            abstention_confidence=0.6,
        )
    apc = APCController(config)

    correct = 0
    total_cost = 0
    total_steps = 0
    n_abstained = 0
    n_degraded = 0
    n_budget_exceeded = 0
    total_avg_clarity = 0.0

    for i, (data, target) in enumerate(test_loader):
        if i >= n: break
        true_class = 1 if target.item() != 0 else 0
        clarity_fn = make_clarity_fn(model, data.squeeze(0), distortion_fn)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)

        if result.abstained:
            n_abstained += 1
        elif result.correct:
            correct += 1

        if result.degraded:
            n_degraded += 1
        if result.cost_budget_exceeded:
            n_budget_exceeded += 1

        total_cost += result.total_cost
        total_steps += result.n_steps
        total_avg_clarity += result.avg_clarity

    n_decided = n - n_abstained
    acc_of_decided = (correct / n_decided * 100) if n_decided > 0 else 0
    acc_overall = (correct / n * 100)
    cost = total_cost / n
    steps = total_steps / n
    avg_clarity = total_avg_clarity / n

    print(f"  {name:<35} "
          f"acc={acc_overall:5.1f}% "
          f"(decided={acc_of_decided:4.1f}%) "
          f"abstain={n_abstained:3d} "
          f"degraded={n_degraded:2d} "
          f"budget={n_budget_exceeded:2d} "
          f"cost={cost:7.1f} "
          f"steps={steps:.1f} "
          f"clarity={avg_clarity:.2f}")

    return {
        "name": name, "acc": acc_overall, "acc_decided": acc_of_decided,
        "abstained": n_abstained, "degraded": n_degraded,
        "budget_exceeded": n_budget_exceeded, "cost": cost, "steps": steps,
    }


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("╔" + "═" * 78 + "╗")
    print("║  ACIES STRESS TEST 4 — Robustness Metrics                                     ║")
    print("╚" + "═" * 78 + "╝")
    print()
    print(f"  {'Name':<35} {'Acc':>5} {'Decided':>8} {'Abst':>5} {'Deg':>4} {'Bud':>4} {'Cost':>7} {'Steps':>5} {'Clar':>5}")
    print("  " + "-" * 110)

    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    random.seed(42)

    # Config robuste
    config = APCConfig(
        confidence_threshold=0.92,
        max_cost_per_image=600.0,
        degradation_patience=3,
        abstention_confidence=0.6,
    )

    # Baseline clean
    run_test("CLEAN (baseline)", model, test_loader, lambda img: img, n=500, config=config)

    # Noise
    print("\n── Noise ──")
    for sigma in [0.3, 0.6, 1.0, 2.0]:
        run_test(f"noise sigma={sigma}", model, test_loader,
                 lambda img, s=sigma: add_gaussian(img, s), n=300, config=config)

    # Blur (was breaking before)
    print("\n── Blur (was breaking) ──")
    for ks in [3, 5, 7, 11]:
        run_test(f"blur kernel={ks}", model, test_loader,
                 lambda img, k=ks: add_blur(img, k), n=300, config=config)

    # Rotation
    print("\n── Rotation ──")
    for angle in [15, 30, 45, 60]:
        run_test(f"rotation {angle}°", model, test_loader,
                 lambda img, a=angle: add_rotation(img, a), n=300, config=config)

    # Occlusion
    print("\n── Occlusion ──")
    for ratio in [0.2, 0.3, 0.4, 0.5]:
        run_test(f"occlusion {ratio*100:.0f}%", model, test_loader,
                 lambda img, r=ratio: add_occlusion(img, r), n=300, config=config)

    # Combined
    print("\n── Combined ──")
    run_test("blur5 + noise0.5", model, test_loader,
             lambda img: add_gaussian(add_blur(img, 5), 0.5), n=300, config=config)
    run_test("noise0.5 + rot30°", model, test_loader,
             lambda img: add_rotation(add_gaussian(img, 0.5), 30), n=300, config=config)

    # Budget sweep
    print("\n── Budget Sweep ──")
    for budget in [200, 400, 600, 1000]:
        cfg = APCConfig(confidence_threshold=0.92, max_cost_per_image=budget,
                        degradation_patience=3, abstention_confidence=0.6)
        run_test(f"budget={budget}", model, test_loader,
                 lambda img: add_gaussian(add_blur(img, 5), 0.5), n=200, config=cfg)

    print()
    print("=" * 78)
    print("KEY: acc=overall accuracy, decided=accuracy on non-abstained,")
    print("     abstain=images skipped, degraded=blur/noise detected, budget=budget exceeded")
    print("=" * 78)


if __name__ == "__main__":
    main()
