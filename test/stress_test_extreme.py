"""
ACIES Stress Test 3 — Extreme distortions on MNIST

Finds real breaking points.
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


def add_salt_pepper(image, prob):
    flat = image.clone().view(-1)
    n = flat.numel()
    mask = torch.rand(n) < prob
    flat[mask] = torch.randint(0, 2, (mask.sum(),)).float()
    return flat.view_as(image)


def add_blur(image, kernel_size):
    if kernel_size < 3:
        return image
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


def fgsm_attack(model, image, epsilon, target):
    img = image.clone().requires_grad_(True)
    output = model(img.unsqueeze(0))
    loss = F.cross_entropy(output, torch.tensor([target]))
    model.zero_grad()
    loss.backward()
    return torch.clamp(img + epsilon * img.grad.sign(), 0, 1).detach()


# ============================================================
# CNN confidence helper
# ============================================================
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
# Test runner
# ============================================================
def run_test(name, model, test_loader, distortion_fn, n=300):
    config = APCConfig(confidence_threshold=0.92)
    apc = APCController(config)

    correct = 0
    total_cost = 0
    total_steps = 0

    start = time.time()
    for i, (data, target) in enumerate(test_loader):
        if i >= n: break
        true_class = 1 if target.item() != 0 else 0
        clarity_fn = make_clarity_fn(model, data.squeeze(0), distortion_fn)
        result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
        if result.decision == true_class:
            correct += 1
        total_cost += result.total_cost
        total_steps += result.n_steps

    elapsed = time.time() - start
    acc = correct / n * 100
    cost = total_cost / n
    steps = total_steps / n

    marker = " ← BREAK" if acc < 90 else ""
    print(f"  {name:<35} acc={acc:5.1f}%  cost={cost:7.1f}  steps={steps:.1f}{marker}")
    return {"name": name, "acc": acc, "cost": cost}


# ============================================================
# Main
# ============================================================
def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  ACIES STRESS TEST 3 — Extreme Distortions                ║")
    print("╚" + "═" * 58 + "╝")
    print()

    model = load_model()
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    random.seed(42)

    results = []

    # === GAUSSIAN NOISE ===
    print("── Gaussian Noise ──")
    for sigma in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]:
        fn = lambda img, s=sigma: add_gaussian(img, s)
        results.append(run_test(f"noise sigma={sigma}", model, test_loader, fn))

    # === SALT & PEPPER ===
    print("\n── Salt & Pepper ──")
    for prob in [0.05, 0.1, 0.2, 0.3, 0.5]:
        fn = lambda img, p=prob: add_salt_pepper(img, p)
        results.append(run_test(f"s&p prob={prob}", model, test_loader, fn))

    # === BLUR ===
    print("\n── Blur ──")
    for ks in [3, 5, 7, 11, 15]:
        fn = lambda img, k=ks: add_blur(img, k)
        results.append(run_test(f"blur kernel={ks}", model, test_loader, fn))

    # === ROTATION ===
    print("\n── Rotation ──")
    for angle in [15, 30, 45, 60, 90]:
        fn = lambda img, a=angle: add_rotation(img, a)
        results.append(run_test(f"rotation {angle}°", model, test_loader, fn))

    # === OCCLUSION ===
    print("\n── Occlusion ──")
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        fn = lambda img, r=ratio: add_occlusion(img, r)
        results.append(run_test(f"occlusion {ratio*100:.0f}%", model, test_loader, fn))

    # === FGSM ADVERSARIAL ===
    print("\n── FGSM Adversarial ──")
    for eps in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        fn = lambda img, e=eps: fgsm_attack(model, img, e, 0)
        results.append(run_test(f"fgsm eps={eps}", model, test_loader, fn))

    # === COMBINED ATTACKS ===
    print("\n── Combined Attacks ──")
    fn = lambda img: add_gaussian(add_blur(img, 5), 0.5)
    results.append(run_test("blur5 + noise0.5", model, test_loader, fn))

    fn = lambda img: add_occlusion(add_gaussian(img, 0.3), 0.3)
    results.append(run_test("noise0.3 + occ30%", model, test_loader, fn))

    fn = lambda img: add_rotation(add_gaussian(img, 0.5), 30)
    results.append(run_test("noise0.5 + rot30°", model, test_loader, fn))

    # === SUMMARY ===
    print()
    print("=" * 60)
    print("BREAKING POINTS (acc < 90%):")
    print("=" * 60)
    found = False
    for r in results:
        if r["acc"] < 90:
            print(f"  ⚠ {r['name']}: {r['acc']:.1f}%")
            found = True
    if not found:
        print("  ✓ No breaking point found")

    print()
    print("ALL RESULTS:")
    print("-" * 60)
    for r in results:
        marker = " ←" if r["acc"] < 90 else ""
        print(f"  {r['name']:<35} {r['acc']:5.1f}%  cost={r['cost']:.1f}{marker}")
    print()


if __name__ == "__main__":
    main()
