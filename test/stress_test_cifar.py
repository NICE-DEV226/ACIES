"""
ACIES Stress Test 2 — CIFAR-10 (quick version)

Finds real breaking points with harder data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import random

from acies import APCController, APCConfig


class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def train_cifar(model, train_loader, epochs=3):
    print("  Training CIFAR-10 CNN (3 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            if batch_idx % 100 == 0:
                print(f"    batch {batch_idx}/{len(train_loader)}", end='\r')
        acc = correct / total * 100
        print(f"\n    Epoch {epoch+1}: loss={total_loss/len(train_loader):.3f} acc={acc:.1f}%")
    model.eval()
    torch.save(model.state_dict(), '/tmp/cifar_cnn.pth')
    return model


def add_gaussian(image, sigma):
    noise = torch.randn_like(image) * sigma
    return torch.clamp(image + noise, 0, 1)


def add_blur(image, kernel_size=5):
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    kernel = kernel.to(image.device)
    blurred = F.conv2d(image.unsqueeze(0), kernel.expand(3, -1, -1, -1), padding=padding, groups=3)
    return blurred.squeeze(0)


def fgsm_attack(image, epsilon, model, target):
    image.requires_grad = True
    output = model(image.unsqueeze(0))
    loss = F.cross_entropy(output, torch.tensor([target]))
    model.zero_grad()
    loss.backward()
    perturbed = image + epsilon * image.grad.sign()
    return torch.clamp(perturbed, 0, 1).detach()


def make_clarity_fn(model, image, distortion_type, level):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    resolutions = {"64p": 8, "128p": 14, "224p": 32, "320p": 32, "512p": 32, "1024p": 32}

    def clarity_fn(action):
        if distortion_type == "noise":
            img = add_gaussian(image, level)
        elif distortion_type == "blur":
            kernel = max(3, int(level * 15))
            if kernel % 2 == 0: kernel += 1
            img = add_blur(image, kernel)
        elif distortion_type == "fgsm":
            img = fgsm_attack(image.clone(), level, model, 0)
        else:
            img = image

        if action.name in resolutions:
            res = resolutions[action.name]
            if res != 32:
                img = F.interpolate(img.unsqueeze(0), size=(res, res), mode='bilinear', align_corners=False).squeeze(0)
                img = F.interpolate(img.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze(0)

        img = (img - mean) / std
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            probs = torch.softmax(output, dim=1)
            return probs.max().item()

    return clarity_fn


def test_noise_cifar(model, test_loader, n_per=200):
    print("=" * 60)
    print("TEST A: Gaussian noise on CIFAR-10")
    print("=" * 60)
    sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    results = []
    for sigma in sigmas:
        config = APCConfig(confidence_threshold=0.90)
        apc = APCController(config)
        correct = 0
        total_cost = 0
        for i, (data, target) in enumerate(test_loader):
            if i >= n_per: break
            true_class = 1 if target.item() < 5 else 0
            clarity_fn = make_clarity_fn(model, data.squeeze(0), "noise", sigma)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost
        acc = correct / n_per * 100
        cost = total_cost / n_per
        results.append({"sigma": sigma, "acc": acc, "cost": cost})
        marker = " ← BREAK" if acc < 90 else ""
        print(f"  sigma={sigma:.2f}  acc={acc:5.1f}%  cost={cost:7.1f}{marker}")
    print()
    return results


def test_blur_cifar(model, test_loader, n_per=200):
    print("=" * 60)
    print("TEST B: Blur on CIFAR-10")
    print("=" * 60)
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = []
    for level in levels:
        config = APCConfig(confidence_threshold=0.90)
        apc = APCController(config)
        correct = 0
        total_cost = 0
        for i, (data, target) in enumerate(test_loader):
            if i >= n_per: break
            true_class = 1 if target.item() < 5 else 0
            clarity_fn = make_clarity_fn(model, data.squeeze(0), "blur", level)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost
        acc = correct / n_per * 100
        cost = total_cost / n_per
        results.append({"level": level, "acc": acc, "cost": cost})
        marker = " ← BREAK" if acc < 90 else ""
        print(f"  level={level:.2f}  acc={acc:5.1f}%  cost={cost:7.1f}{marker}")
    print()
    return results


def test_fgsm_cifar(model, test_loader, n_per=200):
    print("=" * 60)
    print("TEST C: FGSM adversarial attack")
    print("=" * 60)
    epsilons = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
    results = []
    for eps in epsilons:
        config = APCConfig(confidence_threshold=0.90)
        apc = APCController(config)
        correct = 0
        total_cost = 0
        for i, (data, target) in enumerate(test_loader):
            if i >= n_per: break
            true_class = 1 if target.item() < 5 else 0
            clarity_fn = make_clarity_fn(model, data.squeeze(0), "fgsm", eps)
            result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
            if result.decision == true_class:
                correct += 1
            total_cost += result.total_cost
        acc = correct / n_per * 100
        cost = total_cost / n_per
        results.append({"eps": eps, "acc": acc, "cost": cost})
        marker = " ← BREAK" if acc < 90 else ""
        print(f"  eps={eps:.3f}  acc={acc:5.1f}%  cost={cost:7.1f}{marker}")
    print()
    return results


def test_thr_noise_matrix(model, test_loader, n_per=150):
    print("=" * 60)
    print("TEST D: Threshold × Noise matrix")
    print("=" * 60)
    thresholds = [0.75, 0.85, 0.92, 0.98]
    noises = [0.0, 0.3, 0.6, 1.0]
    header = f"{'thr/noise':>10}"
    for n in noises:
        header += f"  {n:.1f:>6}"
    print(header)
    print("-" * 50)
    for thr in thresholds:
        row = f"{thr:>10.2f}"
        for noise in noises:
            config = APCConfig(confidence_threshold=thr)
            apc = APCController(config)
            correct = 0
            total_cost = 0
            for i, (data, target) in enumerate(test_loader):
                if i >= n_per: break
                true_class = 1 if target.item() < 5 else 0
                clarity_fn = make_clarity_fn(model, data.squeeze(0), "noise", noise)
                result = apc.run(true_class=true_class, clarity_fn=clarity_fn)
                if result.decision == true_class:
                    correct += 1
                total_cost += result.total_cost
            acc = correct / n_per * 100
            row += f"  {acc:5.1f}%"
        print(row)
    print()


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  ACIES STRESS TEST 2 — CIFAR-10 + Adversarial            ║")
    print("╚" + "═" * 58 + "╝")
    print()

    import os
    transform = transforms.Compose([transforms.ToTensor()])

    if not os.path.exists('/tmp/cifar_cnn.pth'):
        print("No pre-trained CIFAR model found. Training (3 epochs)...")
        train_data = datasets.CIFAR10('/tmp/cifar', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
        model = CIFARCNN()
        train_cifar(model, train_loader, epochs=3)
    else:
        model = CIFARCNN()
        model.load_state_dict(torch.load('/tmp/cifar_cnn.pth', weights_only=True))
        model.eval()
        print("Loaded pre-trained CIFAR-10 model.")

    # Quick accuracy check
    test_data = datasets.CIFAR10('/tmp/cifar', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    correct = 0
    for i, (data, target) in enumerate(test_loader):
        if i >= 500: break
        with torch.no_grad():
            output = model(data)
            if output.argmax(1).item() == target.item():
                correct += 1
    print(f"  CNN base accuracy (500 images): {correct/500*100:.1f}%")

    random.seed(42)

    test_noise_cifar(model, test_loader, n_per=200)
    test_blur_cifar(model, test_loader, n_per=200)
    test_fgsm_cifar(model, test_loader, n_per=200)
    test_thr_noise_matrix(model, test_loader, n_per=150)

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
