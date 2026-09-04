# Real CNN Test

Benchmark ACIES with a real trained CNN on MNIST.

## Files

| File | Description |
|------|-------------|
| `train_cnn.py` | Train a CNN on MNIST (3 epochs, ~98.5% accuracy) |
| `real_benchmark.py` | ACIES + real CNN on noisy images |
| `demo.py` | Simple demo with ACIES |
| `check_import.py` | Verify package import |
| `mnist_cnn.pth` | Pre-trained CNN weights |

## Usage

```bash
# Train CNN (auto-downloads MNIST)
python3 train_cnn.py

# Run real benchmark
python3 real_benchmark.py

# Simple demo
python3 demo.py
```

## Requirements

```bash
pip install acies torch torchvision
```

## Results

| Method | Accuracy | Cost |
|--------|:--------:|:----:|
| Fixed 28x28 (noisy) | 67.3% | 187.2 |
| ACIES adaptive | 97.7% | 369.4 |

ACIES adapts resolution to image difficulty — 83.7% of actions are 1024p on noisy images.
