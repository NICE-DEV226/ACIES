# CLI Reference

## acies-cli

The Go binary for running ACIES from the command line.

### Installation

```bash
go build -o acies-cli .
```

### Commands

#### `acies-cli run`

Run APC on a single task.

```bash
acies-cli run [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hardware` | default | Hardware profile |
| `--threshold` | 0.95 | Confidence threshold |
| `--max-steps` | 6 | Maximum perception steps |
| `--difficulty` | 0.0 | Task difficulty |
| `--verbose` | false | Show step-by-step output. **Needs an explicit value: `--verbose=true`** (a bare `--verbose` is ignored and swallows the next flag — [#12](https://github.com/NICE-DEV226/ACIES/issues/12)) |

**Examples:**

```bash
# Basic run
acies-cli run

# Jetson profile with high confidence
acies-cli run --hardware jetson --threshold 0.92

# Verbose mode (the value is required, see the flag table)
acies-cli run --verbose=true

# Hard task
acies-cli run --difficulty 0.7 --max-steps 8
```

**Output:**

```
Decision: 1 (true: 1) | CORRECT
Cost: 147.1 | Steps: 2 | Final belief: 0.983 | Risk: 0.166
```

With `--verbose=true`:

```
Profile: Jetson Orin Nano
Threshold: 0.92 | Max steps: 6 | Difficulty: 0.0
──────────────────────────────────────────────────
  Step 0: 224p      (clarity=0.78, score=0.032) → obs=1, belief=0.750 risk=2.500
  Step 1: 64p       (clarity=0.55, score=0.018) → obs=1, belief=0.891 risk=1.090
```

#### `acies-cli bench`

Run benchmark across multiple iterations.

```bash
acies-cli bench [options]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hardware` | default | Hardware profile |
| `--iterations` | 1000 | Number of iterations |
| `--threshold` | 0.95 | Confidence threshold |

**Example:**

```bash
# Benchmark on Jetson
acies-cli bench --hardware jetson --iterations 5000

# Benchmark on Raspberry Pi
acies-cli bench --hardware rpi --iterations 2000
```

**Output:**

```
ACIES Benchmark — Jetson Orin Nano — 5000 iterations
───────────────────────────────────────────────────────
Accuracy:    85.4%
Avg cost:    223.7
Max cost:    503.1
Avg steps:   5.6
EPC:         261.9
Time:        160ms (31058 runs/sec)

vs Fixed 1024p (cost=145): 55% more expensive
```

#### `acies-cli config`

Show hardware profiles.

```bash
# List all profiles
acies-cli config

# Show specific profile
acies-cli config jetson
```

**Output:**

```
Available hardware profiles:
  default    Default
  jetson     Jetson Orin Nano
  rpi        Raspberry Pi 5
  gpu        Desktop GPU (RTX 4090)
  tpu        Edge TPU (Coral)
```

With a specific profile:

```json
{
  "Name": "Jetson Orin Nano",
  "LatencyWeight": 0.5,
  "EnergyWeight": 0.3,
  "MemoryWeight": 0.2,
  "LatencyScale": 0.6,
  "EnergyScale": 0.8,
  "MemoryScale": 1.0
}
```

#### `acies-cli version`

Show version.

```bash
acies-cli version
# ACIES v0.1.4
```

## Python Module

### Direct Usage

```python
from acies import APCController, APCConfig, HardwareProfile

apc = APCController(APCConfig(
    confidence_threshold=0.92,
    hardware=HardwareProfile.jetson_orin(),
))

result = apc.run(true_class=1, clarity_fn=my_clarity_fn)
print(result.summary())
```

### Running Tests

```bash
python3 test_apc.py
```

### Running Benchmarks

```bash
# Simulated benchmark
python3 examples/benchmark.py

# Real MNIST benchmark
python3 examples/real_benchmark.py

# Image folder example
python3 examples/image_folder.py
```
