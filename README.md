<div align="center">

<img src="https://img.shields.io/badge/ACIES-Adaptive%20Perception%20Control-6366f1?style=for-the-badge&labelColor=1e1b4b" alt="ACIES"/>

<br/>

# ⚡ ACIES

### *Perception is not a fixed pipeline. It is a resource to control.*

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-00ff88?style=flat-square&logo=mit&logoColor=white)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Go 1.22+](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white)](https://go.dev/)
[![C++17](https://img.shields.io/badge/C++-17-00599C?style=flat-square&logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Tests](https://img.shields.io/badge/tests-8%2F8%20✓-22c55e?style=flat-square)](#-testing)
[![MNIST](https://img.shields.io/badge/benchmark-MNIST%2010k-f59e0b?style=flat-square&logo=tensorflow&logoColor=white)](#-benchmarks)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](#-docker)
[![PRs](https://img.shields.io/badge/PRs-welcome-00ff88?style=flat-square)](#-contributing)

<br/>

```
 █████╗ ██╗██████╗ ███████╗██╗ ██████╗
██╔══██╗██║██╔══██╗██╔════╝██║██╔════╝
███████║██║██████╔╝█████╗  ██║██║  ███╗
██╔══██║██║██╔══██╗██╔══╝  ██║██║   ██║
██║  ██║██║██║  ██║███████╗██║╚██████╔╝
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝
```

<br/>

A decision-theoretic framework that adaptively controls visual perception —
choosing **what to see** and **how much to see** — minimizing cost while
guaranteeing decision quality.

<br/>

<img src="https://img.shields.io/badge/saves-76%25%20cost-00ff88?style=for-the-badge" alt="76% savings"/>
<img src="https://img.shields.io/badge/accuracy-90.8%25-6366f1?style=for-the-badge" alt="90.8% accuracy"/>
<img src="https://img.shields.io/badge/speed-32k%20runs%2Fsec-f59e0b?style=for-the-badge" alt="32k runs/sec"/>

</div>

---

<br/>

## 🔍 What is ACIES?

Traditional perception systems process every input at **maximum resolution** — wasting compute on easy inputs and energy on obvious decisions.

ACIES flips this: it treats perception as a **sequential decision problem**. At each step, it asks:

> *"Which observation gives me the most information per unit of compute?"*

It then selects the cheapest action that meaningfully reduces decision risk — resizing, cropping, or activating only what's needed.

<br/>

```
  Traditional                         ACIES
  ──────────                          ─────
  Every image → 1024p → CNN          Image → 64p (easy? done.)
  Cost: ████████████ 100%             Image → 224p (maybe? check.)
  Accuracy: 98.8%                     Image → 1024p (hard. full power.)
                                     Cost: ███░░░░░░░░░ 24%
                                     Accuracy: 90.8%
```

<br/>

## 📊 Benchmarks

### Real Data — MNIST Test Set (10,000 images)

```
┌────────────────┬──────────┬─────────┬──────────┬─────────┐
│ Method         │ Accuracy │   Cost  │   Steps  │   EPC   │
├────────────────┼──────────┼─────────┼──────────┼─────────┤
│ Fixed 1024p    │  98.8%   │  385.8  │   2.1    │  390.3  │
│ ACIES          │  90.8%   │   93.6  │   5.2    │  103.1  │  ← 76% cheaper
│ Fixed 224p     │  82.1%   │   76.5  │   5.6    │   93.2  │
│ Random         │  95.5%   │  149.6  │   3.6    │  156.6  │
└────────────────┴──────────┴─────────┴──────────┴─────────┘
```

### Per-Hardware Savings vs Fixed 1024p

```
  Jetson Orin   ████████████████████░░░░░░░░░░░░  -41%
  Raspberry Pi  █████████████████████░░░░░░░░░░░  -45%
  Desktop GPU   ████████████████░░░░░░░░░░░░░░░░  -39%
  Edge TPU      ████████████████████░░░░░░░░░░░░  -42%
                0%                              50%
```

<br/>

## ⚡ Quick Start

### Python

```python
from acies import APCController, APCConfig, HardwareProfile

apc = APCController(APCConfig(
    confidence_threshold=0.92,
    hardware=HardwareProfile.jetson_orin(),
))

def clarity_fn(action):
    """Returns model confidence for this action."""
    return your_model.predict(resize(image, action.resolution)).confidence

result = apc.run(true_class=1, clarity_fn=clarity_fn)
# → Decision: 1 | Cost: 147.1 | Steps: 2 | Actions: 224p → 64p
```

### Go CLI

```bash
# Build
go build -o acies-cli .

# Run
./acies-cli run --hardware jetson --verbose

# Benchmark
./acies-cli bench --iterations 5000 --hardware rpi

# Profiles
./acies-cli config
```

### Docker

```bash
docker build -t acies .
docker run acies bench --iterations 1000 --hardware jetson
```

<br/>

## 🧠 How It Works

```
         ┌─────────────────────────────────────────┐
         │            APCController                 │
         │                                          │
  ┌──────┤  1. SAMPLE    Thompson Sampling          │
  │      │     → estimates clarity for each action  │
  │      │                                          │
  │      │  2. SCORE     ΔR/C for each action       │
  │      │     → risk reduction per unit cost       │
  │      │                                          │
  │      │  3. ADJUST    Conviction mechanism        │
  │      │     → anti-oscillation near threshold    │
  │      │                                          │
  │ loop │  4. FILTER    Safety layer               │
  │      │     → rejects risky actions              │
  │      │                                          │
  │      │  5. EXECUTE   Run selected action        │
  │      │     → resize, crop, or full resolution   │
  │      │                                          │
  │      │  6. UPDATE    Bayesian belief + learner  │
  │      │     → incorporate new observation        │
  │      │                                          │
  │      │  7. CHECK     Change-point detection     │
  │      │     → reset on distribution shift        │
  └──────┤                                          │
         │  8. DECIDE    confidence ≥ threshold?    │
         └──────────────┬──────────────────────────┘
                        │
                   decision
```

<br/>

## 🏗️ Architecture

<div align="center">

```
                    ┌──────────────────────┐
                    │     Input Image      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Action Selection   │
                    │                       │
                    │  ┌─────────────────┐  │
                    │  │  ClarityLearner │  │  Thompson Sampling
                    │  │  Beta(2,2)      │  │  learns P(correct|action)
                    │  └────────┬────────┘  │
                    │           │           │
                    │  ┌────────▼────────┐  │
                    │  │   BeliefState   │  │  Bayesian filter
                    │  │   P(Y=1|obs)    │  │  tracks uncertainty
                    │  └────────┬────────┘  │
                    │           │           │
                    │  ┌────────▼────────┐  │
                    │  │   SafetyLayer   │  │  Risk guarantees
                    │  │   max_risk=2.0  │  │  hard constraints
                    │  └────────┬────────┘  │
                    └───────────┼───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Selected Action      │
                    │  (e.g., 224p, crop)   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Observation          │
                    │  (model inference)    │
                    └───────────┬───────────┘
                                │
                         ┌──────▼──────┐
                         │   Decision   │
                         │  (0 or 1)    │
                         └─────────────┘
```

</div>

<br/>

## 📁 Project Structure

```
ACIES/
├── acies/                  Python package
│   ├── belief.py           Bayesian belief tracker
│   ├── clarity_learner.py  Thompson Sampling
│   ├── safety.py           Risk guarantees
│   ├── conviction.py       Anti-oscillation
│   ├── change_point.py     Distribution shift detection
│   ├── controller.py       Main orchestrator
│   └── accelerator.py      C++ ctypes wrapper
│
├── cpp/                    C++ core (belief + Thompson)
├── core.go + main.go       Go CLI (32k runs/sec)
├── examples/               Benchmarks & examples
├── docs/                   9 documentation files
├── data/                   MNIST test set
│
├── test_apc.py             8 robustness tests
├── Dockerfile              Multi-stage build
└── README.md               You are here
```

<br/>

## 🔧 Configuration

```python
APCConfig(
    # Decision
    confidence_threshold=0.95,    # Stop when confident enough
    max_steps=8,                  # Maximum perception steps

    # Risk
    max_risk=2.0,                 # Never exceed this risk
    emergency_risk=4.0,           # Force best action above this

    # Learning
    prior=0.5,                    # Initial belief
    temperature=1.0,              # Model calibration

    # Hardware
    hardware=HardwareProfile.jetson_orin(),
)
```

**5 built-in hardware profiles:**

| Profile | Device | Latency | Energy | Memory |
|:-------:|--------|:-------:|:------:|:------:|
| `default` | Generic | 1.0× | 1.0× | 1.0× |
| `jetson` | Jetson Orin Nano | 0.6× | 0.8× | 1.0× |
| `rpi` | Raspberry Pi 5 | 2.5× | 0.4× | 0.8× |
| `gpu` | RTX 4090 | 0.1× | 3.0× | 2.0× |
| `tpu` | Edge TPU (Coral) | 0.3× | 0.1× | 0.3× |

<br/>

## 🧪 Testing

```bash
python3 test_apc.py
```

```
╔═══════════════════════════════════════════════════════════╗
║  ACIES — Tests de robustesse                              ║
╚═══════════════════════════════════════════════════════════╝

TEST 7 : Belief math verification          ✓ PASSED
TEST 8 : Thompson Sampling convergence     ✓ PASSED
TEST 1 : Base functionality                ✓ PASSED  (99.3% acc, 2.3 steps)
TEST 2 : Hard tasks                        ✓ PASSED  (80.0% acc, 7.9 steps)
TEST 3 : Distribution shift                ✓ PASSED  (97.0% acc)
TEST 4 : Sensor failure                    ✓ PASSED  (96.0% acc)
TEST 5 : Hardware profiles                 ✓ PASSED  (5 profiles)
TEST 6 : Stress test                       ✓ PASSED  (5000 iters, <5% violations)

RESULTS: 8/8 passed, 0 failed
```

<br/>

## 📚 Documentation

| Document | What's Inside |
|----------|---------------|
| 🚀 [Installation](docs/installation.md) | Python, Go, C++, Docker setup |
| 🧠 [Architecture](docs/architecture.md) | Algorithms, control loop, math |
| ⚙️ [Configuration](docs/configuration.md) | Every parameter explained |
| 💻 [CLI Reference](docs/cli-reference.md) | Go commands & flags |
| 🐍 [Python API](docs/python-api.md) | Full API (classes, methods, types) |
| ⚡ [C++ API](docs/cpp-api.md) | C FFI for Python/Go/Rust |
| 📖 [Examples](docs/examples.md) | 10 real-world examples |
| 📊 [Benchmarks](docs/benchmarks.md) | MNIST results, Go perf, robustness |
| 🤝 [Contributing](docs/contributing.md) | Dev guide, code structure |

<br/>

## 🛠️ Tech Stack

<div align="center">

| Layer | Tech | Purpose |
|:-----:|:----:|---------|
| 🖥️ CLI | **Go** | Single binary, 32k runs/sec |
| 🧠 Core | **Python** | Belief, learner, safety, conviction |
| ⚡ Accelerator | **C++** | High-perf belief + Thompson via ctypes |
| 🐳 Deploy | **Docker** | Multi-stage (Python + Go + C++) |
| 🧪 Tests | **Python** | 8 robustness tests |
| 📊 Bench | **Python** | MNIST (10k images), 5 methods × 5 HW |

</div>

<br/>

## 🗺️ Roadmap

- [ ] Real CNN integration (ResNet, MobileNet)
- [ ] Multi-class classification (MNIST 10 digits)
- [ ] Streaming mode (video frames)
- [ ] REST API endpoint
- [ ] CI/CD with GitHub Actions
- [ ] PyPI package
- [ ] Rust FFI bindings
- [ ] WebAssembly build

<br/>

## 🤝 Contributing

See [docs/contributing.md](docs/contributing.md).

```bash
# Clone
git clone https://github.com/NICE-DEV226/ACIES.git

# Test
python3 test_apc.py

# Build Go CLI
go build -o acies-cli .

# Build C++ library
cd cpp && make
```

<br/>

## 📄 License

MIT License — see [LICENSE](LICENSE).

<br/>

## 📖 Citation

```bibtex
@software{acies2026,
  title   = {ACIES: Adaptive Perception Control},
  year    = {2026},
  url     = {https://github.com/NICE-DEV226/ACIES},
  author  = {NICE-DEV226}
}
```

<br/>

<div align="center">

**[Documentation](docs/)** · **[Examples](examples/)** · **[Benchmarks](docs/benchmarks.md)** · **[Contributing](docs/contributing.md)**

---

*Built with 🧠 by [NICE-DEV226](https://github.com/NICE-DEV226)*

</div>
