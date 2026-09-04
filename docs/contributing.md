# Contributing to ACIES

Thanks for your interest in contributing! ACIES is a project about **adaptive perception control** — making vision systems smarter by choosing *what* to perceive.

## Quick Start (5 minutes)

```bash
# Clone
git clone https://github.com/NICE-DEV226/ACIES.git
cd ACIES

# Run tests (no dependencies needed)
python3 test_apc.py

# Build Go CLI (optional)
go build -o acies-cli .

# Build C++ library (optional)
cd cpp && make && cd ..
```

## Ways to Contribute

### 🟢 Good First Issues (beginners)

Look for issues labeled [`good first issue`](https://github.com/NICE-DEV226/ACIES/labels/good%20first%20issue). These are well-scoped tasks with clear instructions.

Examples:
- Add a new hardware profile
- Improve documentation
- Add unit tests for a specific module
- Fix a typo in docs

### 🟡 Intermediate

- Add a new perception action (e.g., brightness adjustment, blur)
- Implement a new clarity estimation method
- Add logging/metrics to the controller
- Improve the Go CLI (new flags, output formats)

### 🔴 Advanced

- Multi-class extension (currently binary only)
- Neural clarity estimator (learn clarity from data)
- Hardware-in-the-loop integration
- ONNX export of the decision logic

## Code Structure

```
ACIES/
├── acies/                  # Python package (8 modules)
│   ├── controller.py       # Main APC loop — START HERE
│   ├── belief.py           # Bayesian belief tracker
│   ├── clarity_learner.py  # Thompson Sampling
│   ├── safety.py           # Risk guarantees
│   ├── conviction.py       # Anti-oscillation
│   ├── change_point.py     # BOCPD shift detection
│   ├── actions.py          # Action space & HW profiles
│   └── accelerator.py      # C++ ctypes wrapper
│
├── cpp/                    # C++ core library
├── core.go                 # Go implementation
├── main.go                 # Go CLI
├── test_apc.py             # 8 robustness tests
└── examples/               # Benchmarks & demos
```

**Start with `controller.py`** — it's the main loop that connects everything.

## Development Rules

1. **No external dependencies** — Python code uses stdlib only
2. **Run tests before submitting** — `python3 test_apc.py` must pass
3. **Follow existing code style** — PEP 8 (Python), gofmt (Go), Google C++ Style
4. **One PR = one feature/fix** — keep changes focused
5. **Add tests** for new functionality

## Pull Request Process

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `python3 test_apc.py`
5. Commit with clear message: `git commit -m "feat: add brightness action"`
6. Push and open a PR

**PR title format:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`

## Questions?

Open a [GitHub Discussion](https://github.com/NICE-DEV226/ACIES/discussions) or ping us on Discord (link in README).

## License

By contributing, you agree that your contributions will be licensed under MIT.
