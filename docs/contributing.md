# Contributing

## Development Setup

```bash
git clone https://github.com/NICE-DEV226/ACIES.git
cd ACIES

# Python tests
python3 test_apc.py

# C++ library
cd cpp && make && cd ..

# Go CLI
go build -o acies-cli .
```

## Code Structure

```
ACIES/
├── acies/                  # Python package
│   ├── __init__.py         # Package exports
│   ├── actions.py          # Action space & hardware profiles
│   ├── belief.py           # Bayesian belief tracker
│   ├── clarity_learner.py  # Thompson Sampling
│   ├── safety.py           # Safety layer
│   ├── conviction.py       # Anti-oscillation
│   ├── change_point.py     # Change-point detection
│   ├── controller.py       # Main controller
│   └── accelerator.py      # C++ ctypes wrapper
├── cpp/                    # C++ library
│   ├── belief.h/.cpp       # Belief state
│   ├── clarity_learner.h/.cpp  # Thompson Sampling
│   ├── acies.h/.cpp        # C API
│   └── Makefile
├── core.go                 # Go implementation
├── main.go                 # Go CLI
├── test_apc.py             # Python tests
├── examples/               # Examples and benchmarks
└── docs/                   # Documentation
```

## Running Tests

```bash
# All tests
python3 test_apc.py

# Specific test
python3 -c "from test_apc import test_base_functionality; test_base_functionality()"
```

## Code Style

- Python: Follow PEP 8
- Go: Follow `gofmt`
- C++: Follow Google C++ Style Guide
- No external dependencies (stdlib only for Python)

## Adding a New Action

1. Add the action to `build_standard_actions()` in `actions.py`
2. Add corresponding Go action in `core.go`
3. Add C++ action in `clarity_learner.cpp` (if needed)
4. Update tests

## Adding a New Hardware Profile

1. Add static method to `HardwareProfile` in `actions.py`
2. Add to `HardwareProfiles()` in `core.go`
3. Update docs

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python3 test_apc.py`
5. Submit a pull request

## Reporting Issues

Please open an issue on GitHub with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
