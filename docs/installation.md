# Installation

## Requirements

- Python 3.10+ (standard library only, no external dependencies)
- Go 1.22+ (for CLI binary)
- GCC/G++ (for C++ acceleration library)
- Make (for building C++ library)

## Quick Install

```bash
git clone https://github.com/NICE-DEV226/ACIES.git
cd ACIES
```

### Option 1: Python Only

No build step required. The Python modules work out of the box:

```python
from acies import APCController, APCConfig
```

### Option 2: With C++ Acceleration

```bash
cd cpp
make
cd ..
```

This builds `libacies.so` for high-performance belief tracking and Thompson Sampling.

### Option 3: Go CLI

```bash
go build -o acies-cli .
```

This produces a single binary `acies-cli` with no external dependencies.

### Option 4: Docker

```bash
docker build -t acies .
docker run acies bench --iterations 1000
```

## Verify Installation

```bash
# Python tests
python3 test_apc.py

# Go CLI
./acies-cli version

# C++ library
python3 -c "from acies.accelerator import BeliefState; print('C++ OK')"
```

## Platform Support

| Platform | Python | Go CLI | C++ Library |
|----------|:------:|:------:|:-----------:|
| Linux x86_64 | ✓ | ✓ | ✓ |
| Linux ARM64 | ✓ | ✓ | ✓ |
| macOS | ✓ | ✓ | ✓ |
| Windows | ✓ | ✓ | ✓ (MinGW) |

## Troubleshooting

### `ModuleNotFoundError: No module named 'acies'`

Make sure you're running from the project root directory:

```bash
cd /path/to/ACIES
python3 test_apc.py
```

### `g++: command not found`

Install a C++ compiler:

```bash
# Ubuntu/Debian
sudo apt install g++ make

# Arch
sudo pacman -S gcc make

# macOS
xcode-select --install
```

### `cannot find -laces`

The C++ library hasn't been built yet:

```bash
cd cpp && make
```
