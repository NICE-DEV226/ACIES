# Configuration Reference

## APCConfig

The main configuration object for the controller.

```python
from acies import APCConfig, HardwareProfile

config = APCConfig(
    prior=0.5,
    temperature=1.0,
    confidence_threshold=0.95,
    max_steps=6,
    max_risk=2.0,
    emergency_risk=4.0,
    min_observations=1,
    hardware=HardwareProfile.jetson_orin(),
    verbose=False,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prior` | float | 0.5 | Prior probability P(Y=1). Set to 0.5 for no prior knowledge. |
| `temperature` | float | 1.0 | Calibration temperature. >1 = less confident observations. |
| `confidence_threshold` | float | 0.5 | Stop when confidence exceeds this value. |
| `max_steps` | int | 6 | Maximum number of perception steps. |
| `max_risk` | float | 2.0 | Maximum acceptable risk level. |
| `emergency_risk` | float | 4.0 | Risk level that triggers emergency override. |
| `min_observations` | int | 1 | Minimum observations before allowing a decision. |
| `hardware` | HardwareProfile | default | Hardware cost profile. |
| `conviction_zone_start` | float | 0.85 | Fraction of threshold to enter conviction zone. |
| `conviction_oscillation_threshold` | int | 3 | Steps in conviction zone before force-commit. |
| `change_point_enabled` | bool | True | Enable Bayesian change-point detection. |
| `change_point_threshold` | float | 0.5 | Detection threshold for change points. |
| `change_point_hazard` | float | 1/200 | Hazard rate prior for change points. |
| `verbose` | bool | False | Print debug information. |

## HardwareProfile

Defines cost weights and scales for different hardware.

### Built-in Profiles

```python
HardwareProfile.default()        # Generic hardware
HardwareProfile.jetson_orin()    # NVIDIA Jetson Orin Nano
HardwareProfile.raspberry_pi5()  # Raspberry Pi 5
HardwareProfile.desktop_gpu()    # Desktop GPU (RTX 4090)
HardwareProfile.edge_tpu()       # Google Edge TPU (Coral)
```

### Custom Profile

```python
profile = HardwareProfile(
    name="My Custom Device",
    latency_weight=0.4,    # Weight for latency in cost function
    energy_weight=0.4,     # Weight for energy in cost function
    memory_weight=0.2,     # Weight for memory in cost function
    latency_scale=1.0,     # Multiplier for latency (0.5 = 2× faster)
    energy_scale=1.0,      # Multiplier for energy
    memory_scale=1.0,      # Multiplier for memory
)
```

### Cost Formula

```
cost = latency_weight × base_latency × latency_scale
     + energy_weight × base_energy × energy_scale
     + memory_weight × base_memory × memory_scale
```

## SafetyConfig

Configuration for the safety layer.

```python
from acies import SafetyConfig

safety_config = SafetyConfig(
    max_risk=2.0,
    emergency_risk=4.0,
    min_observations=1,
    confidence_threshold=0.95,
    risk_margin=0.6,
    allow_abstention=True,
)
```

## ConvictionConfig

Configuration for the conviction mechanism.

```python
from acies import ConvictionConfig

conviction_config = ConvictionConfig(
    zone_start=0.90,
    bonus_high_clarity=1.3,
    penalty_low_clarity=0.5,
    oscillation_threshold=4,
    force_commit_clarity=0.8,
)
```

## ChangePointConfig

Configuration for change-point detection.

```python
from acies import ChangePointConfig

cp_config = ChangePointConfig(
    hazard_rate=1/200,
    observation_sigma=0.1,
    threshold=0.5,
    min_run_length=5,
    warmup_observations=10,
)
```

## Go CLI Configuration

The Go CLI uses command-line flags:

```bash
aries-cli run --hardware jetson --threshold 0.92 --max-steps 8 --verbose
aries-cli bench --iterations 5000 --hardware rpi
aries-cli config jetson
```

### Available Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--hardware` | default | Hardware profile (default/jetson/rpi/gpu/tpu) |
| `--threshold` | 0.95 | Confidence threshold |
| `--max-steps` | 6 | Maximum perception steps |
| `--difficulty` | 0.0 | Task difficulty (0=easy, 1=impossible) |
| `--verbose` | false | Verbose output |
| `--iterations` | 1000 | Number of benchmark iterations |
