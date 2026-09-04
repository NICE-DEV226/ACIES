# C++ API Reference

## Header: `acies.h`

The C++ library provides a C API for use from any language via FFI.

### Build

```bash
cd cpp
make
# Produces libacies.so
```

### Include

```c
#include "acies.h"
```

---

## BeliefState

### Functions

#### `acies_belief_create`

```c
acies_belief_t acies_belief_create(double prior, double temperature);
```

Create a new belief state.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prior` | double | Prior probability P(Y=1) |
| `temperature` | double | Calibration temperature |

**Returns:** Opaque handle to belief state.

#### `acies_belief_destroy`

```c
void acies_belief_destroy(acies_belief_t b);
```

Free a belief state.

#### `acies_belief_reset`

```c
void acies_belief_reset(acies_belief_t b);
```

Reset belief to prior.

#### `acies_belief_update`

```c
void acies_belief_update(acies_belief_t b, int obs, double clarity);
```

Update belief with observation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `b` | acies_belief_t | Belief state handle |
| `obs` | int | Observation (0 or 1) |
| `clarity` | double | P(obs=Y \| action) |

#### `acies_belief_risk`

```c
double acies_belief_risk(acies_belief_t b);
```

Get current risk level.

#### `acies_belief_confidence`

```c
double acies_belief_confidence(acies_belief_t b);
```

Get confidence in decision.

#### `acies_belief_decision`

```c
int acies_belief_decision(acies_belief_t b);
```

Get optimal decision (0 or 1).

#### `acies_belief_delta_risk`

```c
double acies_belief_delta_risk(acies_belief_t b, double clarity);
```

Expected risk reduction for an action.

#### `acies_belief_delta_risk_efficiency`

```c
double acies_belief_delta_risk_efficiency(acies_belief_t b, double clarity, double cost);
```

Risk reduction per unit cost.

---

## ClarityLearner

### Functions

#### `acies_learner_create`

```c
acies_learner_t acies_learner_create(int n_actions);
```

Create a new clarity learner.

#### `acies_learner_destroy`

```c
void acies_learner_destroy(acies_learner_t l);
```

Free a clarity learner.

#### `acies_learner_reset`

```c
void acies_learner_reset(acies_learner_t l);
```

Reset all posteriors.

#### `acies_learner_reset_posterior`

```c
void acies_learner_reset_posterior(acies_learner_t l, int action_id);
```

Reset posterior for one action.

#### `acies_learner_sample`

```c
double acies_learner_sample(acies_learner_t l, int action_id);
```

Sample estimated clarity (Thompson Sampling).

#### `acies_learner_update`

```c
void acies_learner_update(acies_learner_t l, int action_id, int correct);
```

Update posterior with observation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `l` | acies_learner_t | Learner handle |
| `action_id` | int | Action index |
| `correct` | int | 1 if observation was correct, 0 otherwise |

#### `acies_learner_mean`

```c
double acies_learner_mean(acies_learner_t l, int action_id);
```

Posterior mean (no sampling).

---

## Usage from Python (ctypes)

```python
import ctypes

lib = ctypes.CDLL("./cpp/libacies.so")

# Create belief
belief = lib.acies_belief_create(0.5, 1.0)

# Update
lib.acies_belief_update(belief, 1, 0.85)

# Read
risk = lib.acies_belief_risk(belief)
confidence = lib.acies_belief_confidence(belief)

# Cleanup
lib.acies_belief_destroy(belief)
```

See `acies/accelerator.py` for a complete Python wrapper.

---

## Usage from Go (cgo)

```go
/*
#cgo LDFLAGS: -L. -laces
#include "acies.h"
*/
import "C"

belief := C.acies_belief_create(0.5, 1.0)
C.acies_belief_update(belief, 1, 0.85)
risk := C.acies_belief_risk(belief)
C.acies_belief_destroy(belief)
```

---

## Usage from Rust (FFI)

```rust
use std::os::raw::{c_double, c_int, c_void};

#[repr(C)]
struct AciesBelief(*mut c_void);

extern "C" {
    fn acies_belief_create(prior: c_double, temperature: c_double) -> *mut c_void;
    fn acies_belief_destroy(b: *mut c_void);
    fn acies_belief_update(b: *mut c_void, obs: c_int, clarity: c_double);
    fn acies_belief_risk(b: *mut c_void) -> c_double;
    fn acies_belief_confidence(b: *mut c_void) -> c_double;
}
```

---

## Performance

Benchmark of C++ vs Python (100,000 iterations):

| Operation | Python | C++ | Speedup |
|-----------|--------|-----|---------|
| Belief update | 0.23s | 0.08s | 2.9× |
| Thompson sample | 0.18s | 0.05s | 3.6× |
| Combined | 0.23s | 0.12s | 1.9× |

C++ provides ~2-4× speedup for individual operations. The speedup is most significant when:
- Running thousands of iterations
- Processing high-frequency video frames
- Embedding in latency-critical applications
