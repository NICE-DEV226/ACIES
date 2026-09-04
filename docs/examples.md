# Examples

## 1. Basic Usage

```python
from acies import APCController, APCConfig, HardwareProfile

# Configure for Jetson
apc = APCController(APCConfig(
    confidence_threshold=0.92,
    max_steps=6,
    hardware=HardwareProfile.jetson_orin(),
))

# Define clarity for each action
def clarity_fn(action):
    clarities = {
        "64p": 0.55, "128p": 0.65, "224p": 0.75,
        "320p": 0.82, "512p": 0.88, "1024p": 0.93,
        "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
    }
    return clarities.get(action.name, 0.5)

# Run
result = apc.run(true_class=1, clarity_fn=clarity_fn)

print(f"Decision: {result.decision}")
print(f"Correct: {result.correct}")
print(f"Cost: {result.total_cost:.1f}")
print(f"Steps: {result.n_steps}")
print(f"Actions: {result.actions_taken}")
```

## 2. Batch Evaluation

```python
from acies import APCController, APCConfig
import random

apc = APCController(APCConfig(confidence_threshold=0.95))

def clarity_fn(action):
    return random.uniform(0.6, 0.95)

# Run 1000 trials
results = []
for _ in range(1000):
    true_class = random.randint(0, 1)
    result = apc.run(true_class, clarity_fn)
    results.append(result)

correct = sum(1 for r in results if r.correct)
avg_cost = sum(r.total_cost for r in results) / len(results)

print(f"Accuracy: {correct/len(results)*100:.1f}%")
print(f"Avg cost: {avg_cost:.1f}")
```

## 3. Custom Hardware Profile

```python
from acies import APCController, APCConfig, HardwareProfile

# Define a custom edge device
custom_profile = HardwareProfile(
    name="Custom Edge Device",
    latency_weight=0.4,
    energy_weight=0.4,
    memory_weight=0.2,
    latency_scale=1.5,    # 50% slower than reference
    energy_scale=0.8,     # 20% more efficient
    memory_scale=0.5,     # 50% less memory
)

apc = APCController(APCConfig(
    hardware=custom_profile,
))
```

## 4. Monitoring Safety

```python
from acies import APCController, APCConfig

apc = APCController(APCConfig(verbose=True))

result = apc.run(true_class=1, clarity_fn=my_fn)

# Check safety metrics
print(f"Violations: {apc.safety.state.n_violations}")
print(f"Emergency overrides: {apc.safety.state.n_emergency}")
print(f"Max risk ever: {max(apc.safety.state.risk_history):.2f}")
```

## 5. Change-Point Detection

```python
from acies import APCController, APCConfig, ChangePointDetector

apc = APCController(APCConfig(
    change_point_enabled=True,
    change_point_threshold=0.5,
))

# Simulate distribution shift
def clarity_fn(action):
    if step < 50:
        return 0.85  # Normal
    else:
        return 0.45  # Degraded

result = apc.run(true_class=1, clarity_fn=clarity_fn)

# Check change points
cp = apc.change_detector
print(f"Change points detected: {cp.n_total_cp}")
```

## 6. Go CLI

```bash
# Single run
./acies-cli run --hardware jetson --verbose

# Benchmark
./acies-cli bench --iterations 5000 --hardware rpi

# Show profiles
./acies-cli config
```

## 7. Docker

```bash
# Build
docker build -t acies .

# Run benchmark
docker run acies bench --iterations 1000 --hardware jetson

# Run with verbose output
docker run acies run --verbose
```

## 8. Real MNIST Benchmark

```bash
# Download MNIST CSV
cd data
wget "https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv/mnist_test.csv"

# Run benchmark
python3 examples/real_benchmark.py
```

## 9. Using the C++ Accelerator

```python
from acies.accelerator import BeliefState, ClarityLearner
import random

# Fast belief updates in C++
belief = BeliefState(prior=0.5)
for _ in range(1000):
    belief.update(obs=1, clarity=0.85)
    if belief.confidence > 0.95:
        break

# Fast Thompson Sampling in C++
learner = ClarityLearner(n_actions=9)
for _ in range(100):
    for i in range(9):
        p = learner.sample(i)
        correct = random.random() < p
        learner.update(i, correct)
```

## 10. Integration with Real Model

```python
from acies import APCController, APCConfig, HardwareProfile

# Your actual model
def get_clarity(action):
    """
    In production:
    1. Resize image to action.resolution
    2. Run your model
    3. Return confidence score
    """
    # Example: use a pre-trained model
    image = load_image("input.jpg")
    resized = resize(image, action.resolution)
    prediction = model.predict(resized)
    return prediction.confidence

apc = APCController(APCConfig(
    confidence_threshold=0.92,
    hardware=HardwareProfile.jetson_orin(),
))

result = apc.run(true_class=get_true_label(), clarity_fn=get_clarity)
print(f"Decision: {result.decision} (confidence: {result.final_belief:.3f})")
```
