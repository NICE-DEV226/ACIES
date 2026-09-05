"""
ACIES — Robustness Tests (pytest)

Run with: pytest test_apc.py -v
"""

import random
import time
import pytest

from acies import (
    APCController, APCConfig, HardwareProfile,
    BeliefState, ClarityLearner, SafetyLayer, SafetyConfig,
    build_standard_actions,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def seed():
    random.seed(42)


def make_clarity_fn(base_clarities, difficulty=0.0, noise=0.0):
    def clarity_fn(action):
        base = base_clarities.get(action.name, 0.5)
        adjusted = base * (1.0 - difficulty * 0.4)
        adjusted += random.gauss(0, noise)
        return max(0.01, min(0.99, adjusted))
    return clarity_fn


def make_shift_clarity_fn(phase_clarities, phase_length=50):
    call_count = [0]
    def clarity_fn(action):
        phase = call_count[0] // phase_length
        clarities = phase_clarities[min(phase, len(phase_clarities) - 1)]
        call_count[0] += 1
        return max(0.01, min(0.99, clarities.get(action.name, 0.5)))
    return clarity_fn


STANDARD_CLARITIES = {
    "64p": 0.55, "128p": 0.65, "224p": 0.75,
    "320p": 0.82, "512p": 0.88, "1024p": 0.93,
    "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
}


# ============================================================
# Tests
# ============================================================

class TestBeliefMath:
    """Tests unitaires du BeliefState."""

    def test_perfect_observation(self):
        b = BeliefState(prior=0.5)
        b.update(obs=1, clarity=1.0)
        assert abs(b.belief - 1.0) < 0.01

    def test_random_observation(self):
        b = BeliefState(prior=0.5)
        b.update(obs=1, clarity=0.5)
        assert abs(b.belief - 0.5) < 0.01

    def test_consistent_observations(self):
        b = BeliefState(prior=0.5)
        for _ in range(5):
            b.update(obs=1, clarity=0.8)
        assert b.belief > 0.8

    def test_delta_risk(self):
        b = BeliefState(prior=0.5)
        dr = b.delta_risk(0.9)
        assert dr > 0
        assert dr < b.risk

    def test_reset(self):
        b = BeliefState(prior=0.5)
        b.update(obs=1, clarity=1.0)
        b.reset()
        assert abs(b.belief - 0.5) < 0.01

    def test_efficiency(self):
        b = BeliefState(prior=0.5)
        eff = b.delta_risk_efficiency(0.9, 1.0)
        assert eff > 0


class TestThompsonSampling:
    """Le Thompson Sampling converge vers la vraie clarté."""

    def test_convergence(self):
        learner = ClarityLearner(n_actions=3)
        true_clarities = [0.7, 0.85, 0.95]

        for _ in range(500):
            for i in range(3):
                learner.sample(i)
                correct = random.random() < true_clarities[i]
                learner.update(i, correct)

        for i in range(3):
            error = abs(learner.mean(i) - true_clarities[i])
            assert error < 0.1, f"Action {i}: error {error:.3f} > 0.1"


class TestController:
    """Tests fonctionnels du contrôleur APC."""

    def test_base_functionality(self):
        apc = APCController(APCConfig(
            confidence_threshold=0.95,
            max_steps=6,
            hardware=HardwareProfile.default(),
        ))

        results = []
        for _ in range(300):
            true_class = random.randint(0, 1)
            result = apc.run(true_class, make_clarity_fn(STANDARD_CLARITIES))
            results.append(result)

        correct = sum(1 for r in results if r.correct)
        avg_steps = sum(r.n_steps for r in results) / len(results)

        obs_per_action = set()
        for r in results:
            for step in r.steps:
                obs_per_action.add(step.action.name)

        assert correct / len(results) > 0.7
        assert avg_steps >= 1.0
        assert len(obs_per_action) >= 3

    def test_hard_tasks(self):
        apc = APCController(APCConfig(
            confidence_threshold=0.95,
            max_steps=8,
        ))

        results = []
        for _ in range(200):
            true_class = random.randint(0, 1)
            result = apc.run(true_class, make_clarity_fn(STANDARD_CLARITIES, difficulty=0.7))
            results.append(result)

        avg_steps = sum(r.n_steps for r in results) / len(results)
        assert avg_steps >= 1.0

    def test_distribution_shift(self):
        apc = APCController(APCConfig(
            confidence_threshold=0.95,
            max_steps=8,
        ))

        phase_clarities = [
            STANDARD_CLARITIES,
            {k: v * 0.6 for k, v in STANDARD_CLARITIES.items()},
        ]

        results = []
        for _ in range(100):
            true_class = random.randint(0, 1)
            clarity_fn = make_shift_clarity_fn(phase_clarities, phase_length=3)
            result = apc.run(true_class, clarity_fn)
            results.append(result)

        correct = sum(1 for r in results if r.correct)
        assert correct / len(results) > 0.5

    def test_sensor_failure(self):
        apc = APCController(APCConfig(
            confidence_threshold=0.95,
            max_steps=6,
        ))

        clarities_failure = STANDARD_CLARITIES.copy()
        clarities_failure["crop_320"] = 0.50

        results = []
        for _ in range(200):
            true_class = random.randint(0, 1)
            result = apc.run(true_class, make_clarity_fn(clarities_failure))
            results.append(result)

        correct = sum(1 for r in results if r.correct)
        assert correct / len(results) > 0.5

    def test_hardware_profiles(self):
        profiles = [
            HardwareProfile.default(),
            HardwareProfile.jetson_orin(),
            HardwareProfile.raspberry_pi5(),
            HardwareProfile.desktop_gpu(),
            HardwareProfile.edge_tpu(),
        ]

        for profile in profiles:
            apc = APCController(APCConfig(
                confidence_threshold=0.90,
                hardware=profile,
            ))

            results = []
            for _ in range(100):
                true_class = random.randint(0, 1)
                result = apc.run(true_class, make_clarity_fn(STANDARD_CLARITIES))
                results.append(result)

            acc = sum(1 for r in results if r.correct) / len(results)
            assert acc > 0.5, f"Profile {profile} failed with accuracy {acc}"

    def test_stress(self):
        apc = APCController(APCConfig(
            confidence_threshold=0.95,
            max_steps=6,
        ))

        start = time.time()
        results = []
        for _ in range(5000):
            true_class = random.randint(0, 1)
            result = apc.run(true_class, make_clarity_fn(STANDARD_CLARITIES))
            results.append(result)
        elapsed = time.time() - start

        correct = sum(1 for r in results if r.correct)
        total_violations = apc.safety.state.n_violations

        assert total_violations < len(results) * 0.05
        assert correct / len(results) > 0.8
