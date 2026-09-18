"""
ACIES — Control-loop regression tests (pytest)

Covers the three things that used to be broken or missing:
  1. the emergency override no longer hijacks the untouched prior (it used to force
     1024p on every task);
  2. the exact optimal policy (acies.optimal) is correct and is a true lower bound;
  3. the step API (begin / next_action / observe / finish) works without ground truth.

Run with: python3 -m pytest test_control.py -v
"""

import random

import pytest

from acies import (
    APCConfig, APCController, BeliefState, HardwareProfile, SafetyConfig,
    SafetyLayer, build_standard_actions,
)
from acies.optimal import frontier, min_cost_for_error, solve_optimal

CLARITIES = {
    "64p": 0.55, "128p": 0.65, "224p": 0.75,
    "320p": 0.82, "512p": 0.88, "1024p": 0.93,
    "crop_224": 0.85, "crop_320": 0.90, "crop_512": 0.92,
}


def clarity_fn(action):
    return CLARITIES[action.name]


@pytest.fixture(autouse=True)
def seed():
    random.seed(42)


def _score_candidates(actions):
    """Candidates as the controller would hand them to the safety layer."""
    return [(a, 1.0 - 0.01 * i) for i, a in enumerate(actions)]


# ============================================================
# 1. Emergency override
# ============================================================

class TestEmergencyOverride:
    def test_prior_is_not_an_emergency(self):
        """belief=0.5 has risk 5.0 >= emergency_risk=4.0; it must not trigger the override."""
        actions = build_standard_actions()
        safety = SafetyLayer(SafetyConfig())
        belief = BeliefState(prior=0.5)
        chosen = safety.select(belief, _score_candidates(actions), n_observations=0)
        assert chosen is actions[0]          # top-scored candidate, not "the most expensive"
        assert safety.state.n_emergency == 0

    def test_emergency_fires_once_after_an_observation(self):
        actions = build_standard_actions()
        safety = SafetyLayer(SafetyConfig())
        belief = BeliefState(prior=0.5)
        safety.select(belief, _score_candidates(actions), n_observations=1)
        assert safety.state.n_emergency == 1   # counted once (used to be counted twice)

    def test_emergency_ranks_by_learned_clarity(self):
        actions = build_standard_actions()
        by_name = {a.name: a for a in actions}
        learned = {a.id: 0.5 for a in actions}
        learned[by_name["crop_224"].id] = 0.95   # learned to be the most informative
        safety = SafetyLayer(SafetyConfig())
        chosen = safety.select(
            BeliefState(prior=0.5), _score_candidates(actions), n_observations=1,
            emergency_clarity=learned,
        )
        assert chosen.name == "crop_224"

    def test_controller_does_not_lock_onto_1024p(self):
        apc = APCController(APCConfig(confidence_threshold=0.95, max_steps=6))
        results = [apc.run(random.randint(0, 1), clarity_fn) for _ in range(300)]
        first = [r.steps[0].action.name for r in results if r.steps]
        assert first.count("1024p") / len(first) < 0.2
        assert len({n for r in results for n in r.actions_taken}) >= 4

    def test_default_config_is_much_cheaper_than_fixed_1024p(self):
        """Regression guard: the old controller cost 245 here (always 1024p first)."""
        apc = APCController(APCConfig(confidence_threshold=0.95, max_steps=6))
        results = [apc.run(random.randint(0, 1), clarity_fn) for _ in range(1500)]
        cost = sum(r.total_cost for r in results) / len(results)
        err = 1 - sum(r.correct for r in results) / len(results)
        assert cost < 150
        assert err < 0.05

    def test_result_n_emergency_is_per_run(self):
        apc = APCController(APCConfig(confidence_threshold=0.95))
        results = [apc.run(random.randint(0, 1), clarity_fn) for _ in range(200)]
        assert sum(r.n_emergency for r in results) == apc.safety.state.n_emergency
        assert max(r.n_emergency for r in results) < 6   # not a cumulative counter


# ============================================================
# 2. Optimal policy
# ============================================================

class TestOptimalPolicy:
    hw = HardwareProfile.default()
    acts = build_standard_actions()

    def test_single_action_single_step_closed_form(self):
        one = [a for a in self.acts if a.name == "crop_224"]
        pol = solve_optimal(one, CLARITIES, self.hw, error_cost=1e6, horizon=1)
        ev = pol.evaluate()
        assert ev.steps == pytest.approx(1.0)
        assert ev.error == pytest.approx(1 - CLARITIES["crop_224"], abs=1e-6)
        assert ev.cost == pytest.approx(one[0].cost(self.hw))

    def test_stops_immediately_when_errors_are_cheap(self):
        pol = solve_optimal(self.acts, CLARITIES, self.hw, error_cost=1.0, horizon=6)
        assert pol.action(0.5, 0) is None
        ev = pol.evaluate()
        assert ev.cost == 0.0 and ev.error == pytest.approx(0.5)

    def test_matches_monte_carlo_of_its_own_policy(self):
        """Independent check: simulate the policy with the real BeliefState."""
        pol = solve_optimal(self.acts, CLARITIES, self.hw, error_cost=1000, horizon=6)
        exact = pol.evaluate()
        n, cost, errors = 30000, 0.0, 0
        rng = random.Random(7)
        for _ in range(n):
            y = rng.randint(0, 1)
            belief, t = BeliefState(prior=0.5), 0
            while (a := pol.action(belief.belief, t)) is not None:
                c = CLARITIES[a.name]
                obs = y if rng.random() < c else 1 - y
                belief.update(obs, c)
                cost += a.cost(self.hw)
                t += 1
            errors += belief.decision != y
        assert cost / n == pytest.approx(exact.cost, rel=0.03)
        assert errors / n == pytest.approx(exact.error, abs=0.004)

    def test_frontier_is_monotone(self):
        pts = frontier(self.acts, CLARITIES, self.hw, [50, 200, 1000, 3000, 20000], horizon=6)
        costs = [ev.cost for _, ev in pts]
        errs = [ev.error for _, ev in pts]
        assert costs == sorted(costs)
        assert errs == sorted(errs, reverse=True)

    def test_min_cost_for_error_meets_target(self):
        ev = min_cost_for_error(self.acts, CLARITIES, self.hw, target_error=0.02, horizon=6)
        assert ev.error <= 0.02
        assert 25 < ev.cost < 45      # ~37 for this clarity table

    def test_unreachable_target_raises(self):
        with pytest.raises(ValueError, match="unreachable"):
            min_cost_for_error(self.acts, CLARITIES, self.hw, target_error=1e-9, horizon=3)

    @pytest.mark.parametrize("kwargs", [
        {"error_cost": 0}, {"error_cost": 10, "horizon": 0}, {"error_cost": 10, "grid_size": 2},
    ])
    def test_input_validation(self, kwargs):
        with pytest.raises(ValueError):
            solve_optimal(self.acts, CLARITIES, self.hw, **kwargs)

    def test_rejects_degenerate_clarity(self):
        with pytest.raises(ValueError):
            solve_optimal(self.acts, {**CLARITIES, "64p": 1.0}, self.hw, error_cost=10)

    def test_controller_never_beats_the_optimum(self):
        """The optimum is a lower bound: at the controller's own error, it costs at least as much."""
        apc = APCController(APCConfig(confidence_threshold=0.95, max_steps=6))
        results = [apc.run(random.randint(0, 1), clarity_fn) for _ in range(3000)]
        err = 1 - sum(r.correct for r in results) / len(results)
        cost = sum(r.total_cost for r in results) / len(results)
        floor = min_cost_for_error(self.acts, CLARITIES, self.hw, target_error=err + 0.004, horizon=6)
        assert cost >= floor.cost


# ============================================================
# 3. Step API
# ============================================================

def _drive(apc, true_class, oracle=False):
    """Deployment-style loop: the environment is outside the controller."""
    apc.begin()
    while (action := apc.next_action()) is not None:
        c = CLARITIES[action.name]
        obs = true_class if random.random() < c else 1 - true_class
        apc.observe(action, obs, clarity=c if oracle else None)
    return apc.finish()


class TestStepAPI:
    def test_no_ground_truth_needed(self):
        apc = APCController(APCConfig(confidence_threshold=0.95))
        r = _drive(apc, true_class=1)
        assert r.correct is None                 # unknown at deployment
        assert r.decision in (0, 1, -1)
        assert r.n_steps == len(r.steps) >= 1

    def test_matches_run_exactly_under_same_seed(self):
        cfg = APCConfig(confidence_threshold=0.95)
        random.seed(5); a = APCController(cfg)
        ra = [a.run(random.randint(0, 1), clarity_fn) for _ in range(50)]
        random.seed(5); b = APCController(cfg)
        rb = []
        for _ in range(50):
            y = random.randint(0, 1)
            b.begin()
            while (act := b.next_action()) is not None:
                c = clarity_fn(act)
                obs = y if random.random() < c else 1 - y
                b.observe(act, obs, clarity=c, correct=(obs == y))
            rb.append(b.finish(y))
        assert [(r.decision, round(r.total_cost, 6), r.n_steps) for r in ra] == \
               [(r.decision, round(r.total_cost, 6), r.n_steps) for r in rb]

    def test_run_without_oracle_still_works(self):
        apc = APCController(APCConfig(confidence_threshold=0.95))
        results = [apc.run(random.randint(0, 1), clarity_fn, oracle_clarity=False)
                   for _ in range(1500)]
        assert 1 - sum(r.correct for r in results) / len(results) < 0.06

    def test_feedback_updates_learner(self):
        apc = APCController(APCConfig())
        action = apc.actions[0]
        before = apc.learner.n_observations(action.id)
        apc.feedback(action, True)
        assert apc.learner.n_observations(action.id) == before + 1

    def test_lifecycle_errors(self):
        apc = APCController(APCConfig())
        with pytest.raises(RuntimeError):
            apc.next_action()
        with pytest.raises(RuntimeError):
            apc.finish()
        apc.begin()
        action = apc.next_action()
        with pytest.raises(RuntimeError):
            apc.next_action()                                 # observe() pending
        with pytest.raises(RuntimeError):
            apc.observe(apc.actions[-1] if apc.actions[-1].id != action.id else apc.actions[0], 1)
        with pytest.raises(ValueError):
            apc.observe(action, 2)
        with pytest.raises(ValueError):
            apc.observe(action, 1, clarity=1.5)

    def test_new_task_resets_belief_and_conviction_but_keeps_learning(self):
        apc = APCController(APCConfig(confidence_threshold=0.95))
        _drive(apc, 1)
        learned = sum(apc.learner.n_observations(i) for i in range(len(apc.actions)))
        apc.begin()
        assert apc.belief.belief == 0.5
        assert apc.conviction.state.zone_steps == 0
        assert learned == sum(apc.learner.n_observations(i) for i in range(len(apc.actions)))

    def test_budget_is_enforced(self):
        apc = APCController(APCConfig(confidence_threshold=0.999, max_cost_per_image=20.0))
        r = _drive(apc, 1)
        assert r.total_cost <= 20.0
