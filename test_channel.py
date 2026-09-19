"""
ACIES — tests for the channel controller and the contextual selector.

Run with: python3 -m pytest test_channel.py -v
"""

import math
import random

import pytest

from acies import BeliefState, HardwareProfile
from acies.actions import Action, ActionType
from acies.channel import ChannelConfig, ChannelController, ChannelLearner
from acies.selector import ResolutionSelector, detection_features

HW = HardwareProfile(name="unit", latency_weight=1.0, energy_weight=0.0, memory_weight=0.0)


def make_actions(costs):
    return [Action(id=i, name=f"a{i}", action_type=ActionType.RESOLUTION,
                   base_latency_ms=float(c), pixel_ratio=0.1 * (i + 1))
            for i, c in enumerate(costs)]


def symmetric_learner(accuracies, n=200000):
    """K=2 learner whose tables are exactly a binary symmetric channel of the given accuracy."""
    lr = ChannelLearner(len(accuracies), 2, alpha=1e-9)
    for a, c in enumerate(accuracies):
        lr.counts[a][1] = [(1 - c) * n, c * n]
        lr.counts[a][0] = [c * n, (1 - c) * n]
    return lr


@pytest.fixture(autouse=True)
def seed():
    random.seed(3)


class TestChannelMath:
    def test_reduces_to_the_binary_symmetric_model(self):
        """K=2, carry=1, tau=1: identical to BeliefState's Bayes update."""
        acc = [0.6, 0.75, 0.9]
        ctl = ChannelController(make_actions([1, 2, 3]), symmetric_learner(acc),
                                ChannelConfig(prior=0.5, carry=1.0, tau=1.0, hardware=HW))
        ref = BeliefState(prior=0.5)
        b = 0.5
        for idx, obs in ((0, 1), (1, 1), (2, 0), (1, 0)):
            b = ctl._posterior(b, idx, obs)
            ref.update(obs, acc[idx])
            assert b == pytest.approx(ref.belief, abs=1e-4)

    def test_voi_is_nonnegative_and_zero_for_an_uninformative_channel(self):
        lr = ChannelLearner(2, 3, alpha=1e-9)
        for y in (0, 1):
            lr.counts[0][y] = [1000.0, 1000.0, 1000.0]      # outcome independent of Y
        lr.counts[1][1], lr.counts[1][0] = [10.0, 10.0, 980.0], [980.0, 10.0, 10.0]
        ctl = ChannelController(make_actions([1, 1]), lr, ChannelConfig(prior=0.5, hardware=HW))
        assert ctl.voi(0.5, 0) == pytest.approx(0.0, abs=1e-9)
        assert ctl.voi(0.5, 1) > 0.3
        for b in (0.05, 0.3, 0.5, 0.8):
            assert ctl.voi(b, 1) >= -1e-12

    def test_learner_prior_is_the_class_frequency(self):
        lr = ChannelLearner(2, 2)
        for i in range(100):
            for a in range(2):
                lr.feedback(a, int(i < 10), 0)
        assert lr.prior() == pytest.approx(0.10, abs=0.02)

    def test_rare_class_without_evidence_decides_no_for_free(self):
        lr = ChannelLearner(2, 2)
        for i in range(400):
            for a in range(2):
                lr.feedback(a, int(i % 10 == 0), random.randint(0, 1))   # outcomes carry no signal
        ctl = ChannelController(make_actions([1, 2]), lr, ChannelConfig(error_cost=100, hardware=HW))
        ctl.begin()
        assert ctl.next_action() is None
        r = ctl.finish()
        assert r.decision == 0 and r.total_cost == 0.0


class TestChannelController:
    def _learner(self):
        lr = ChannelLearner(3, 4, alpha=0.5)
        rng = random.Random(0)
        for _ in range(3000):
            y = rng.randint(0, 1)
            for a, c in enumerate((0.6, 0.8, 0.95)):
                good = rng.random() < c
                o = (2 + y) if good else rng.choice([0, 1] if y else [2, 3])
                lr.feedback(a, y, o)
        return lr

    @pytest.mark.parametrize("planning", [True, False])
    def test_never_repeats_an_action_and_terminates(self, planning):
        acts = make_actions([1, 3, 10])
        ctl = ChannelController(acts, self._learner(),
                                ChannelConfig(error_cost=5000, planning=planning, hardware=HW))
        for _ in range(200):
            ctl.begin()
            seen = []
            while (a := ctl.next_action()) is not None:
                assert a.id not in seen
                seen.append(a.id)
                ctl.observe(a, random.randint(0, 3))
            r = ctl.finish()
            assert r.n_steps == len(seen) <= 3
            assert r.total_cost == pytest.approx(sum(acts[i].base_latency_ms for i in seen))

    def test_more_expensive_errors_buy_more_perception(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        costs = []
        for lam in (5, 50, 500, 5000):
            ctl = ChannelController(acts, lr, ChannelConfig(error_cost=lam, hardware=HW))
            c = 0.0
            for _ in range(300):
                y = random.randint(0, 1)
                ctl.begin()
                while (a := ctl.next_action()) is not None:
                    ctl.observe(a, (2 + y) if random.random() < (0.6, 0.8, 0.95)[a.id] else random.randint(0, 3))
                c += ctl.finish().total_cost
            costs.append(c / 300)
        assert costs == sorted(costs) and costs[-1] > costs[0]

    def test_planning_beats_greedy_when_a_cheap_first_step_does_not_save_the_second(self):
        """
        Cheap weak action (cost 3, 62 %), decent one (cost 5, 90 %), costly one (cost 40, 99 %).
        Greedy value-of-information per cost always starts with the cheap one and then still
        pays for the decent one; the plan goes straight to it. Objective = cost + lambda*error,
        simulated from the true channels.
        """
        acts = make_actions([3, 5, 40])
        lr = symmetric_learner([0.62, 0.9, 0.99])
        lam = 400.0

        def objective(planning, n=2500):
            rng = random.Random(11)
            ctl = ChannelController(acts, lr, ChannelConfig(error_cost=lam, planning=planning,
                                                            carry=1.0, hardware=HW, grid=101))
            tot = 0.0
            for _ in range(n):
                y = rng.random() < ctl.prior
                ctl.begin()
                while (a := ctl.next_action()) is not None:
                    ctl.observe(a, rng.choices(range(2), weights=lr.likelihood(a.id, int(y)))[0])
                r = ctl.finish()
                tot += r.total_cost + lam * (r.decision != int(y))
            return tot / n

        plan, greedy = objective(True), objective(False)
        assert plan < 0.9 * greedy          # measured: ~25 % better

    def test_lifecycle_and_validation(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        ctl = ChannelController(acts, lr, ChannelConfig(error_cost=5000, hardware=HW))
        with pytest.raises(RuntimeError):
            ctl.next_action()
        ctl.begin()
        a = ctl.next_action()
        with pytest.raises(RuntimeError):
            ctl.next_action()                       # observe() pending
        with pytest.raises(RuntimeError):
            ctl.observe(acts[-1] if acts[-1].id != a.id else acts[0], 0)
        with pytest.raises(ValueError):
            ctl.observe(a, 99)
        with pytest.raises(ValueError):
            ChannelController(acts, lr, ChannelConfig(carry=1.5))
        with pytest.raises(ValueError):
            ChannelController(acts, lr, ChannelConfig(tau=0.0))
        with pytest.raises(ValueError):
            ChannelController(acts[:2], lr)
        with pytest.raises(ValueError):
            ChannelLearner(3, 1)
        with pytest.raises(ValueError):
            lr.feedback(0, 2, 0)

    def test_first_action_is_forced_and_later_actions_move_upward(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        ctl = ChannelController(acts, lr, ChannelConfig(error_cost=5000, first_action=1, hardware=HW))
        for _ in range(100):
            ctl.begin()
            seen = []
            while (a := ctl.next_action()) is not None:
                seen.append(a.id)
                ctl.observe(a, random.randint(0, 3))
            ctl.finish()
            assert seen[0] == 1 and 0 not in seen          # forced start; nothing below it afterwards
            assert seen == sorted(set(seen))
        with pytest.raises(ValueError):
            ChannelController(acts, lr, ChannelConfig(first_action=3))

    def test_external_belief_overrides_the_update_and_planning_continues(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        ctl = ChannelController(acts, lr, ChannelConfig(error_cost=300, first_action=0, hardware=HW))
        ctl.begin()
        a = ctl.next_action()
        step = ctl.observe(a, 0, belief=0.999)
        assert step.belief_after == pytest.approx(0.999)
        assert ctl.next_action() is None                    # certain: nothing left worth its cost
        ctl.finish()
        ctl.begin()
        a = ctl.next_action()
        ctl.observe(a, 0, belief=0.5)                       # maximally unsure: escalates
        assert ctl.next_action() is not None
        with pytest.raises(ValueError):
            ctl.observe(ctl._task["pending"] and acts[1], 0, belief=1.5)

    def test_asymmetric_costs_move_the_decision_threshold(self):
        """A miss costing rho times a false alarm: decide 'positive' from belief 1/(1+rho) upward."""
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        for rho, belief, expected in ((1.0, 0.4, 0), (1.0, 0.6, 1), (5.0, 0.2, 1), (5.0, 0.1, 0), (0.25, 0.7, 0)):
            ctl = ChannelController(acts, lr, ChannelConfig(error_cost=1.0, miss_cost=rho, first_action=0, hardware=HW))
            ctl.begin()
            a = ctl.next_action()
            ctl.observe(a, 0, belief=belief)
            assert ctl.next_action() is None            # error_cost = 1: nothing is worth running
            assert ctl.finish().decision == expected, (rho, belief)

    def test_higher_miss_cost_buys_more_perception_near_the_positive_side(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        cost = {}
        for rho in (1.0, 6.0):
            ctl = ChannelController(acts, lr, ChannelConfig(error_cost=40, miss_cost=rho, first_action=0, hardware=HW))
            c = 0.0
            for b1 in (0.02, 0.03, 0.04):
                ctl.begin()
                a = ctl.next_action()
                ctl.observe(a, 0, belief=b1)
                while (a := ctl.next_action()) is not None:
                    ctl.observe(a, random.randint(0, 3))
                c += ctl.finish().total_cost
            cost[rho] = c
        assert cost[6.0] > cost[1.0]        # a 2-4 % chance of a costly miss is worth another look; of a plain error, not

    def test_miss_cost_validation_and_default_is_symmetric(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        with pytest.raises(ValueError):
            ChannelController(acts, lr, ChannelConfig(miss_cost=0.0))
        assert ChannelConfig().miss_cost == 1.0

    def test_replans_when_the_learner_changes(self):
        acts = make_actions([1, 3, 10])
        lr = self._learner()
        ctl = ChannelController(acts, lr, ChannelConfig(error_cost=300, hardware=HW))
        ctl.begin()
        ctl.next_action()
        first = ctl._plan
        ctl.begin()
        ctl.next_action()
        assert ctl._plan is first                    # nothing changed: plan reused
        ctl.feedback(acts[0], 1, 3)
        ctl.begin()
        ctl.next_action()
        assert ctl._plan is not first                # learner changed: re-planned


class TestSelector:
    def _data(self, n=400):
        rng = random.Random(5)
        X, Q = [], []
        for _ in range(n):
            x = [rng.uniform(0, 1), rng.uniform(0, 1)]
            Q.append([0.3 + 0.4 * x[0], 0.5 + 0.3 * x[1] - 0.1 * x[0], 0.9 - 0.2 * x[0] * x[1]])
            X.append(x)
        return X, Q

    def test_fit_recovers_a_smooth_target(self):
        X, Q = self._data()
        sel = ResolutionSelector.fit(X, Q, ["a", "b", "c"], ridge=0.1)
        err = sum(abs(p - q) for x, row in zip(X, Q) for p, q in zip(sel.predict(x), row)) / (3 * len(X))
        assert err < 0.01

    def test_cost_penalty_moves_the_choice_to_cheaper_actions(self):
        X, Q = self._data()
        sel = ResolutionSelector.fit(X, Q, ["a", "b", "c"], ridge=1.0)
        costs = [0.0, 10.0, 100.0]
        x = [0.5, 0.5]
        assert sel.select(x, costs, mu=0.0) == 2       # quality only: the best action
        assert sel.select(x, costs, mu=1.0) == 0       # cost dominates: the cheapest

    def test_save_load_roundtrip(self, tmp_path):
        X, Q = self._data(100)
        sel = ResolutionSelector.fit(X, Q, ["a", "b", "c"])
        path = str(tmp_path / "sel.json")
        sel.save(path)
        back = ResolutionSelector.load(path)
        assert back.predict([0.2, 0.7]) == pytest.approx(sel.predict([0.2, 0.7]))

    def test_validation(self):
        X, Q = self._data(50)
        with pytest.raises(ValueError):
            ResolutionSelector.fit([], [], ["a"])
        with pytest.raises(ValueError):
            ResolutionSelector.fit(X, Q, ["a", "b"])           # Q has 3 columns
        with pytest.raises(ValueError):
            ResolutionSelector.fit(X, Q, ["a", "b", "c"], ridge=0)
        sel = ResolutionSelector.fit(X, Q, ["a", "b", "c"])
        with pytest.raises(ValueError):
            sel.predict([0.1])                                  # wrong feature count
        with pytest.raises(ValueError):
            sel.select([0.1, 0.2], [0, 1, 2], mu=-1)

    def test_detection_features(self):
        dets = [(10, 10, 110, 110, 0.9, 0), (200, 50, 230, 90, 0.3, 2), (5, 5, 20, 20, 0.1, 1)]
        f = detection_features(dets, (640, 480))
        assert len(f) == 15 and all(math.isfinite(v) for v in f)
        assert f[5] == pytest.approx(0.9)                        # top confidence
        empty = detection_features([], (640, 480))
        assert len(empty) == 15 and all(v == 0.0 for v in empty)
