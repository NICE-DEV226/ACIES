"""
ACIES — tests for the certified two-stage cascade (acies.cascade).

Run with: python3 -m pytest test_cascade.py -v
"""
import random

import pytest

from acies.cascade import CascadeSample, calibrate_cascade

COSTS = (30.0, 96.0)


def make_sample(rng, n, sep=0.55, noise=0.22, prevalence=0.4):
    """Reference answer r; first-stage score is a noisy, clipped signal of it; second stage = reference."""
    ref, score = [], []
    for _ in range(n):
        r = 1 if rng.random() < prevalence else 0
        ref.append(r)
        score.append(min(max(0.5 + (r - 0.5) * sep + rng.gauss(0, noise), 0.0), 1.0))
    return CascadeSample(score, list(ref), ref)


def disagreement(cal, sample):
    bad = esc = 0
    for s, r in zip(sample.first_score, sample.reference):
        a, e = cal.run(s, second=lambda r=r: r)
        bad += a != r
        esc += e
    return bad / len(sample.reference), esc / len(sample.reference)


def test_guarantee_holds_empirically_and_it_saves_cost():
    """Over many calibration draws, a certified cascade violates its bound at most delta of the time."""
    rng = random.Random(1)
    eps, delta, trials = 0.05, 0.1, 60
    violations = certified = 0
    for _ in range(trials):
        cal = calibrate_cascade(make_sample(rng, 500), make_sample(rng, 500), COSTS, eps=eps, delta=delta)
        if cal.certified:
            certified += 1
            dis, _ = disagreement(cal, make_sample(rng, 6000))        # large sample ~ population risk
            violations += dis > eps + 0.008                            # ~3 sigma of test-sampling noise
    assert certified > trials * 0.5                                    # not vacuous
    assert violations / trials <= delta + 0.05


def test_certified_cascade_is_cheaper_than_always_escalating():
    rng = random.Random(2)
    cal = calibrate_cascade(make_sample(rng, 1500, sep=0.9, noise=0.15), make_sample(rng, 1500, sep=0.9, noise=0.15),
                            COSTS, eps=0.05)
    assert cal.certified and cal.expected_cost < 0.7 * COSTS[1]


def test_no_proof_no_saving():
    """Too little data for a strict tolerance: nothing is certified and the reference always runs."""
    rng = random.Random(3)
    cal = calibrate_cascade(make_sample(rng, 40), make_sample(rng, 40), COSTS, eps=0.005)
    assert not cal.certified and cal.expected_cost == COSTS[1]
    calls = []
    answer, escalated = cal.run(0.99, second=lambda: calls.append(1) or 1)
    assert escalated and calls == [1] and answer == 1


def test_second_stage_is_called_lazily():
    rng = random.Random(4)
    cal = calibrate_cascade(make_sample(rng, 1500, sep=0.9, noise=0.15), make_sample(rng, 1500, sep=0.9, noise=0.15),
                            COSTS, eps=0.05)
    assert cal.certified
    calls = []
    cal.run(1.0, second=lambda: calls.append(1) or 1)
    cal.run(0.0, second=lambda: calls.append(1) or 0)
    assert calls == []                                                 # confident on both ends: never escalates
    mid = (cal.hi + cal.lo) / 2
    cal.run(mid, second=lambda: calls.append(1) or 0)
    assert calls == [1]


def test_conditional_risk_bounds_misses_and_false_alarms():
    rng = random.Random(5)
    em, ef, delta = 0.10, 0.05, 0.1
    viol = certified = 0
    for _ in range(40):
        cal = calibrate_cascade(make_sample(rng, 800), make_sample(rng, 800), COSTS, risk="conditional",
                                eps_miss=em, eps_fa=ef, delta=delta)
        if cal.certified:
            certified += 1
            test = make_sample(rng, 6000)
            miss = fa = npos = nneg = 0
            for s, r in zip(test.first_score, test.reference):
                a, _ = cal.run(s, second=lambda r=r: r)
                if r == 1:
                    npos += 1
                    miss += a == 0
                else:
                    nneg += 1
                    fa += a == 1
            viol += (miss / npos > em + 0.015) or (fa / nneg > ef + 0.008)
    assert certified > 15 and viol / 40 <= delta + 0.08


def test_conditional_needs_enough_positives():
    rng = random.Random(6)
    rare = lambda n: make_sample(rng, n, prevalence=0.01)
    cal = calibrate_cascade(rare(300), rare(300), COSTS, risk="conditional")
    assert not cal.certified                                           # ~3 positives: cannot certify a miss rate


def test_validation():
    rng = random.Random(7)
    s = make_sample(rng, 50)
    with pytest.raises(ValueError):
        CascadeSample([0.1], [1, 0], [1])
    with pytest.raises(ValueError):
        calibrate_cascade(s, s, COSTS, risk="nope")
    with pytest.raises(ValueError):
        calibrate_cascade(s, s, (10.0, 0.0))
    with pytest.raises(ValueError):
        calibrate_cascade(CascadeSample([], [], []), s, COSTS)
    with pytest.raises(ValueError):
        calibrate_cascade(s, s, COSTS, grid_hi=(0.1,), grid_lo=(0.5,))


def test_several_first_stages_share_one_guarantee_and_the_best_is_picked():
    from acies.cascade import Ladder, calibrate_ladders
    rng = random.Random(8)
    n = 1500
    ref = [1 if rng.random() < 0.4 else 0 for _ in range(n)]

    def stage(sep, noise):
        return [min(max(0.5 + (r - 0.5) * sep + rng.gauss(0, noise), 0.0), 1.0) for r in ref]
    weak, strong = stage(0.35, 0.3), stage(0.95, 0.12)
    half = n // 2
    mk = lambda sc, sl: CascadeSample(sc[sl], ref[sl], ref[sl])
    lad = lambda sc, cost, name: Ladder(mk(sc, slice(0, half)), mk(sc, slice(half, n)), cost, name)
    cal = calibrate_ladders([lad(weak, (10.0, 96.0), "weak-cheap"), lad(strong, (30.0, 96.0), "strong")], eps=0.05)
    assert cal.certified and cal.ladder_name == "strong"          # the cheap one cannot be trusted enough
    assert cal.expected_cost < COSTS[1]


def test_ladders_must_share_inputs():
    from acies.cascade import Ladder, calibrate_ladders
    rng = random.Random(9)
    a, b = make_sample(rng, 60), make_sample(rng, 60)             # different reference answers
    with pytest.raises(ValueError):
        calibrate_ladders([Ladder(a, a, COSTS), Ladder(b, b, COSTS)])
    with pytest.raises(ValueError):
        calibrate_ladders([])
