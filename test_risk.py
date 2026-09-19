"""
ACIES — tests for distribution-free risk control (acies.risk).

The guarantee is checked empirically, not just algebraically: over thousands of simulated
calibration sets the certified setting must violate the risk bound with frequency <= delta.

Run with: python3 -m pytest test_risk.py -v
"""
import math
import random

import pytest

from acies.risk import (Certificate, binary_pvalue, binom_cdf, certify, hb_pvalue, risk_pvalue)


class TestBinomialTail:
    def test_closed_forms(self):
        assert binom_cdf(0, 10, 0.1) == pytest.approx(0.9 ** 10)
        assert binom_cdf(2, 5, 0.5) == pytest.approx(0.5)
        assert binom_cdf(1, 2, 0.3) == pytest.approx(1 - 0.3 ** 2)

    def test_edges_and_monotonicity(self):
        assert binom_cdf(-1, 10, 0.3) == 0.0
        assert binom_cdf(10, 10, 0.3) == 1.0
        vals = [binom_cdf(k, 50, 0.2) for k in range(0, 51)]
        assert vals == sorted(vals)
        assert binom_cdf(3, 40, 0.0) == 1.0 and binom_cdf(3, 40, 1.0) == 0.0

    def test_matches_exhaustive_sum(self):
        n, p, k = 30, 0.17, 7
        direct = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        assert binom_cdf(k, n, p) == pytest.approx(direct, rel=1e-9)


class TestPValues:
    def test_pvalue_is_valid_at_the_boundary(self):
        """H is true at R = alpha: the test may reject with probability <= delta."""
        rng = random.Random(1)
        n, alpha, delta, trials = 200, 0.05, 0.1, 4000
        rej = sum(binary_pvalue(sum(rng.random() < alpha for _ in range(n)), n, alpha) <= delta
                  for _ in range(trials))
        assert rej / trials <= delta + 0.01

    def test_hb_is_valid_and_conservative_for_bounded_losses(self):
        rng = random.Random(2)
        n, alpha, delta, trials = 200, 0.2, 0.1, 3000
        rej = 0
        for _ in range(trials):
            # losses in [0, 1] with mean exactly alpha
            losses = [rng.random() * 0.4 for _ in range(n)]          # mean 0.2
            rej += hb_pvalue(sum(losses) / n, n, alpha) <= delta
        assert rej / trials <= delta

    def test_pvalue_shrinks_with_evidence(self):
        assert binary_pvalue(0, 500, 0.05) < binary_pvalue(5, 500, 0.05) < binary_pvalue(25, 500, 0.05)
        assert hb_pvalue(0.3, 100, 0.2) == 1.0                       # empirical risk above alpha
        assert risk_pvalue([0, 0, 0, 0, 1], 0.5) == binary_pvalue(1, 5, 0.5)
        assert risk_pvalue([0.1, 0.2, 0.0], 0.5) == pytest.approx(hb_pvalue(0.1, 3, 0.5))


class TestCertify:
    def _losses(self, rng, n, risks):
        """One shared uniform per input: settings are nested and strongly correlated."""
        rows = []
        for _ in range(n):
            u = rng.random()
            rows.append([1.0 if u < r else 0.0 for r in risks])
        return rows

    def test_family_wise_guarantee_holds_empirically(self):
        """P(some certified setting has true risk > alpha) <= delta, over many calibration sets."""
        risks = [0.01, 0.02, 0.03, 0.045, 0.06, 0.09, 0.14]          # ordered safest -> riskiest
        alpha, delta, n, trials = 0.04, 0.1, 400, 1500
        for method in ("fixed_sequence", "bonferroni"):
            rng = random.Random(3)
            bad = 0
            for _ in range(trials):
                cert = certify(self._losses(rng, n, risks), alpha, delta, method=method)
                bad += any(risks[j] > alpha for j in cert.certified)
            assert bad / trials <= delta + 0.02, (method, bad / trials)

    def test_it_is_not_vacuous(self):
        """With clearly safe settings and enough data, they are certified."""
        rng = random.Random(4)
        risks = [0.002, 0.004, 0.2]
        cert = certify(self._losses(rng, 3000, risks), alpha=0.03, delta=0.1)
        assert cert.certified == [0, 1]

    def test_fixed_sequence_stops_at_first_failure(self):
        losses = [[1.0 if i < 60 else 0.0, 0.0, 0.0] for i in range(500)]   # setting 0 unsafe
        cert = certify(losses, alpha=0.02, delta=0.1)
        assert cert.certified == [] and cert.tested == [0]
        cert2 = certify(losses, alpha=0.02, delta=0.1, order=[1, 2, 0])
        assert cert2.certified == [1, 2] and cert2.tested == [1, 2, 0]
        cert3 = certify(losses, alpha=0.02, delta=0.1, method="bonferroni")
        assert cert3.certified == [1, 2]

    def test_best_picks_the_cheapest_certified(self):
        losses = [[0.0, 0.0, 0.0] for _ in range(300)]
        cert = certify(losses, alpha=0.05, delta=0.1)
        assert cert.certified == [0, 1, 2]
        assert cert.best([9.0, 3.0, 5.0]) == 1
        assert Certificate([], [], 0.1, 0.1, 1, "x").best([1.0]) is None

    @pytest.mark.parametrize("args", [
        dict(alpha=0.0), dict(alpha=1.0), dict(alpha=0.1, delta=0.0), dict(alpha=0.1, method="nope"),
        dict(alpha=0.1, order=[0, 0]), dict(alpha=0.1, order=[0]),
    ])
    def test_input_validation(self, args):
        with pytest.raises(ValueError):
            certify([[0.0, 0.0]] * 10, **args)

    def test_rejects_malformed_losses(self):
        with pytest.raises(ValueError):
            certify([], 0.1)
        with pytest.raises(ValueError):
            certify([[0.0, 0.0], [0.0]], 0.1)
        with pytest.raises(ValueError):
            certify([[0.0, 1.5]], 0.1)


class TestConjunctiveConstraints:
    def test_certify_pvalues_matches_certify(self):
        rng = random.Random(6)
        L = [[1.0 if rng.random() < r else 0.0 for r in (0.01, 0.03, 0.2)] for _ in range(500)]
        from acies.risk import certify_pvalues
        a = certify(L, alpha=0.05, delta=0.1)
        b = certify_pvalues(a.pvalues, delta=0.1)
        assert a.certified == b.certified and a.tested == b.tested

    def test_intersection_union_is_valid_when_one_constraint_sits_on_its_boundary(self):
        """Miss rate exactly at its limit, false alarms comfortably fine: still <= delta certified."""
        rng = random.Random(7)
        em, ef, delta, trials = 0.05, 0.02, 0.1, 3000
        certified = 0
        for _ in range(trials):
            miss = [1.0 if rng.random() < em else 0.0 for _ in range(150)]        # R_miss = em (H true)
            fa = [1.0 if rng.random() < 0.004 else 0.0 for _ in range(400)]       # R_fa well below ef
            p = max(risk_pvalue(miss, em), risk_pvalue(fa, ef))
            certified += p <= delta
        assert certified / trials <= delta + 0.015

    def test_conjunction_needs_both_constraints_to_hold(self):
        from acies.risk import certify_pvalues
        p_ok, p_bad = 0.01, 0.6
        assert certify_pvalues([max(p_ok, p_ok)], 0.1).certified == [0]
        assert certify_pvalues([max(p_ok, p_bad)], 0.1).certified == []
