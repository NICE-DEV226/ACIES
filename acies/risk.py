"""
ACIES — Distribution-free risk control (Learn-then-Test)

The question a deployer asks is not "what is the best lambda" but "how far may the accelerated
system stray from the full-cost system, and can you promise it?". This module answers with a
finite-sample guarantee, following the Learn-then-Test framework (Angelopoulos, Bates, Candes,
Jordan, Lei, 2021) that CALM (Schuster et al., 2022) and early-exit risk control (Jazbec et al.,
2024) apply to layer-wise early exits of models one can modify. Here it certifies the setting of
a controller that drives a black-box model through cheaper or dearer *input* actions.

Risk. For a controller setting lambda and an input x, take a loss l(x; lambda) in [0, 1] — for
example 1[decision(lambda) != decision of the reference (full-cost) system]. The risk is
R(lambda) = E[l]. With l a disagreement indicator, accuracy(lambda) >= accuracy(reference) - R.
No labels are needed: the reference system labels the calibration data.

Guarantee. Given n exchangeable calibration inputs, a tolerance alpha and a confidence delta,
`certify` returns a set of settings such that, with probability >= 1 - delta over the calibration
sample, EVERY returned setting has R(lambda) <= alpha. Each setting is tested against
H_lambda: R(lambda) > alpha with a valid p-value; family-wise error is controlled by
  * fixed-sequence testing (settings tested in a pre-specified order, stop at the first failure), or
  * Bonferroni (any order, alpha-spending delta / m).

Validity needs (i) calibration and test inputs exchangeable, and (ii) the order / grid fixed
without looking at the calibration losses (choose it on separate data). Anything used to fit the
controller (channels, predictors) must also come from data independent of the calibration set.

Pure Python; exact binomial tails via log-gamma, no dependencies.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


def _log_binom_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(Bin(n, p) <= k), summed in log-space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    logs = [_log_binom_pmf(i, n, p) for i in range(k + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in logs))


def binary_pvalue(n_bad: int, n: int, alpha: float) -> float:
    """
    Exact p-value for H: R > alpha with 0/1 losses. Under H the number of bad calibration inputs
    stochastically dominates Bin(n, alpha), so P(Bin(n, alpha) <= n_bad) is super-uniform.
    """
    return binom_cdf(n_bad, n, alpha)


def _h1(a: float, b: float) -> float:
    """KL divergence between Bernoulli(a) and Bernoulli(b)."""
    def term(x, y):
        return 0.0 if x == 0.0 else x * math.log(x / y)
    return term(a, b) + term(1.0 - a, 1.0 - b)


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """
    Hoeffding-Bentkus p-value for H: R > alpha, valid for any loss in [0, 1]
    (Bates et al., 2021): min( exp(-n h1(min(r, alpha), alpha)),  e * P(Bin(n, alpha) <= ceil(n r)) ).
    """
    if risk_hat >= alpha:
        return 1.0
    hoeff = math.exp(-n * _h1(min(risk_hat, alpha), alpha))
    bentkus = math.e * binom_cdf(math.ceil(n * risk_hat - 1e-12), n, alpha)
    return min(1.0, hoeff, bentkus)


def risk_pvalue(losses: Sequence[float], alpha: float) -> float:
    """p-value for H: R > alpha, exact for 0/1 losses and Hoeffding-Bentkus otherwise."""
    n = len(losses)
    if n == 0:
        return 1.0
    if all(l in (0, 1) for l in losses):
        return binary_pvalue(int(round(sum(losses))), n, alpha)
    return hb_pvalue(sum(losses) / n, n, alpha)


@dataclass
class Certificate:
    """Outcome of `certify`."""
    certified: List[int]                       # indices of settings guaranteed to satisfy R <= alpha
    pvalues: List[float]                       # one per setting (in setting order)
    alpha: float
    delta: float
    n: int
    method: str
    tested: List[int] = field(default_factory=list)   # order in which settings were tested

    def best(self, cost: Sequence[float]) -> Optional[int]:
        """Cheapest certified setting under the given per-setting costs (None if none certified)."""
        return min(self.certified, key=lambda j: cost[j]) if self.certified else None


def certify_pvalues(
    pvalues: Sequence[float],
    delta: float = 0.1,
    order: Optional[Sequence[int]] = None,
    method: str = "fixed_sequence",
    alpha: float = float("nan"),
    n: int = 0,
) -> Certificate:
    """
    Family-wise selection from precomputed p-values (one per setting, each valid for its own
    null H_j: "setting j violates its constraint"). Use it when the constraint is a conjunction:
    for "miss rate <= a AND false-alarm rate <= b" take p_j = max(p_miss_j, p_fa_j) — the
    intersection-union test rejects H_j only if both component nulls are rejected, so it stays
    valid at level delta whatever the dependence between the two.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if method not in ("fixed_sequence", "bonferroni"):
        raise ValueError("method must be 'fixed_sequence' or 'bonferroni'")
    m = len(pvalues)
    order = list(range(m)) if order is None else list(order)
    if sorted(order) != list(range(m)):
        raise ValueError("order must be a permutation of the settings")
    certified, tested = [], []
    if method == "fixed_sequence":
        for j in order:
            tested.append(j)
            if pvalues[j] <= delta:
                certified.append(j)
            else:
                break
    else:
        certified = [j for j in order if pvalues[j] <= delta / m]
        tested = list(order)
    return Certificate(certified, list(pvalues), alpha, delta, n, method, tested)


def certify(
    losses: Sequence[Sequence[float]],
    alpha: float,
    delta: float = 0.1,
    order: Optional[Sequence[int]] = None,
    method: str = "fixed_sequence",
) -> Certificate:
    """
    losses[i][j]: loss in [0, 1] of setting j on calibration input i.
    order: pre-specified testing order (fixed-sequence) — from most conservative (safest, dearest)
           to most aggressive; it must not depend on `losses`.
    Returns settings certified so that P(all certified settings have R <= alpha) >= 1 - delta.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if method not in ("fixed_sequence", "bonferroni"):
        raise ValueError("method must be 'fixed_sequence' or 'bonferroni'")
    n = len(losses)
    if n == 0:
        raise ValueError("need at least one calibration input")
    m = len(losses[0])
    for row in losses:
        if len(row) != m:
            raise ValueError("ragged loss matrix")
        for l in row:
            if not 0.0 <= l <= 1.0:
                raise ValueError("losses must lie in [0, 1]")
    cols = [[losses[i][j] for i in range(n)] for j in range(m)]
    pv = [risk_pvalue(c, alpha) for c in cols]
    return certify_pvalues(pv, delta, order, method, alpha, n)

