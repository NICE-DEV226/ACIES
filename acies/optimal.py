"""
ACIES — Optimal sequential policy (exact baseline)

Under the generative model ACIES assumes — Y ~ Bernoulli(prior); action a returns
o with P(o = Y) = clarity_a, i.i.d. given Y; action a costs C_a — the best possible
controller is the solution of a finite-horizon optimal stopping problem over the
belief b = P(Y=1):

    V_T(b) = λ · min(b, 1-b)
    V_t(b) = min( λ · min(b, 1-b) ,                        # stop and decide
                  min_a  C_a + Σ_o P(o | b, a) V_{t+1}(b'(o, a)) )   # observe more

λ is the price of one error, in the same units as cost. Sweeping λ traces the
optimal cost/error frontier: no controller that follows this model can beat it,
so it is the reference against which any heuristic (including APCController)
must be measured.

Pure Python, no dependencies. The belief axis is discretised on a uniform grid
with linear interpolation; policies are then evaluated exactly on the (small)
tree of reachable beliefs, so reported cost/error are those of the policy, not
of the grid.

Usage:
    from acies import build_standard_actions, HardwareProfile
    from acies.optimal import solve_optimal, min_cost_for_error

    clarity = {"64p": 0.55, ..., "crop_224": 0.85}
    policy = solve_optimal(build_standard_actions(), clarity, HardwareProfile.default(),
                           error_cost=1000, horizon=6)
    print(policy.evaluate())              # OptimalEvaluation(cost=..., error=..., steps=...)
    policy.action(belief=0.5, step=0)     # first Action, or None to stop

    best = min_cost_for_error(actions, clarity, hw, target_error=0.02, horizon=6)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .actions import Action, HardwareProfile

ClarityMap = Union[Dict[str, float], Callable[[Action], float]]


@dataclass(frozen=True)
class OptimalEvaluation:
    """Exact expected performance of a policy from a given prior."""
    cost: float    # expected total cost
    error: float   # expected error probability (Bayes-optimal decision at stop)
    steps: float   # expected number of observations


def _clarity_of(clarities: ClarityMap, action: Action) -> float:
    c = clarities(action) if callable(clarities) else clarities[action.name]
    if not 0.0 < c < 1.0:
        raise ValueError(f"clarity of {action.name} must be in (0, 1), got {c}")
    return c


def _interp(values: List[float], x: float) -> float:
    n = len(values) - 1
    pos = min(max(x, 0.0), 1.0) * n
    i = int(pos)
    if i >= n:
        return values[n]
    frac = pos - i
    return values[i] * (1.0 - frac) + values[i + 1] * frac


def _outcomes(b: float, c: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """((P(o=1), b'|o=1), (P(o=0), b'|o=0)) for a symmetric channel of clarity c."""
    p1 = c * b + (1.0 - c) * (1.0 - b)
    p0 = 1.0 - p1
    b1 = c * b / p1 if p1 > 0.0 else b
    b0 = (1.0 - c) * b / p0 if p0 > 0.0 else b
    return (p1, b1), (p0, b0)


class OptimalPolicy:
    """Exact finite-horizon optimal policy for the ACIES observation model."""

    def __init__(
        self,
        actions: Sequence[Action],
        clarities: List[float],
        costs: List[float],
        error_cost: float,
        horizon: int,
        grid_size: int,
    ):
        self.actions = list(actions)
        self.clarities = clarities
        self.costs = costs
        self.error_cost = error_cost
        self.horizon = horizon
        self.grid_size = grid_size
        self._values = self._solve()

    def _solve(self) -> List[List[float]]:
        g = self.grid_size
        grid = [i / (g - 1) for i in range(g)]
        stop = [self.error_cost * min(b, 1.0 - b) for b in grid]
        values: List[List[float]] = [[] for _ in range(self.horizon + 1)]
        values[self.horizon] = stop
        for t in range(self.horizon - 1, -1, -1):
            nxt = values[t + 1]
            cur = []
            for i, b in enumerate(grid):
                best = stop[i]
                for c, cost in zip(self.clarities, self.costs):
                    (p1, b1), (p0, b0) = _outcomes(b, c)
                    q = cost + p1 * _interp(nxt, b1) + p0 * _interp(nxt, b0)
                    if q < best:
                        best = q
                cur.append(best)
            values[t] = cur
        return values

    def action(self, belief: float, step: int) -> Optional[Action]:
        """Optimal action at `step` given `belief`, or None to stop and decide."""
        stop = self.error_cost * min(belief, 1.0 - belief)
        if step >= self.horizon:
            return None
        nxt = self._values[step + 1]
        best_q, best_a = stop, None
        for a, c, cost in zip(self.actions, self.clarities, self.costs):
            (p1, b1), (p0, b0) = _outcomes(belief, c)
            q = cost + p1 * _interp(nxt, b1) + p0 * _interp(nxt, b0)
            if q < best_q:
                best_q, best_a = q, a
        return best_a

    def evaluate(self, prior: float = 0.5) -> OptimalEvaluation:
        """Exact expected cost / error / steps of this policy (tree enumeration)."""
        by_id = {a.id: i for i, a in enumerate(self.actions)}

        def rec(b: float, t: int, spent: float, prob: float):
            a = self.action(b, t)
            if a is None:
                return prob * spent, prob * min(b, 1.0 - b), prob * t
            i = by_id[a.id]
            tot = [0.0, 0.0, 0.0]
            for p, nb in _outcomes(b, self.clarities[i]):
                if p <= 0.0:
                    continue
                r = rec(nb, t + 1, spent + self.costs[i], prob * p)
                for k in range(3):
                    tot[k] += r[k]
            return tot[0], tot[1], tot[2]

        cost, err, steps = rec(prior, 0, 0.0, 1.0)
        return OptimalEvaluation(cost=cost, error=err, steps=steps)


def solve_optimal(
    actions: Sequence[Action],
    clarities: ClarityMap,
    hardware: HardwareProfile,
    error_cost: float,
    horizon: int = 6,
    grid_size: int = 1001,
) -> OptimalPolicy:
    """
    Solve the optimal stopping problem for a given price of error.

    Args:
        actions: action space
        clarities: {action.name: P(correct)} or callable(Action) -> P(correct)
        hardware: profile used to price actions (Action.cost)
        error_cost: λ, cost of one wrong decision, in cost units
        horizon: maximum number of observations
        grid_size: belief discretisation (>= 101 recommended)
    """
    if error_cost <= 0:
        raise ValueError("error_cost must be > 0")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if grid_size < 3:
        raise ValueError("grid_size must be >= 3")
    cl = [_clarity_of(clarities, a) for a in actions]
    costs = [a.cost(hardware) for a in actions]
    return OptimalPolicy(actions, cl, costs, error_cost, horizon, grid_size)


def frontier(
    actions: Sequence[Action],
    clarities: ClarityMap,
    hardware: HardwareProfile,
    error_costs: Sequence[float],
    horizon: int = 6,
    grid_size: int = 1001,
    prior: float = 0.5,
) -> List[Tuple[float, OptimalEvaluation]]:
    """Optimal (cost, error) frontier: one point per error price λ."""
    return [
        (lam, solve_optimal(actions, clarities, hardware, lam, horizon, grid_size).evaluate(prior))
        for lam in error_costs
    ]


def min_cost_for_error(
    actions: Sequence[Action],
    clarities: ClarityMap,
    hardware: HardwareProfile,
    target_error: float,
    horizon: int = 6,
    grid_size: int = 1001,
    prior: float = 0.5,
    iterations: int = 40,
) -> OptimalEvaluation:
    """
    Cheapest achievable expected cost with expected error <= target_error.

    Error is non-increasing in λ, so bisect on log λ. Raises ValueError if the
    target is unreachable within `horizon` observations.
    """
    lo, hi = 1e-3, 1e9
    best = solve_optimal(actions, clarities, hardware, hi, horizon, grid_size).evaluate(prior)
    if best.error > target_error:
        raise ValueError(
            f"target_error={target_error} unreachable within horizon={horizon} "
            f"(best achievable {best.error:.4f})"
        )
    for _ in range(iterations):
        mid = (lo * hi) ** 0.5
        ev = solve_optimal(actions, clarities, hardware, mid, horizon, grid_size).evaluate(prior)
        if ev.error <= target_error:
            hi, best = mid, ev
        else:
            lo = mid
        if hi / lo < 1.0005:
            break
    return best
