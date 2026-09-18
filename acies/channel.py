"""
ACIES — Channel controller (multi-outcome evidence, value of information per cost)

What the original controller gets wrong on real perception
----------------------------------------------------------
1. One number per action ("clarity") models each observation as a coin flip that is right
   with probability p, independent of every other observation. A deterministic model run
   twice at the same resolution returns the same answer: repeating an action adds no
   information, yet the controller counts it as fresh evidence and pays for it.
2. It reduces what the perception model returned to a hard 0/1 vote. A detector says far
   more than that: "0.02" and "0.24" are both "no" but mean very different things, and
   a borderline "0.3" is exactly the case that deserves a second, more expensive look.

The model here
--------------
Each action is a *channel*: the perception output is quantised into K outcomes
(the caller chooses the bins, e.g. bins of the top confidence) and the action is described by
its likelihood table  P(outcome | Y, action)  learned from labelled examples (Dirichlet
posterior, so uncertainty shrinks with data). Given belief b = P(Y=1):

    posterior after outcome o :   logit b' = carry*logit b + (1-carry)*logit prior + tau*LLR(o)
    value of information      :   VOI(a) = risk(b) - E_o[ risk(b'_o) ],   risk(b) = min(b, 1-b)
    plan                      :   dynamic programming over (belief, lowest action still allowed)
                                  (config.planning=False: greedy argmax VOI(a)/cost(a) instead)
    stop                      :   when continuing is not worth its cost:  error_cost * risk is
                                  lower than any plan's expected cost + residual risk

`error_cost` is the price of one wrong decision in cost units (the single knob that trades
accuracy for cost). Errors of different actions on the same image are correlated (hard at
160p usually means hard at 320p), so summing their evidence over-counts it: `carry` in [0, 1]
is the share of previous evidence kept when a new outcome arrives (1 = independent actions,
0 = the latest outcome replaces earlier ones — measured best for a deterministic detector
whose higher-resolution output supersedes the cheaper one). Fit `carry` on held-out data;
leave `tau` = 1 (a second discount on top of `carry` was measured to hurt).

With K = 2 outcomes and no discount this reduces to the original binary-symmetric model,
so the classic results still hold; the novelty is using the whole output of the perception
model and never paying twice for the same answer.

    learner = ChannelLearner(n_actions=len(actions), n_outcomes=8)
    learner.fit(outcomes, labels)            # outcomes[i][a] in 0..K-1, labels[i] in {0, 1}
    ctl = ChannelController(actions, learner, ChannelConfig(error_cost=300))
    ctl.begin()
    while (a := ctl.next_action()) is not None:
        ctl.observe(a, my_model_outcome(a))   # no ground truth needed
    res = ctl.finish()                        # res.decision, res.confidence, res.total_cost
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .actions import Action, HardwareProfile


@dataclass
class ChannelConfig:
    error_cost: float = 300.0      # cost units per unit of error probability (accuracy/cost knob)
    prior: Optional[float] = None  # P(Y=1); None = class frequency seen by the learner
    tau: float = 1.0               # weight of NEW evidence in (0, 1]
    carry: float = 1.0             # share of previous evidence kept, in [0, 1]. 1 = naive Bayes
                                   # (independent actions); 0 = only the latest outcome counts;
                                   # in between discounts redundant evidence (correlated errors)
    max_steps: int = 8
    confidence_threshold: float = 1.0   # optional early stop (1.0 = disabled)
    abstention_confidence: float = 0.0  # decide -1 (abstain) when confidence is below this
    hardware: HardwareProfile = field(default_factory=HardwareProfile.default)
    min_belief: float = 1e-4
    planning: bool = True          # plan the whole escalation (DP) instead of greedy VOI/cost
    horizon: int = 4               # max number of observations the plan looks ahead
    grid: int = 201                # belief discretisation of the plan


class ChannelLearner:
    """Likelihood table P(outcome | Y, action) with a Dirichlet(alpha) posterior per row."""

    def __init__(self, n_actions: int, n_outcomes: int, alpha: float = 1.0):
        if n_actions < 1 or n_outcomes < 2:
            raise ValueError("need n_actions >= 1 and n_outcomes >= 2")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        self.n_actions, self.n_outcomes, self.alpha = n_actions, n_outcomes, alpha
        # counts[a][y][o]
        self.counts = [[[0.0] * n_outcomes for _ in range(2)] for _ in range(n_actions)]
        self._cache = {}

    def feedback(self, action_id: int, label: int, outcome: int):
        """One labelled example: action `action_id` produced `outcome` on an image of class `label`."""
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        if not 0 <= outcome < self.n_outcomes:
            raise ValueError(f"outcome must be in [0, {self.n_outcomes})")
        self.counts[action_id][label][outcome] += 1.0
        self._cache.clear()

    def prior(self) -> float:
        """Class frequency P(Y=1) seen so far (Laplace-smoothed, clipped away from 0/1)."""
        pos = neg = 0.0
        for a in range(self.n_actions):
            pos += sum(self.counts[a][1])
            neg += sum(self.counts[a][0])
        p = (pos + 1.0) / (pos + neg + 2.0)
        return min(max(p, 0.005), 0.995)

    def fit(self, outcomes: Sequence[Sequence[int]], labels: Sequence[int]):
        """Batch fit: outcomes[i][a] = outcome of action a on example i."""
        for row, y in zip(outcomes, labels):
            for a, o in enumerate(row):
                self.feedback(a, int(y), int(o))
        return self

    def likelihood(self, action_id: int, label: int) -> List[float]:
        key = ("p", action_id, label)
        if key not in self._cache:
            row = self.counts[action_id][label]
            tot = sum(row) + self.alpha * self.n_outcomes
            self._cache[key] = [(c + self.alpha) / tot for c in row]
        return self._cache[key]

    def llr(self, action_id: int) -> List[float]:
        """log P(o|1) / P(o|0) for every outcome."""
        key = ("l", action_id)
        if key not in self._cache:
            p1, p0 = self.likelihood(action_id, 1), self.likelihood(action_id, 0)
            self._cache[key] = [math.log(a / b) for a, b in zip(p1, p0)]
        return self._cache[key]

    @property
    def version(self) -> int:
        """Changes whenever the tables change (planners use it to know when to re-plan)."""
        return int(sum(sum(sum(r) for r in a) for a in self.counts))

    def n_examples(self, action_id: int) -> int:
        return int(sum(self.counts[action_id][0]) + sum(self.counts[action_id][1]))


class _Plan:
    """
    Finite-horizon optimal escalation plan over (belief, lowest action index still allowed).

    Perception is deterministic, so an action is never worth repeating, and going back to a
    cheaper action after a dearer one adds nothing: plans move upward through the action list
    (ordered by cost). That makes the state (belief, j) and the DP small:

        V_t(b, j) = min( error_cost * min(b, 1-b) ,
                         min_{a >= j}  cost(a) + sum_o P(o | b, a) * V_{t+1}(b'(b, a, o), a + 1) )
    """

    def __init__(self, ctl: "ChannelController"):
        n, K, G = len(ctl.actions), ctl.learner.n_outcomes, ctl.config.grid
        H = ctl.config.horizon
        lam = ctl.config.error_cost
        self.grid = [ctl.config.min_belief + (1.0 - 2 * ctl.config.min_belief) * i / (G - 1)
                     for i in range(G)]
        self.costs = [ctl._cost(a) for a in ctl.actions]
        self.p = [[(ctl.learner.likelihood(a, 1), ctl.learner.likelihood(a, 0)) for a in range(n)]]
        # next belief for every (action, grid point, outcome): depends only on (b, a, o)
        self.nb = [[[ctl._posterior(b, a, o) for o in range(K)] for b in self.grid] for a in range(n)]
        self.pred = [[[b * ctl.learner.likelihood(a, 1)[o] + (1 - b) * ctl.learner.likelihood(a, 0)[o]
                       for o in range(K)] for b in self.grid] for a in range(n)]
        stop = [lam * min(b, 1.0 - b) for b in self.grid]
        # V[t][j] : list over grid; layer H is forced-stop
        self.V = [[list(stop) for _ in range(n + 1)] for _ in range(H + 1)]
        for t in range(H - 1, -1, -1):
            for j in range(n, -1, -1):
                cur = []
                for i in range(G):
                    best = stop[i]
                    for a in range(j, n):
                        q = self.costs[a]
                        nxt = self.V[t + 1][a + 1]
                        for o in range(K):
                            po = self.pred[a][i][o]
                            if po > 0.0:
                                q += po * self._interp(nxt, self.nb[a][i][o])
                        if q < best:
                            best = q
                    cur.append(best)
                self.V[t][j] = cur
        self.stop, self.n, self.K, self.H = stop, n, K, H

    def _interp(self, values, b):
        G = len(values) - 1
        lo, hi = self.grid[0], self.grid[-1]
        pos = (min(max(b, lo), hi) - lo) / (hi - lo) * G
        i = int(pos)
        if i >= G:
            return values[G]
        f = pos - i
        return values[i] * (1 - f) + values[i + 1] * f

    def action(self, ctl: "ChannelController", b: float, t: int, j: int):
        """(index, voi) of the best next action from state (b, t, j), or (None, 0) to stop."""
        if t >= self.H or j >= self.n:
            return None, 0.0
        stop = ctl.config.error_cost * min(b, 1.0 - b)
        best_q, best_a = stop, None
        nxt_layer = self.V[t + 1]
        for a in range(j, self.n):
            p1, p0 = ctl.learner.likelihood(a, 1), ctl.learner.likelihood(a, 0)
            q = self.costs[a]
            for o in range(self.K):
                po = b * p1[o] + (1 - b) * p0[o]
                if po > 0.0:
                    q += po * self._interp(nxt_layer[a + 1], ctl._posterior(b, a, o))
            if q < best_q:
                best_q, best_a = q, a
        return best_a, (stop - best_q if best_a is not None else 0.0)


@dataclass
class ChannelStep:
    action: Action
    outcome: int
    belief_before: float
    belief_after: float
    cost: float
    voi: float


@dataclass
class ChannelResult:
    decision: int                 # 0 / 1, or -1 when abstaining
    confidence: float
    belief: float
    total_cost: float
    n_steps: int
    abstained: bool
    steps: List[ChannelStep]
    correct: Optional[bool] = None

    @property
    def actions_taken(self) -> List[str]:
        return [s.action.name for s in self.steps]


class ChannelController:
    """Sequential controller over channels; API mirrors APCController's step API."""

    def __init__(self, actions: Sequence[Action], learner: ChannelLearner,
                 config: ChannelConfig = None):
        if len(actions) != learner.n_actions:
            raise ValueError("learner and actions disagree on the number of actions")
        self.actions = list(actions)
        self._plan = None
        self._plan_key = None
        self.learner = learner
        self.config = config or ChannelConfig()
        if not 0.0 < self.config.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if not 0.0 <= self.config.carry <= 1.0:
            raise ValueError("carry must be in [0, 1]")
        self._task = None

    @property
    def prior(self) -> float:
        return self.config.prior if self.config.prior is not None else self.learner.prior()

    # ------------------------------------------------------------------ maths
    def _cost(self, a: Action) -> float:
        return a.cost(self.config.hardware)

    def _posterior(self, belief: float, action_idx: int, outcome: int) -> float:
        """
        logit b' = carry * logit b + (1 - carry) * logit prior + tau * LLR(outcome)

        carry = 1 is the exact Bayes update for independent actions; smaller values stop an
        old, cheap, correlated observation from being counted again next to a new one.
        """
        prior = self.prior
        c = self.config.carry
        lo = (c * math.log(belief / (1.0 - belief))
              + (1.0 - c) * math.log(prior / (1.0 - prior))
              + self.learner.llr(action_idx)[outcome] * self.config.tau)
        b = 1.0 / (1.0 + math.exp(-lo))
        m = self.config.min_belief
        return min(max(b, m), 1.0 - m)

    def voi(self, belief: float, action_idx: int) -> float:
        """Expected reduction of the error probability from running this action once."""
        p1 = self.learner.likelihood(action_idx, 1)
        p0 = self.learner.likelihood(action_idx, 0)
        expected = 0.0
        for o in range(self.learner.n_outcomes):
            p_o = belief * p1[o] + (1.0 - belief) * p0[o]
            if p_o <= 0.0:
                continue
            b2 = self._posterior(belief, action_idx, o)
            expected += p_o * min(b2, 1.0 - b2)
        return min(belief, 1.0 - belief) - expected

    # ------------------------------------------------------------------ step API
    def _get_plan(self) -> "_Plan":
        key = (self.learner.version, self.config.error_cost, self.config.tau, self.config.carry,
               self.config.horizon, self.config.grid, self.prior)
        if self._plan is None or self._plan_key != key:
            self._plan, self._plan_key = _Plan(self), key
        return self._plan

    def begin(self):
        self._task = {"belief": self.prior, "used": set(), "j": 0, "steps": [], "cost": 0.0,
                      "pending": None, "done": False}

    def _t(self):
        if self._task is None:
            raise RuntimeError("No task in progress: call begin() first")
        return self._task

    def next_action(self) -> Optional[Action]:
        t = self._t()
        if t["done"]:
            return None
        if t["pending"] is not None:
            raise RuntimeError("observe() must be called for the pending action first")
        b = t["belief"]
        if (len(t["steps"]) >= self.config.max_steps
                or max(b, 1.0 - b) >= self.config.confidence_threshold):
            t["done"] = True
            return None
        if self.config.planning:
            idx, gain = self._get_plan().action(self, b, len(t["steps"]), t["j"])
            if idx is None:
                t["done"] = True
                return None
            t["pending"] = (idx, gain)
            return self.actions[idx]
        best, best_score, best_voi = None, 0.0, 0.0
        for i, a in enumerate(self.actions):
            if i in t["used"]:
                continue
            gain = self.voi(b, i)
            cost = self._cost(a)
            # worth running only if the expected error reduction, priced, exceeds its cost
            if gain * self.config.error_cost <= cost:
                continue
            score = gain / max(cost, 1e-9)
            if score > best_score:
                best, best_score, best_voi = i, score, gain
        if best is None:
            t["done"] = True
            return None
        t["pending"] = (best, best_voi)
        return self.actions[best]

    def observe(self, action: Action, outcome: int) -> ChannelStep:
        t = self._t()
        if t["pending"] is None or self.actions[t["pending"][0]].id != action.id:
            raise RuntimeError("observe() must follow next_action() with the same action")
        if not 0 <= outcome < self.learner.n_outcomes:
            raise ValueError(f"outcome must be in [0, {self.learner.n_outcomes})")
        idx, voi = t["pending"]
        t["pending"] = None
        before = t["belief"]
        t["belief"] = self._posterior(before, idx, outcome)
        t["used"].add(idx)
        t["j"] = max(t["j"], idx + 1)
        cost = self._cost(action)
        t["cost"] += cost
        step = ChannelStep(action, outcome, before, t["belief"], cost, voi)
        t["steps"].append(step)
        return step

    def feedback(self, action: Action, label: int, outcome: int):
        """Late label: refine the channel of `action` (online learning)."""
        self.learner.feedback(self.actions.index(action), label, outcome)

    def finish(self, true_class: int = None) -> ChannelResult:
        t = self._t()
        b = t["belief"]
        conf = max(b, 1.0 - b)
        abstained = conf < self.config.abstention_confidence
        decision = -1 if abstained else int(b >= 0.5)
        correct = None if true_class is None else (False if abstained else decision == true_class)
        res = ChannelResult(decision, conf, b, t["cost"], len(t["steps"]), abstained, t["steps"], correct)
        self._task = None
        return res
