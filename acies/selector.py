"""
ACIES — Contextual resolution selector

The original controller keeps ONE clarity number per action, learned as if every
observation were an independent draw. Real perception is not like that: running the same
model twice at the same resolution returns the same answer, and an image that is hard at
64p is usually hard at 224p. Repeating actions therefore buys no information, and a
context-free clarity table cannot tell an easy image from a hard one.

This module replaces that table with a *contextual* model. After one cheap pass, cheap
label-free features of what the model saw (how many detections, how confident, how many
borderline, how small) predict the quality each candidate action would reach on THIS image.
The next action is then chosen by utility:

    a* = argmax_a   Q̂(a | x)  -  mu * extra_cost(a)

`mu` is the price of cost in quality units; sweeping it traces the cost/quality curve.

Pure Python (no dependencies). Ridge regression on standardised degree-2 features, solved
with Gauss-Jordan; a model with ~20 features trains on thousands of samples in seconds and
predicts with a dot product.

    sel = ResolutionSelector.fit(X, Q, action_names, ridge=30.0)  # offline, with labels
    q = sel.predict(x)                                            # deployment, label-free
    k = sel.select(x, extra_costs, mu=0.002)
    sel.save("selector.json"); sel = ResolutionSelector.load("selector.json")
"""

import json
import math
from typing import List, Sequence, Tuple


def _poly2(v: Sequence[float]) -> List[float]:
    out = list(v)
    n = len(v)
    for i in range(n):
        for j in range(i, n):
            out.append(v[i] * v[j])
    return out


def _solve(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """Solve A X = B (A symmetric positive definite, small) by Gauss-Jordan with pivoting."""
    n, m = len(a), len(b[0])
    aug = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            raise ValueError("singular system: increase `ridge`")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv = 1.0 / aug[col][col]
        aug[col] = [x * inv for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:n + m] for row in aug]


class ResolutionSelector:
    """Predicts per-action quality from context features and picks by cost-aware utility."""

    def __init__(self, action_names, mean, std, weights, ridge):
        self.action_names = list(action_names)
        self.mean = mean          # standardisation of the degree-2 features
        self.std = std
        self.weights = weights    # [n_features + 1][n_actions] (last row = bias)
        self.ridge = ridge

    # -- training ---------------------------------------------------------
    @classmethod
    def fit(
        cls,
        X: Sequence[Sequence[float]],
        Q: Sequence[Sequence[float]],
        action_names: Sequence[str],
        ridge: float = 30.0,
    ) -> "ResolutionSelector":
        """
        X: context features per sample (label-free, available at deployment)
        Q: measured quality of each action on that sample (needs ground truth; offline)
        """
        if len(X) != len(Q) or not X:
            raise ValueError("X and Q must be non-empty and of equal length")
        if len(Q[0]) != len(action_names):
            raise ValueError("Q must have one column per action")
        if ridge <= 0:
            raise ValueError("ridge must be > 0")
        P = [_poly2(x) for x in X]
        n, d = len(P), len(P[0])
        mean = [sum(row[j] for row in P) / n for j in range(d)]
        std = [math.sqrt(sum((row[j] - mean[j]) ** 2 for row in P) / n) + 1e-6 for j in range(d)]
        Z = [[(row[j] - mean[j]) / std[j] for j in range(d)] + [1.0] for row in P]
        dim = d + 1
        ata = [[0.0] * dim for _ in range(dim)]
        atq = [[0.0] * len(action_names) for _ in range(dim)]
        for z, q in zip(Z, Q):
            for i in range(dim):
                zi = z[i]
                row = ata[i]
                for j in range(i, dim):
                    row[j] += zi * z[j]
                atq_i = atq[i]
                for k, qk in enumerate(q):
                    atq_i[k] += zi * qk
        for i in range(dim):
            for j in range(i):
                ata[i][j] = ata[j][i]
            ata[i][i] += ridge
        return cls(action_names, mean, std, _solve(ata, atq), ridge)

    # -- inference --------------------------------------------------------
    def predict(self, x: Sequence[float]) -> List[float]:
        """Predicted quality of every action on this context (clipped to [0, 1])."""
        p = _poly2(x)
        if len(p) != len(self.mean):
            raise ValueError(f"expected {int(self._n_raw())} raw features, got {len(x)}")
        z = [(p[j] - self.mean[j]) / self.std[j] for j in range(len(p))] + [1.0]
        out = []
        for k in range(len(self.action_names)):
            q = sum(z[i] * self.weights[i][k] for i in range(len(z)))
            out.append(min(max(q, 0.0), 1.0))
        return out

    def _n_raw(self) -> float:
        return (math.sqrt(8 * len(self.mean) + 1) - 1) / 2

    def select(self, x: Sequence[float], extra_costs: Sequence[float], mu: float) -> int:
        """Index of the action maximising predicted quality minus mu * extra cost."""
        if mu < 0:
            raise ValueError("mu must be >= 0")
        q = self.predict(x)
        return max(range(len(q)), key=lambda k: (q[k] - mu * extra_costs[k], -extra_costs[k]))

    # -- persistence ------------------------------------------------------
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"action_names": self.action_names, "mean": self.mean, "std": self.std,
                       "weights": self.weights, "ridge": self.ridge}, f)

    @classmethod
    def load(cls, path: str) -> "ResolutionSelector":
        with open(path) as f:
            d = json.load(f)
        return cls(d["action_names"], d["mean"], d["std"], d["weights"], d["ridge"])


def detection_features(
    dets: Sequence[Tuple[float, float, float, float, float, int]],
    image_wh: Tuple[int, int],
) -> List[float]:
    """
    Label-free context features of one detection set.

    dets: (x1, y1, x2, y2, conf, cls) for every detection kept with a LOW confidence floor
          (0.001-0.05), so that uncertain detections are visible — they are the signal that
          a higher resolution would change the answer.
    """
    w, h = image_wh
    img_area = float(w * h)
    confs = sorted((d[4] for d in dets), reverse=True)
    hi = [d for d in dets if d[4] >= 0.25]
    n = len(hi)
    low = [d[4] for d in dets if 0.05 <= d[4] < 0.25]
    rel = [math.sqrt(max((d[2] - d[0]) * (d[3] - d[1]), 1.0) / img_area) for d in hi]

    def count(pred):
        return sum(1 for r in rel if pred(r)) / 5.0

    return [
        math.log1p(n),
        math.log1p(sum(1 for d in dets if d[4] >= 0.5)),
        math.log1p(sum(1 for d in dets if d[4] >= 0.75)),
        math.log1p(len(low)),
        sum(low) / 5.0,
        confs[0] if confs else 0.0,
        sum(confs[:3]) / len(confs[:3]) if confs else 0.0,
        sum(d[4] for d in hi) / n if n else 0.0,
        count(lambda r: r < 0.08),
        count(lambda r: 0.08 <= r < 0.25),
        count(lambda r: r >= 0.25),
        min(rel) if rel else 0.0,
        sum(rel) / n if n else 0.0,
        len({d[5] for d in hi}) / 5.0,
        sum(1 for d in dets if 0.15 <= d[4] < 0.5) / 5.0,
    ]
