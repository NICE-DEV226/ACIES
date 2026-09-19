"""
ACIES — Certified two-stage cascade

The most reliable way to cut the cost of a black-box perception model is a cascade: run a cheap
pass, accept its answer when it is confident, and only otherwise pay for the expensive pass. What
is usually missing is a promise. Thresholds are tuned until the accuracy "looks fine" on some
validation set, and nothing says how far the accelerated system can stray from the full one on
new inputs. This module calibrates the two thresholds with `acies.risk` (Learn-then-Test) so that
they come with a finite-sample guarantee.

    first stage : cheap pass  -> score s in [0, 1] (e.g. top confidence of the class of interest)
                  s >= hi  -> answer 1        s < lo -> answer 0        otherwise escalate
    second stage: the reference (expensive) pass decides.

Guarantee (with probability >= 1 - delta over the calibration sample, for exchangeable inputs):
  risk="disagree"     P(answer != reference answer) <= eps
  risk="conditional"  P(answer = 0 | reference = 1) <= eps_miss   AND
                      P(answer = 1 | reference = 0) <= eps_fa
No labels are needed: the reference system labels the calibration data. If no setting can be
certified with the data at hand (too few examples for the tolerance), `calibrate_cascade`
returns a cascade that always runs the reference — the honest answer is "no saving proven".

Three disjoint samples are used, as the guarantee requires: `train` fixes the grid order,
`cert` certifies, and anything else (the deployed traffic) is where the promise applies.

    cal = calibrate_cascade(train, cert, costs=(31.0, 96.0), risk="disagree", eps=0.02, delta=0.1)
    answer, escalated = cal.run(first_score, second=lambda: reference_answer())

    # several candidate first stages, one guarantee:
    cal = calibrate_ladders([Ladder(tr160, ce160, (12.8, 96.0), "160"), Ladder(tr224, ce224, (31.0, 96.0), "224")], eps=0.02)
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .risk import Certificate, certify, certify_pvalues, risk_pvalue


@dataclass
class CascadeSample:
    """Calibration data: per input, the first-stage score, the reference-stage answer, and the
    answer the reference system gives (identical to `second` when the second stage IS the
    reference; kept separate so a cheaper-than-reference second stage can also be certified)."""
    first_score: Sequence[float]
    second: Sequence[int]
    reference: Sequence[int]

    def __post_init__(self):
        if not (len(self.first_score) == len(self.second) == len(self.reference)):
            raise ValueError("first_score, second and reference must have equal length")


@dataclass
class Ladder:
    """One candidate first stage / second stage pair with its calibration data and costs."""
    train: CascadeSample
    cert: CascadeSample
    costs: Tuple[float, float]      # (first stage, second stage)
    name: str = ""


@dataclass
class CalibratedCascade:
    hi: float
    lo: float
    certified: bool                 # False: nothing certified, the cascade always runs the second stage
    certificate: Optional[Certificate]
    costs: Tuple[float, float]      # (first stage, second stage) of the chosen ladder
    expected_cost: float            # mean cost on the certification sample
    n_settings: int
    risk: str
    tolerance: Tuple[float, ...]
    ladder: int = 0                 # index of the chosen first/second stage pair
    ladder_name: str = ""

    def run(self, first_score: float, second: Callable[[], int]) -> Tuple[int, bool]:
        """Decide one input. `first_score` comes from the chosen ladder's first stage; `second` is
        called only when the first stage is not confident."""
        if not self.certified:
            return int(second()), True
        if first_score >= self.hi:
            return 1, False
        if first_score < self.lo:
            return 0, False
        return int(second()), True


def _decide(s: float, second: int, hi: float, lo: float) -> int:
    return 1 if s >= hi else (0 if s < lo else second)


def calibrate_cascade(
    train: CascadeSample,
    cert: CascadeSample,
    costs: Tuple[float, float],
    **kw,
) -> CalibratedCascade:
    """Single first/second stage pair; see `calibrate_ladders` for the options."""
    return calibrate_ladders([Ladder(train, cert, costs)], **kw)


def calibrate_ladders(
    ladders: Sequence[Ladder],
    risk: str = "disagree",
    eps: float = 0.02,
    eps_miss: float = 0.10,
    eps_fa: float = 0.03,
    delta: float = 0.1,
    grid_hi: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.8),
    grid_lo: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1),
    min_conditional_examples: int = 20,
) -> CalibratedCascade:
    """
    Choose a first stage and its thresholds (hi, lo) with a distribution-free guarantee.

    Every (ladder, hi, lo) triple is a setting. The settings are ordered from the safest to the
    riskiest on the ladders' `train` samples (independent of `cert`), tested in that single fixed
    sequence with Learn-then-Test (stop at the first that cannot be certified), and the cheapest
    certified setting on `cert` is returned. Trying several first stages therefore costs nothing
    in guarantee — it only changes which settings are in the sequence.
    """
    if risk not in ("disagree", "conditional"):
        raise ValueError("risk must be 'disagree' or 'conditional'")
    if not ladders:
        raise ValueError("need at least one ladder")
    n_tr, n_ce = len(ladders[0].train.first_score), len(ladders[0].cert.first_score)
    for lad in ladders:
        if not lad.costs[0] >= 0 or not lad.costs[1] > 0:
            raise ValueError("costs must be (first >= 0, second > 0)")
        if not len(lad.train.first_score) or not len(lad.cert.first_score):
            raise ValueError("train and cert must be non-empty")
        if len(lad.train.first_score) != n_tr or len(lad.cert.first_score) != n_ce:
            raise ValueError("all ladders must be evaluated on the same train and cert inputs")
        if (list(lad.train.reference) != list(ladders[0].train.reference)
                or list(lad.cert.reference) != list(ladders[0].cert.reference)):
            raise ValueError("all ladders must share the same reference answers")
    settings = [(k, h, l) for k in range(len(ladders)) for h in grid_hi for l in grid_lo if l < h]
    if not settings:
        raise ValueError("empty grid")
    tol = (eps,) if risk == "disagree" else (eps_miss, eps_fa)

    def dec(sample, j):
        k, h, l = settings[j]
        return [_decide(s, d, h, l) for s, d in zip(sample.first_score, sample.second)]

    def cost(which, j):
        k, h, l = settings[j]
        lad = ladders[k]
        sample = lad.train if which == "train" else lad.cert
        esc = sum(1 for s in sample.first_score if l <= s < h)
        return lad.costs[0] + lad.costs[1] * esc / len(sample.first_score)

    fb_cost = min(l.costs[1] for l in ladders)
    fallback = CalibratedCascade(float("inf"), float("-inf"), False, None, ladders[0].costs, fb_cost,
                                 len(settings), risk, tol)
    ref_tr, ref_ce = ladders[0].train.reference, ladders[0].cert.reference
    m = len(settings)
    cost_tr = [cost("train", j) for j in range(m)]
    if risk == "disagree":
        risk_tr = [sum(a != r for a, r in zip(dec(ladders[settings[j][0]].train, j), ref_tr)) / len(ref_tr)
                   for j in range(m)]
        order = sorted(range(m), key=lambda j: (risk_tr[j], cost_tr[j]))
        cols = [dec(ladders[settings[j][0]].cert, j) for j in range(m)]
        L = [[float(cols[j][i] != ref_ce[i]) for j in range(m)] for i in range(len(ref_ce))]
        cr = certify(L, alpha=eps, delta=delta, order=order)
    else:
        pos_c = [i for i, r in enumerate(ref_ce) if r == 1]
        neg_c = [i for i, r in enumerate(ref_ce) if r == 0]
        pos_t = [i for i, r in enumerate(ref_tr) if r == 1]
        neg_t = [i for i, r in enumerate(ref_tr) if r == 0]
        if (len(pos_c) < min_conditional_examples or len(neg_c) < min_conditional_examples
                or not pos_t or not neg_t):
            return fallback
        score = []
        for j in range(m):
            d = dec(ladders[settings[j][0]].train, j)
            score.append(sum(d[i] == 0 for i in pos_t) / len(pos_t) / eps_miss
                         + sum(d[i] == 1 for i in neg_t) / len(neg_t) / eps_fa)
        order = sorted(range(m), key=lambda j: (score[j], cost_tr[j]))
        pv = []
        for j in range(m):
            d = dec(ladders[settings[j][0]].cert, j)
            pv.append(max(risk_pvalue([float(d[i] == 0) for i in pos_c], eps_miss),
                          risk_pvalue([float(d[i] == 1) for i in neg_c], eps_fa)))
        cr = certify_pvalues(pv, delta=delta, order=order, n=len(ref_ce))
    if not cr.certified:
        fallback.certificate = cr
        return fallback
    best = min(cr.certified, key=lambda j: cost("cert", j))
    k, h, l = settings[best]
    return CalibratedCascade(h, l, True, cr, ladders[k].costs, cost("cert", best), m, risk, tol,
                             ladder=k, ladder_name=ladders[k].name)
