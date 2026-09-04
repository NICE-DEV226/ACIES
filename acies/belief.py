"""
ACIES — Bayesian Belief Tracker

Filtre bayésien pour estimer P(Y=1 | observations).

Caractéristiques :
- Mise à jour bayésienne exacte pour classification binaire
- Lissage anti-divergence (clipping numérique)
- Calibration temperature (compense les modèles mal calibrés)
- Historique pour diagnostic
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BeliefState:
    """
    État de croyance bayésien pour classification binaire.

    Maintient P(Y=1 | O₁:t) avec lissage numérique.
    """
    prior: float = 0.5          # P(Y=1) initial
    belief: float = 0.5         # P(Y=1 | observations courantes)
    temperature: float = 1.0    # Calibrage température (>1 = moins confiant)
    min_belief: float = 0.001   # Borne inférieure (anti-divergence)
    max_belief: float = 0.999   # Borne supérieure
    history: List[float] = field(default_factory=list)
    n_updates: int = 0

    def __post_init__(self):
        self.belief = self.prior
        self.history = [self.prior]

    def update(self, obs: int, clarity: float):
        """
        Met à jour la croyance avec une observation binaire.

        Args:
            obs: Observation (0 ou 1)
            clarity: P(obs=Y | action) — clarté de l'action utilisée
        """
        # Appliquer la température de calibrage
        p = clarity
        if self.temperature != 1.0:
            # Temperature scaling : ajuste la "force" de l'observation
            # p_eff = p^(1/T) normalisé pour rester dans [0,1]
            logit = math.log(max(p / (1 - p), 1e-10))
            logit_scaled = logit / self.temperature
            p = 1.0 / (1.0 + math.exp(-logit_scaled))

        # Bayes rule
        if obs == 1:
            p_obs_y1 = p
            p_obs_y0 = 1 - p
        else:
            p_obs_y1 = 1 - p
            p_obs_y0 = p

        p_obs = p_obs_y1 * self.belief + p_obs_y0 * (1 - self.belief)

        if p_obs < 1e-15:
            return  # Observation impossible — ne pas mettre à jour

        posterior = (p_obs_y1 * self.belief) / p_obs

        # Lissage numérique
        self.belief = max(self.min_belief, min(self.max_belief, posterior))
        self.history.append(self.belief)
        self.n_updates += 1

    def update_continuous(self, log_likelihood_ratio: float):
        """
        Mise à jour directe par log-likelihood ratio (pour observations continues).

        LLR = log P(o|Y=1) / P(o|Y=0)
        B' = 1 / (1 + (1-B)/B * exp(-LLR))
        """
        odds = self.belief / (1 - self.belief)
        new_odds = odds * math.exp(log_likelihood_ratio)
        self.belief = new_odds / (1 + new_odds)
        self.belief = max(self.min_belief, min(self.max_belief, self.belief))
        self.history.append(self.belief)
        self.n_updates += 1

    @property
    def risk(self) -> float:
        """Bayes risk (0-1 loss amplifié)."""
        return 10.0 * min(self.belief, 1 - self.belief)

    @property
    def risk_squared(self) -> float:
        """Bayes risk (squared error)."""
        return 20.0 * self.belief * (1 - self.belief)

    @property
    def risk_log(self) -> float:
        """Bayes risk (log loss)."""
        p = max(1e-10, min(1 - 1e-10, self.belief))
        h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        return 10.0 * h / math.log(2)

    @property
    def entropy(self) -> float:
        """Entropie de la croyance (incertitude)."""
        p = max(1e-10, min(1 - 1e-10, self.belief))
        return -(p * math.log(p) + (1 - p) * math.log(1 - p))

    @property
    def confidence(self) -> float:
        """Confiance dans la décision (0=incertain, 1=certain)."""
        return max(self.belief, 1 - self.belief)

    @property
    def decision(self) -> int:
        """Décision optimale (0 ou 1)."""
        return 0 if self.belief < 0.5 else 1

    @property
    def is_confident(self) -> float:
        """Seuil de confiance atteint ?"""
        return self.confidence

    def risk_after_action(self, clarity: float) -> float:
        """
        Estime le risque attendu après une observation avec cette clarté.
        Utilisé pour calculer ΔR sans exécuter l'action.
        """
        expected_risk = 0.0
        for obs in [0, 1]:
            if obs == 1:
                p_obs = clarity * self.belief + (1 - clarity) * (1 - self.belief)
                new_belief = (clarity * self.belief) / max(p_obs, 1e-15)
            else:
                p_obs = (1 - clarity) * self.belief + clarity * (1 - self.belief)
                new_belief = ((1 - clarity) * self.belief) / max(p_obs, 1e-15)

            new_belief = max(self.min_belief, min(self.max_belief, new_belief))
            risk = 10.0 * min(new_belief, 1 - new_belief)
            expected_risk += p_obs * risk

        return expected_risk

    def delta_risk(self, clarity: float) -> float:
        """
        Réduction de risque attendue pour une action avec cette clarté.
        ΔR = R(B) - E[R(B')]
        """
        return self.risk - self.risk_after_action(clarity)

    def delta_risk_efficiency(self, clarity: float, cost: float) -> float:
        """
        Efficacité de réduction de risque : ΔR / coût.
        C'est le score utilisé par APC pour sélectionner l'action.
        """
        if cost <= 0:
            return 0.0
        return self.delta_risk(clarity) / cost

    def reset(self):
        """Remet la croyance au prior."""
        self.belief = self.prior
        self.history = [self.prior]
        self.n_updates = 0

    def summary(self) -> dict:
        return {
            "belief": round(self.belief, 4),
            "risk": round(self.risk, 4),
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "n_updates": self.n_updates,
            "entropy": round(self.entropy, 4),
        }
