"""
ACIES — Clarity Learner (Thompson Sampling)

Estime en ligne la clarté de chaque action :
  p_i = P(observation correcte | action a_i)

Utilise Thompson Sampling avec un modèle Beta-Bernoulli :
  - Prior : Beta(α₀, β₀)
  - Après n_obs observations avec n_correct correctes :
    posterior = Beta(α₀ + n_correct, β₀ + n_obs - n_correct)
  - Échantillonne p_i ~ posterior pour l'exploration

Avantages :
  - Calibré naturellement (incertitude décroît avec les observations)
  - Exploration/exploitation automatique
  - O(|A|) par step — ultra-léger
  - Pas d'hyperparamètre à régler (α₀=1, β₀=1 = prior uniforme)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BetaPosterior:
    """
    Posterior Beta pour un paramètre de clarté.
    """
    alpha: float = 1.0   # Succès + prior
    beta: float = 1.0    # Échecs + prior

    def update(self, correct: bool):
        """Met à jour le posterior avec une observation binaire."""
        if correct:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def sample(self) -> float:
        """Échantillonne une valeur du posterior (Thompson Sampling)."""
        # Approximation de Beta par normal si α, β > 20 (rapide)
        if self.alpha > 20 and self.beta > 20:
            mean = self.alpha / (self.alpha + self.beta)
            var = (self.alpha * self.beta) / (
                (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
            std = math.sqrt(max(var, 1e-10))
            # Box-Muller pour normal
            u1 = max(random.random(), 1e-10)
            u2 = random.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            sample = mean + std * z
            return max(0.01, min(0.99, sample))
        else:
            # Beta exact via Gamma (Jöhnk's algorithm pour petits α, β)
            return self._sample_exact()

    def _sample_exact(self) -> float:
        """Échantillonnage exact Beta via Gamma (algorithm de Jöhnk)."""
        x = self._sample_gamma(self.alpha)
        y = self._sample_gamma(self.beta)
        if x + y < 1e-10:
            return 0.5
        return max(0.01, min(0.99, x / (x + y)))

    def _sample_gamma(self, shape: float) -> float:
        """
        Échantillonnage Gamma via Marsaglia & Tsang.
        shape > 0.
        """
        if shape < 1.0:
            # Pour shape < 1, utiliser Gamma(shape+1) * U^(1/shape)
            return self._sample_gamma(shape + 1.0) * (random.random() ** (1.0 / shape))

        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)

        while True:
            while True:
                x = self._normal()
                v = 1.0 + c * x
                if v > 0:
                    break
            v = v * v * v
            u = random.random()
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def _normal(self) -> float:
        """Normal(0,1) via Box-Muller."""
        u1 = max(random.random(), 1e-10)
        u2 = random.random()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        ab = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab * ab * (ab + 1))

    @property
    def confidence(self) -> float:
        """Confiance dans l'estimation (0=incertain, 1=certain)."""
        return 1.0 - min(1.0, 4.0 * self.variance)  # Var max = 0.25 pour Beta(1,1)

    @property
    def n_observations(self) -> int:
        return self.alpha + self.beta - 2.0  # Sans le prior


@dataclass
class ClarityLearner:
    """
    Estime la clarté de chaque action en Thompson Sampling.

    Usage:
        learner = ClarityLearner(n_actions=9)
        # Après chaque exécution d'action i :
        sampled_p = learner.sample(i)         # Estimation pour le planning
        learner.update(i, observation_correct) # Mise à jour du posterior
    """
    n_actions: int
    posteriors: Dict[int, BetaPosterior] = field(default_factory=dict)

    def __post_init__(self):
        # Prior Beta(2,2) : plus conservateur que Beta(1,1)
        # Centre les estimations initiales vers 0.5 (incertain)
        for i in range(self.n_actions):
            self.posteriors[i] = BetaPosterior(alpha=2.0, beta=2.0)

    def sample(self, action_id: int) -> float:
        """
        Échantillonne la clarté estimée pour l'action donnée.
        Utilisé pour le scoring ΔR/C dans le contrôleur.
        """
        return self.posteriors[action_id].sample()

    def sample_all(self) -> List[float]:
        """Échantillonne la clarté pour toutes les actions."""
        return [self.sample(i) for i in range(self.n_actions)]

    def update(self, action_id: int, correct: bool):
        """
        Met à jour le posterior avec le résultat de l'observation.
        correct = True si l'observation était cohérente avec la vraie classe.
        """
        self.posteriors[action_id].update(correct)

    def reset_posterior(self, action_id: int):
        """
        Réinitialise le posterior d'une action (après un changement point).
        Revient au prior Beta(2,2).
        """
        self.posteriors[action_id] = BetaPosterior(alpha=2.0, beta=2.0)

    def mean(self, action_id: int) -> float:
        """Estimation moyenne (sans échantillonnage)."""
        return self.posteriors[action_id].mean

    def confidence(self, action_id: int) -> float:
        """Confiance dans l'estimation de l'action."""
        return self.posteriors[action_id].confidence

    def n_observations(self, action_id: int) -> int:
        """Nombre d'observations pour cette action."""
        return int(self.posteriors[action_id].n_observations)

    def best_action(self) -> int:
        """Action avec la meilleure clarté moyenne."""
        return max(range(self.n_actions), key=lambda i: self.mean(i))

    def exploration_ratio(self) -> float:
        """
        Ratio d'exploration : actions peu observées / total.
        Utile pour diagnostiquer si le système explore assez.
        """
        min_obs = min(self.n_observations(i) for i in range(self.n_actions))
        max_obs = max(self.n_observations(i) for i in range(self.n_actions))
        if max_obs == 0:
            return 1.0
        return min_obs / max_obs

    def summary(self) -> Dict:
        """Résumé de l'état du learner."""
        return {
            "actions": {
                i: {
                    "mean_clarity": round(self.mean(i), 3),
                    "confidence": round(self.confidence(i), 3),
                    "n_obs": self.n_observations(i),
                }
                for i in range(self.n_actions)
            },
            "exploration_ratio": round(self.exploration_ratio(), 3),
        }
