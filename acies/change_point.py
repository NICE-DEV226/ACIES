"""
ACIES — Bayesian Change-Point Detection

Détecte quand la distribution de clarté change (shift soudain)
et signale qu'il faut réinitialiser les posteriors Thompson Sampling.

Algorithme : Bayesian Online Change-Point Detection (Adams & MacKay 2007)
- Maintient P(run_length = k | observations) pour chaque step
- Quand P(run_length = 0) dépasse un seuil → changement détecté
- Utilise un modèle Normal pour la clarté (moyenne + variance)

Utilisation dans APC :
  detector = ChangePointDetector(n_actions=9)
  # À chaque step :
  is_cp = detector.update(action_id, clarity_observed)
  if is_cp:
      learner.reset_posterior(action_id)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ChangePointConfig:
    """Configuration de la détection de changement."""
    hazard_rate: float = 1/200       # Taux de hazard prior (1/mean_run_length)
    observation_sigma: float = 0.1   # Écart-type des observations de clarté
    threshold: float = 0.5           # Seuil de détection (P(cp) > threshold → alarme)
    min_run_length: int = 5          # Longueur min avant de pouvoir détecter un CP
    warmup_observations: int = 10    # Observations avant d'activer la détection


@dataclass
class RunLengthDistribution:
    """Distribution sur la longueur de run pour une action."""
    log_probs: List[float] = field(default_factory=list)  # log P(r_t = k)
    t: int = 0  # Nombre d'observations
    max_run: int = 0

    def init(self, t: int):
        """Initialise la distribution à t=0."""
        self.t = 0
        self.log_probs = [0.0]  # log P(r_0 = 0) = log(1) = 0
        self.max_run = 0

    @property
    def probs(self) -> List[float]:
        """Convertit les log probs en probs normalisées."""
        if not self.log_probs:
            return []
        max_lp = max(self.log_probs)
        raw = [math.exp(lp - max_lp) for lp in self.log_probs]
        total = sum(raw)
        if total < 1e-300:
            return [1.0 / len(raw)] * len(raw)
        return [p / total for p in raw]

    @property
    def prob_cp(self) -> float:
        """P(run_length = 0) = probabilité de changement point."""
        probs = self.probs
        if not probs:
            return 0.0
        return probs[0]

    @property
    def most_likely_run(self) -> int:
        """Run length le plus probable."""
        probs = self.probs
        if not probs:
            return 0
        return max(range(len(probs)), key=lambda i: probs[i])


class ChangePointDetector:
    """
    Détecteur de changement bayésien en ligne.

    Usage:
        detector = ChangePointDetector(n_actions=9)
        # À chaque step :
        is_change = detector.update(action_id, clarity_observed)
        if is_change:
            # Réinitialiser le posterior Thompson pour cette action
            learner.reset_posterior(action_id)
    """

    def __init__(self, n_actions: int, config: ChangePointConfig = None):
        self.config = config or ChangePointConfig()
        self.n_actions = n_actions

        # Une distribution de run-length par action
        self.run_lengths: Dict[int, RunLengthDistribution] = {}
        for i in range(n_actions):
            rl = RunLengthDistribution()
            rl.init(0)
            self.run_lengths[i] = rl

        # Historique des changements détectés
        self.change_points: Dict[int, List[int]] = {i: [] for i in range(n_actions)}
        self.n_total_cp: int = 0

        # Observations par action (pour le modèle Normal)
        self.observations: Dict[int, List[float]] = {i: [] for i in range(n_actions)}

    def reset(self):
        """Remet tout à zéro."""
        for i in range(self.n_actions):
            self.run_lengths[i].init(0)
            self.change_points[i] = []
            self.observations[i] = []
        self.n_total_cp = 0

    def _log_likelihood(self, x: float, mean: float, var: float) -> float:
        """Log-vraisemblance Normal pour une observation x."""
        if var < 1e-10:
            var = 1e-10
        return -0.5 * (math.log(2 * math.pi * var) + (x - mean) ** 2 / var)

    def update(self, action_id: int, observation: float) -> bool:
        """
        Met à jour la détection avec une nouvelle observation.

        Args:
            action_id: ID de l'action
            observation: Valeur observée (clarity)

        Returns:
            True si un changement point est détecté
        """
        rl = self.run_lengths[action_id]
        obs_list = self.observations[action_id]

        rl.t += 1
        obs_list.append(observation)

        # Warmup : pas de détection pendant les premières observations
        if len(obs_list) < self.config.warmup_observations:
            return False

        t = rl.t
        old_log_probs = list(rl.log_probs)

        # Modèle : Normal(μ, σ²) avec σ fixe
        # μ estimé sur les observations depuis le dernier CP
        sigma = self.config.observation_sigma

        # Calculer les log-likelihoods pour chaque run-length
        new_log_probs = []

        # Taille du prior sur les run-lengths
        h = self.config.hazard_rate

        for k in range(len(old_log_probs)):
            # P(x_t | r_t = k, x_{1:t-1})
            # Si run_length = k, on a k observations depuis le dernier CP
            # Utiliser les k dernières observations pour estimer μ
            start_idx = len(obs_list) - k - 1
            if start_idx < 0:
                start_idx = 0
            window = obs_list[start_idx:]
            if len(window) < 2:
                mean_est = observation
                var_est = sigma ** 2
            else:
                mean_est = sum(window) / len(window)
                var_est = max(1e-10, sum((x - mean_est) ** 2 for x in window) / len(window))

            ll = self._log_likelihood(observation, mean_est, var_est)
            new_log_probs.append(old_log_probs[k] + ll)

        # Growth problem : run-length peut augmenter de 1
        # P(r_t = k+1 | r_{t-1} = k) = 1 - h
        growth_log_probs = [new_log_probs[k] + math.log(1 - h)
                           for k in range(len(new_log_probs))]
        growth_log_probs.insert(0, new_log_probs[0] + math.log(1 - h))

        # Change point : run-length revient à 0
        # P(r_t = 0 | r_{t-1} = k) = h pour tout k
        # Marginaliser sur tous les k precedents
        log_cp = math.log(h) + _log_sum_exp(new_log_probs)

        # Combiner growth + change point
        all_log_probs = [log_cp] + growth_log_probs[1:]

        # Tronquer à une taille max (pour la mémoire)
        max_size = min(t + 1, 200)
        if len(all_log_probs) > max_size:
            # Garder les derniers + le CP
            all_log_probs = all_log_probs[:1] + all_log_probs[-(max_size - 1):]

        rl.log_probs = all_log_probs

        # Vérifier si changement détecté
        is_cp = rl.prob_cp > self.config.threshold and t >= self.config.min_run_length

        if is_cp:
            self.change_points[action_id].append(t)
            self.n_total_cp += 1
            # Reset la run-length
            rl.log_probs = [0.0]

        return is_cp

    def get_clarity_stats(self, action_id: int, window: int = 20) -> Optional[Tuple[float, float]]:
        """
        Retourne (mean, variance) des clartés récentes pour une action.
        Utile pour diagnostiquer les changements.
        """
        obs = self.observations[action_id]
        if len(obs) < 2:
            return None
        recent = obs[-window:]
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        return (mean, var)

    def summary(self) -> dict:
        return {
            "total_change_points": self.n_total_cp,
            "per_action": {
                i: len(self.change_points[i])
                for i in range(self.n_actions)
            },
            "avg_run_length": {
                i: self.run_lengths[i].most_likely_run
                for i in range(self.n_actions)
            },
        }


def _log_sum_exp(log_vals: List[float]) -> float:
    """Calcule log(sum(exp(log_vals))) de manière numériquement stable."""
    if not log_vals:
        return -math.inf
    max_val = max(log_vals)
    if max_val == -math.inf:
        return -math.inf
    total = sum(math.exp(lp - max_val) for lp in log_vals)
    return max_val + math.log(total)
