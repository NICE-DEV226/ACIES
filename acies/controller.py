"""
ACIES — Main Controller

Le contrôleur APC qui combine :
- BeliefState (filtre bayésien)
- ClarityLearner (Thompson Sampling)
- SafetyLayer (garanties de risque)
- CostModel (coûts hardware)

Boucle de contrôle :
1. Échantillonner les clartés estimées (Thompson)
2. Calculer ΔR/C pour chaque action
3. Filtrer par safety layer
4. Exécuter la meilleure action
5. Mettre à jour les croyances et les estimations
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

from .actions import Action, HardwareProfile, build_standard_actions
from .belief import BeliefState
from .clarity_learner import ClarityLearner
from .safety import SafetyLayer, SafetyConfig
from .conviction import Conviction, ConvictionConfig


@dataclass
class APCConfig:
    """Configuration complète du contrôleur APC."""
    # Croyance
    prior: float = 0.5
    temperature: float = 1.0       # Calibrage température

    # Arrêt
    confidence_threshold: float = 0.95
    max_steps: int = 8

    # Sécurité
    max_risk: float = 2.0
    emergency_risk: float = 4.0
    min_observations: int = 1

    # Hardware
    hardware: HardwareProfile = field(default_factory=HardwareProfile.default)

    # Thompson Sampling
    clarity_prior_alpha: float = 1.0
    clarity_prior_beta: float = 1.0

    # Conviction (anti-oscillation)
    conviction_zone_start: float = 0.85
    conviction_oscillation_threshold: int = 3

    # Debug
    verbose: bool = False


@dataclass
class APCStep:
    """Résultat d'un step du contrôleur."""
    step: int
    action: Action
    observation: int
    belief_before: float
    belief_after: float
    risk_before: float
    risk_after: float
    score: float
    clarity_sampled: float
    clarity_true: float
    cost: float
    latency_ms: float
    safe: bool


@dataclass
class APCResult:
    """Résultat complet de l'exécution du contrôleur."""
    decision: int
    correct: bool
    total_cost: float
    total_latency_ms: float
    total_energy_mJ: float
    peak_memory_MB: float
    total_flops_M: float
    n_steps: int
    n_emergency: int
    steps: List[APCStep]
    final_belief: float
    final_risk: float
    abstained: bool

    @property
    def actions_taken(self) -> List[str]:
        return [s.action.name for s in self.steps]

    def summary(self) -> dict:
        return {
            "decision": self.decision,
            "correct": self.correct,
            "total_cost": round(self.total_cost, 2),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "total_energy_mJ": round(self.total_energy_mJ, 1),
            "peak_memory_MB": round(self.peak_memory_MB, 1),
            "n_steps": self.n_steps,
            "n_emergency": self.n_emergency,
            "final_belief": round(self.final_belief, 4),
            "final_risk": round(self.final_risk, 4),
            "abstained": self.abstained,
            "actions": self.actions_taken,
        }


class APCController:
    """
    Adaptive Perception Controller — version robuste.

    Usage:
        apc = APCController(APCConfig(
            confidence_threshold=0.92,
            hardware=HardwareProfile.jetson_orin(),
        ))

        # Boucle de contrôle pour chaque image
        result = apc.run(
            true_class=1,           # Pour la simulation
            clarity_fn=lambda a: get_clarity(a),  # Fonction de clarté réelle
            observation_fn=lambda a, tc: observe(a, tc),  # Fonction d'observation
        )
    """

    def __init__(self, config: APCConfig = None, actions: List[Action] = None):
        self.config = config or APCConfig()
        self.actions = actions or build_standard_actions()

        # Initialiser les sous-systèmes
        self.belief = BeliefState(
            prior=self.config.prior,
            temperature=self.config.temperature,
        )
        self.learner = ClarityLearner(
            n_actions=len(self.actions),
        )
        self.safety = SafetyLayer(
            config=SafetyConfig(
                max_risk=self.config.max_risk,
                emergency_risk=self.config.emergency_risk,
                min_observations=self.config.min_observations,
                confidence_threshold=self.config.confidence_threshold,
            ),
        )
        self.conviction = Conviction(
            config=ConvictionConfig(
                zone_start=self.config.conviction_zone_start,
                oscillation_threshold=self.config.conviction_oscillation_threshold,
            ),
        )

        # Historique
        self._run_history: List[APCResult] = []

    def reset(self):
        """Remet le contrôleur à zéro (nouvelle image/tâche)."""
        self.belief.reset()
        self.safety.reset()
        self.conviction.reset()

    def reset_all(self):
        """Remet tout (beliefs + learner + safety)."""
        self.belief.reset()
        self.learner = ClarityLearner(n_actions=len(self.actions))
        self.safety.reset()
        self._run_history.clear()

    def _score_actions(self) -> List[Tuple[Action, float, float]]:
        """
        Score chaque action par ΔR/C avec exploration bonus UCB1.

        Score = ΔR/C + bonus_exploration
        Si le risque est élevé (> max_risk * 0.7), pénalise les actions cheap
        pour forcer l'utilisation d'actions plus informatives.
        Si on est dans la zone de conviction, ajuste les scores.
        """
        scored = []
        sampled_clarities = self.learner.sample_all()

        total_obs = sum(self.learner.n_observations(i)
                       for i in range(len(self.actions)))

        high_risk = self.belief.risk > self.config.max_risk * 0.7

        for i, action in enumerate(self.actions):
            clarity = sampled_clarities[i]
            cost = action.cost(self.config.hardware)
            base_score = self.belief.delta_risk_efficiency(clarity, cost)

            # UCB1 exploration bonus
            n_i = max(self.learner.n_observations(i), 1)
            exploration = 3.0 * math.sqrt(math.log(max(total_obs, 1) + 1) / n_i)

            # Bonus ×5 si jamais essayé
            if self.learner.n_observations(i) == 0:
                exploration *= 5.0

            # Pénalité si risque élevé : favoriser les actions clarifiées
            if high_risk and clarity < 0.7:
                base_score *= 0.3  # Forte pénalité pour les actions peu claires

            score = base_score + exploration
            scored.append((action, score, clarity))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Appliquer les ajustements de conviction
        clarity_estimates = {self.actions[i].id: self.learner.mean(i)
                            for i in range(len(self.actions))}
        scored = self.conviction.adjust_scores(scored, clarity_estimates)

        return scored

    def run(
        self,
        true_class: int,
        clarity_fn: Callable[[Action], float],
        max_steps: int = None,
    ) -> APCResult:
        """
        Exécute le contrôleur APC sur une tâche.

        Args:
            true_class: La vraie classe (0 ou 1) — pour la simulation
            clarity_fn: Fonction qui retourne la vraie clarté pour une action
            max_steps: Nombre maximum d'étapes ( défaut: config.max_steps)

        Returns:
            APCResult avec toutes les métriques
        """
        if max_steps is None:
            max_steps = self.config.max_steps

        self.belief.reset()
        total_cost = 0.0
        total_latency = 0.0
        total_energy = 0.0
        peak_memory = 0.0
        total_flops = 0.0
        steps = []
        n_emergency = 0

        for step in range(max_steps):
            # 0. Mettre à jour la conviction
            self.conviction.update(self.belief.confidence)

            # 1. Scanner les actions et calculer les scores
            scored = self._score_actions()

            # 2. Safety layer : sélectionner une action sûre
            candidates = [(a, s) for a, s, c in scored]
            n_obs = sum(1 for s in steps)
            clarity_estimates = {self.actions[i].id: self.learner.mean(i)
                                for i in range(len(self.actions))}
            safe_action = self.safety.select(
                self.belief, candidates, n_obs,
                clarity_estimates=clarity_estimates,
            )

            if safe_action is None:
                # STOP
                break

            # 3. Vérifier l'urgence
            if self.safety.state.n_emergency > n_emergency:
                n_emergency = self.safety.state.n_emergency

            # 4. Exécuter l'action
            clarity_true = clarity_fn(safe_action)
            clarity_sampled = self.learner.sample(safe_action.id)
            cost = safe_action.cost(self.config.hardware)

            # 5. Générer l'observation
            obs = 1 if (true_class == 1 and random.random() < clarity_true) else \
                  (0 if true_class == 0 and random.random() < clarity_true else \
                   (1 if true_class == 1 else 0))
            if true_class == 1:
                obs = 1 if random.random() < clarity_true else 0
            else:
                obs = 0 if random.random() < clarity_true else 1

            # 6. Calculer le score de cette action
            score = self.belief.delta_risk_efficiency(clarity_sampled, cost)

            # 7. Mettre à jour les états
            belief_before = self.belief.belief
            risk_before = self.belief.risk

            self.belief.update(obs, clarity_true)

            # 8. Mettre à jour le learner
            # L'observation est "correcte" si elle est cohérente avec la vraie classe
            observation_correct = (obs == true_class)
            self.learner.update(safe_action.id, observation_correct)

            # 9. Vérifier la sécurité post-action
            safe = self.safety.check_post_action(self.belief, safe_action)

            # 10. Si le risque est trop élevé après l'action, forcer continuation
            if self.belief.risk > self.safety.config.max_risk:
                # Ne pas s'arrêter — forcer une observation corrective au step suivant
                pass

            # 10. Accumuler les coûts
            total_cost += cost
            total_latency += safe_action.base_latency_ms * self.config.hardware.latency_scale
            total_energy += safe_action.base_energy_mJ * self.config.hardware.energy_scale
            peak_memory = max(peak_memory, safe_action.base_memory_MB * self.config.hardware.memory_scale)
            total_flops += safe_action.pixel_ratio * 1400  # Approximation FLOPs

            # 11. Enregistrer le step
            steps.append(APCStep(
                step=step,
                action=safe_action,
                observation=obs,
                belief_before=belief_before,
                belief_after=self.belief.belief,
                risk_before=risk_before,
                risk_after=self.belief.risk,
                score=score,
                clarity_sampled=clarity_sampled,
                clarity_true=clarity_true,
                cost=cost,
                latency_ms=safe_action.base_latency_ms * self.config.hardware.latency_scale,
                safe=safe,
            ))

            # 12. Debug
            if self.config.verbose:
                zone = f" [CONVICTION ZONE, step={self.conviction.state.zone_steps}]" if self.conviction.state.in_zone else ""
                print(f"  Step {step}: {safe_action.name} "
                      f"(clarity={clarity_true:.2f}, score={score:.3f}) "
                      f"→ obs={obs}, belief={self.belief.belief:.3f} "
                      f"risk={self.belief.risk:.3f}{zone}")

        # Décision finale
        decision = self.belief.decision
        correct = (decision == true_class)
        abstained = self.safety.should_abstain(self.belief, len(steps))

        result = APCResult(
            decision=decision,
            correct=correct,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            total_energy_mJ=total_energy,
            peak_memory_MB=peak_memory,
            total_flops_M=total_flops,
            n_steps=len(steps),
            n_emergency=n_emergency,
            steps=steps,
            final_belief=self.belief.belief,
            final_risk=self.belief.risk,
            abstained=abstained,
        )

        self._run_history.append(result)
        return result

    def batch_run(
        self,
        tasks: List[Tuple[int, float, Callable]],
        n_trials: int = 1,
    ) -> List[APCResult]:
        """
        Exécute le contrôleur sur plusieurs tâches.

        Args:
            tasks: Liste de (true_class, difficulty, clarity_fn)
            n_trials: Nombre de répétitions par tâche

        Returns:
            Liste de résultats
        """
        all_results = []
        for true_class, difficulty, clarity_fn in tasks:
            for _ in range(n_trials):
                self.belief.reset()
                result = self.run(true_class, clarity_fn)
                all_results.append(result)
        return all_results

    @property
    def avg_cost(self) -> float:
        if not self._run_history:
            return 0.0
        return sum(r.total_cost for r in self._run_history) / len(self._run_history)

    @property
    def avg_accuracy(self) -> float:
        if not self._run_history:
            return 0.0
        return sum(1 for r in self._run_history if r.correct) / len(self._run_history)

    @property
    def avg_latency(self) -> float:
        if not self._run_history:
            return 0.0
        return sum(r.total_latency_ms for r in self._run_history) / len(self._run_history)

    def summary(self) -> dict:
        return {
            "n_runs": len(self._run_history),
            "avg_cost": round(self.avg_cost, 2),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "avg_latency_ms": round(self.avg_latency, 1),
            "avg_epc": round(self.avg_cost / max(self.avg_accuracy, 1e-10), 2),
            "learner": self.learner.summary(),
            "safety": self.safety.summary(),
            "conviction": self.conviction.summary(),
        }
