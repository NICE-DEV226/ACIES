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
from .change_point import ChangePointDetector, ChangePointConfig


@dataclass
class APCConfig:
    """Configuration complète du contrôleur APC."""
    # Croyance
    prior: float = 0.5
    temperature: float = 1.0       # Calibrage température

    # Arrêt
    confidence_threshold: float = 0.95
    max_steps: int = 8

    # Robustesse
    max_cost_per_image: float = 800.0      # Budget max par image
    min_clarity_threshold: float = 0.4     # Clarté en dessous = image dégradée
    degradation_patience: int = 3          # Steps avant d'abandonner si clarity basse
    abstention_confidence: float = 0.6     # Confiance min pour ne pas abstain

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

    # Change-point detection
    change_point_enabled: bool = False  # Disable for independent images; enable for streaming
    change_point_threshold: float = 0.5
    change_point_hazard: float = 1/200

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
    decision: int            # 0/1, or -1 when abstaining
    correct: Optional[bool]  # None when the true class is unknown (deployment)
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
    degraded: bool           # Image trop dégradée (blur, noise extreme)
    cost_budget_exceeded: bool  # Budget max dépassé
    avg_clarity: float       # Clarté moyenne observée

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
            "degraded": self.degraded,
            "cost_budget_exceeded": self.cost_budget_exceeded,
            "avg_clarity": round(self.avg_clarity, 3),
            "actions": self.actions_taken,
        }


@dataclass
class _TaskState:
    """État d'une tâche en cours (une image / une décision)."""
    max_steps: int
    emergency_base: int
    steps: List[APCStep] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency: float = 0.0
    total_energy: float = 0.0
    peak_memory: float = 0.0
    total_flops: float = 0.0
    low_clarity_count: int = 0
    clarity_sum: float = 0.0
    degraded: bool = False
    budget_exceeded: bool = False
    done: bool = False
    pending: Optional[Action] = None


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
        self.change_detector = None
        if self.config.change_point_enabled:
            self.change_detector = ChangePointDetector(
                n_actions=len(self.actions),
                config=ChangePointConfig(
                    threshold=self.config.change_point_threshold,
                    hazard_rate=self.config.change_point_hazard,
                ),
            )

        # Historique
        self._run_history: List[APCResult] = []
        self._task: Optional[_TaskState] = None

    def reset(self):
        """Remet le contrôleur à zéro (nouvelle image/tâche)."""
        self.belief.reset()
        self.safety.reset()
        self.conviction.reset()
        if self.change_detector:
            self.change_detector.reset()
        self._task = None

    def reset_all(self):
        """Remet tout (beliefs + learner + safety)."""
        self.belief.reset()
        self.learner = ClarityLearner(n_actions=len(self.actions))
        self.safety.reset()
        self.conviction.reset()
        self._task = None
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

    # ------------------------------------------------------------------
    # Step API — décision découplée de l'environnement
    #
    #   apc.begin()
    #   while (action := apc.next_action()) is not None:
    #       obs = my_perception(action)          # 0/1 : vote de la perception
    #       apc.observe(action, obs)             # aucune vérité terrain requise
    #   result = apc.finish()                    # result.decision, result.abstained
    #   apc.feedback(action, correct=True)       # optionnel : label tardif
    # ------------------------------------------------------------------

    def _require_task(self) -> _TaskState:
        if self._task is None:
            raise RuntimeError("No task in progress: call begin() first")
        return self._task

    def begin(self, max_steps: int = None):
        """Démarre une tâche. Remet à zéro l'état par-tâche, garde ce qui est appris."""
        self.belief.reset()
        self.conviction.reset()
        self._task = _TaskState(
            max_steps=self.config.max_steps if max_steps is None else max_steps,
            emergency_base=self.safety.state.n_emergency,
        )

    def _emergency_clarity(self) -> Dict[int, float]:
        """Clarté attendue par action pour l'action d'urgence : apprise si essayée,
        sinon a priori monotone en résolution (jamais l'ancienne valeur fixe)."""
        out = {}
        for i, a in enumerate(self.actions):
            if self.learner.n_observations(i) > 0:
                out[a.id] = self.learner.mean(i)
            else:
                out[a.id] = 0.5 + 0.49 * a.pixel_ratio
        return out

    def next_action(self) -> Optional[Action]:
        """
        Choisit la prochaine action de perception, ou None s'il faut s'arrêter
        (confiance atteinte, nombre max d'étapes, budget de coût, dégradation).
        """
        t = self._require_task()
        if t.done:
            return None
        if t.pending is not None:
            raise RuntimeError("observe() must be called for the pending action first")
        if t.degraded or len(t.steps) >= t.max_steps:
            t.done = True
            return None
        if t.total_cost >= self.config.max_cost_per_image:
            t.budget_exceeded = True
            t.done = True
            return None

        self.conviction.update(self.belief.confidence)
        scored = self._score_actions()
        candidates = [(a, s) for a, s, c in scored]
        clarity_estimates = {self.actions[i].id: self.learner.mean(i)
                            for i in range(len(self.actions))}
        action = self.safety.select(
            self.belief, candidates, len(t.steps),
            clarity_estimates=clarity_estimates,
            emergency_clarity=self._emergency_clarity(),
        )
        if action is None:
            t.done = True  # STOP décidé par la couche de sécurité
            return None

        # Budget vérifié AVANT d'exécuter l'action
        if t.total_cost + action.cost(self.config.hardware) > self.config.max_cost_per_image:
            t.budget_exceeded = True
            t.done = True
            return None

        t.pending = action
        return action

    def observe(
        self,
        action: Action,
        obs: int,
        clarity: float = None,
        correct: bool = None,
    ) -> APCStep:
        """
        Enregistre l'observation (0/1) produite par `action`.

        Args:
            action: l'action renvoyée par next_action()
            obs: vote binaire de la perception
            clarity: P(obs correcte | action) à utiliser pour la mise à jour bayésienne.
                Par défaut, l'estimation apprise (moyenne du posterior Beta) : aucune
                vérité terrain n'est requise. Fournir une valeur seulement si elle est
                réellement connue (simulation, modèle calibré).
            correct: si le label est connu maintenant, met à jour l'apprentissage de
                clarté. Sinon, appeler feedback() plus tard ou ne rien faire.
        """
        t = self._require_task()
        if t.pending is None or t.pending.id != action.id:
            raise RuntimeError("observe() must follow next_action() with the same action")
        if obs not in (0, 1):
            raise ValueError(f"obs must be 0 or 1, got {obs!r}")
        if clarity is None:
            clarity = self.learner.mean(action.id)
        if not 0.0 <= clarity <= 1.0:
            raise ValueError(f"clarity must be in [0, 1], got {clarity!r}")
        clarity = min(max(clarity, 0.01), 0.99)
        t.pending = None

        hw = self.config.hardware
        cost = action.cost(hw)
        clarity_sampled = self.learner.sample(action.id)
        score = self.belief.delta_risk_efficiency(clarity_sampled, cost)
        belief_before = self.belief.belief
        risk_before = self.belief.risk

        # Détection de dégradation (clarté basse consécutive)
        t.clarity_sum += clarity
        if clarity < self.config.min_clarity_threshold:
            t.low_clarity_count += 1
        else:
            t.low_clarity_count = 0
        if t.low_clarity_count >= self.config.degradation_patience:
            t.degraded = True

        self.belief.update(obs, clarity)
        if correct is not None:
            self.learner.update(action.id, correct)

        if self.change_detector:
            if self.change_detector.update(action.id, clarity):
                self.learner.reset_posterior(action.id)
                if self.config.verbose:
                    print(f"    ⚡ CHANGE POINT detected on {action.name} "
                          f"at step {len(t.steps)}")

        safe = self.safety.check_post_action(self.belief, action)

        latency = action.base_latency_ms * hw.latency_scale
        t.total_cost += cost
        t.total_latency += latency
        t.total_energy += action.base_energy_mJ * hw.energy_scale
        t.peak_memory = max(t.peak_memory, action.base_memory_MB * hw.memory_scale)
        t.total_flops += action.pixel_ratio * 1400

        step = APCStep(
            step=len(t.steps), action=action, observation=obs,
            belief_before=belief_before, belief_after=self.belief.belief,
            risk_before=risk_before, risk_after=self.belief.risk,
            score=score, clarity_sampled=clarity_sampled, clarity_true=clarity,
            cost=cost, latency_ms=latency, safe=safe,
        )
        t.steps.append(step)

        if self.config.verbose:
            zone = (f" [CONVICTION ZONE, step={self.conviction.state.zone_steps}]"
                    if self.conviction.state.in_zone else "")
            print(f"  Step {step.step}: {action.name} "
                  f"(clarity={clarity:.2f}, score={score:.3f}) "
                  f"→ obs={obs}, belief={self.belief.belief:.3f} "
                  f"risk={self.belief.risk:.3f}{zone}")
        return step

    def feedback(self, action: Action, correct: bool):
        """Label tardif : met à jour l'estimation de clarté de `action`."""
        self.learner.update(action.id, correct)

    def finish(self, true_class: int = None) -> APCResult:
        """Termine la tâche et renvoie le résultat (décision, ou abstention = -1)."""
        t = self._require_task()
        n_steps_done = len(t.steps)
        avg_clarity = t.clarity_sum / max(n_steps_done, 1)

        # Abstention si pas assez d'observations OU confiance trop basse
        abstained = (
            n_steps_done < self.config.min_observations or
            self.belief.confidence < self.config.abstention_confidence
        )
        if abstained:
            decision = -1
            correct = False if true_class is not None else None
        else:
            decision = self.belief.decision
            correct = (decision == true_class) if true_class is not None else None

        result = APCResult(
            decision=decision,
            correct=correct,
            total_cost=t.total_cost,
            total_latency_ms=t.total_latency,
            total_energy_mJ=t.total_energy,
            peak_memory_MB=t.peak_memory,
            total_flops_M=t.total_flops,
            n_steps=n_steps_done,
            n_emergency=self.safety.state.n_emergency - t.emergency_base,
            steps=t.steps,
            final_belief=self.belief.belief,
            final_risk=self.belief.risk,
            abstained=abstained,
            degraded=t.degraded,
            cost_budget_exceeded=t.budget_exceeded,
            avg_clarity=avg_clarity,
        )
        self._run_history.append(result)
        self._task = None
        return result

    def run(
        self,
        true_class: int,
        clarity_fn: Callable[[Action], float],
        max_steps: int = None,
        oracle_clarity: bool = True,
    ) -> APCResult:
        """
        SIMULATION : exécute une tâche complète contre un environnement synthétique.

        Génère les observations à partir de `true_class` et `clarity_fn` (canal
        symétrique). Pour un déploiement réel, utiliser begin/next_action/observe/finish.

        Args:
            true_class: La vraie classe (0 ou 1), utilisée par le simulateur
            clarity_fn: clarté réelle (simulateur) d'une action
            max_steps: Nombre maximum d'étapes (défaut: config.max_steps)
            oracle_clarity: True (défaut, comportement historique) : le belief est mis
                à jour avec la vraie clarté du simulateur. False : avec l'estimation
                apprise, comme en déploiement (résultats plus réalistes, démarrage plus
                lent car le prior Beta(2,2) ne donne aucune information au début).
        """
        self.begin(max_steps)
        while True:
            action = self.next_action()
            if action is None:
                break
            clarity_true = clarity_fn(action)
            if true_class == 1:
                obs = 1 if random.random() < clarity_true else 0
            else:
                obs = 0 if random.random() < clarity_true else 1
            self.observe(
                action, obs,
                clarity=clarity_true if oracle_clarity else None,
                correct=(obs == true_class),
            )
        return self.finish(true_class)

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
        result = {
            "n_runs": len(self._run_history),
            "avg_cost": round(self.avg_cost, 2),
            "avg_accuracy": round(self.avg_accuracy, 4),
            "avg_latency_ms": round(self.avg_latency, 1),
            "avg_epc": round(self.avg_cost / max(self.avg_accuracy, 1e-10), 2),
            "learner": self.learner.summary(),
            "safety": self.safety.summary(),
            "conviction": self.conviction.summary(),
        }
        if self.change_detector:
            result["change_points"] = self.change_detector.summary()
        return result
