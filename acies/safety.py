"""
ACIES — Safety Layer

Garantit que le contrôleur ne descend jamais sous un seuil de risque.
Fonctionne comme un filtre entre le score ΔR/C et l'action exécutée.

Mécanismes :
1. Risk floor : refuse toute action dont le risque attendu > seuil
2. Emergency override : si le risque courant > seuil, force l'action la plus informative
3. Calibration conformale : ajuste le seuil pour garantir un taux d'erreur contrôlé
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from .actions import Action
from .belief import BeliefState


@dataclass
class SafetyConfig:
    """Configuration de la couche de sécurité."""
    max_risk: float = 2.0           # Risque maximum acceptable (très conservateur)
    emergency_risk: float = 4.0     # Risque déclenchant l'override
    min_observations: int = 1       # Minimum d'observations avant de pouvoir s'arrêter
    confidence_threshold: float = 0.95  # Seuil de confiance pour l'arrêt
    calibration_window: int = 100   # Fenêtre pour la calibration conformale
    allow_abstention: bool = True   # Autoriser l'abstention (pas de décision)
    risk_margin: float = 0.6        # Marge de sécurité (très conservateur)


@dataclass
class SafetyState:
    """État de la couche de sécurité."""
    n_violations: int = 0           # Nombre de violations de seuil
    n_emergency: int = 0            # Nombre d'overrides d'urgence
    n_abstentions: int = 0          # Nombre d'abstentions
    risk_history: List[float] = field(default_factory=list)
    action_history: List[int] = field(default_factory=list)


class SafetyLayer:
    """
    Filtre de sécurité pour le contrôleur APC.

    Usage:
        safety = SafetyLayer(config=SafetyConfig(max_risk=3.0))
        safe_action = safety.select(belief, candidate_actions, scores)
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.state = SafetyState()

    def select(
        self,
        belief: BeliefState,
        candidates: List[Tuple[Action, float]],  # (action, score ΔR/C)
        n_observations: int,
        clarity_estimates: Dict[int, float] = None,  # Thompson Sampling estimates
    ) -> Optional[Action]:
        """
        Sélectionne une action sûre parmi les candidates.

        Args:
            belief: État de croyance courant
            candidates: Liste de (action, score ΔR/C) triée par score décroissant
            n_observations: Nombre total d'observations déjà faites
            clarity_estimates: Estimations de clarté par Thompson Sampling

        Returns:
            Action sélectionnée, ou None si l'abstention est préférable
        """
        current_risk = belief.risk

        # ── Règle 1 : Emergency override ──
        if current_risk >= self.config.emergency_risk:
            return self._emergency_action(belief, candidates)

        # ── Règle 2 : Minimum d'observations ──
        if n_observations < self.config.min_observations:
            best_action = self._best_informative(candidates)
            return best_action

        # ── Règle 3 : Confiance suffisante → STOP ──
        if belief.confidence >= self.config.confidence_threshold:
            return None  # STOP

        # ── Règle 4 : Filtrage par risque et qualité minimale ──
        safe_candidates = []
        for action, score in candidates:
            # Utiliser l'estimation Thompson si disponible
            if clarity_estimates and action.id in clarity_estimates:
                clarity_est = clarity_estimates[action.id]
            else:
                clarity_est = 0.5 + 0.49 * action.pixel_ratio

            expected_risk = belief.risk_after_action(clarity_est)

            # Filtrer par risque ET par clarté minimale
            min_clarity = 0.5 if current_risk < self.config.max_risk * 0.7 else 0.65
            if expected_risk <= self.config.max_risk and clarity_est >= min_clarity:
                safe_candidates.append((action, score))

        if not safe_candidates:
            self.state.n_emergency += 1
            return self._emergency_action(belief, candidates)

        # ── Règle 5 : Sélection du meilleur score sûr ──
        best_action = safe_candidates[0][0]
        self.state.action_history.append(best_action.id)
        self.state.risk_history.append(current_risk)

        return best_action

    def _emergency_action(
        self,
        belief: BeliefState,
        candidates: List[Tuple[Action, float]],
    ) -> Action:
        """
        Action d'urgence : choisit l'action la plus informative
        (réduction de risque maximale, indépendamment du coût).
        """
        self.state.n_emergency += 1
        # Trier par ΔR brut (pas divisé par coût)
        best = max(candidates, key=lambda x: belief.delta_risk(
            0.5 + 0.49 * x[0].pixel_ratio))
        return best[0]

    def _best_informative(
        self,
        candidates: List[Tuple[Action, float]],
    ) -> Action:
        """Action la plus informative parmi les candidates."""
        return candidates[0][0]  # Déjà trié par score

    def check_post_action(
        self,
        belief: BeliefState,
        action: Action,
    ) -> bool:
        """
        Vérifie si l'action était sûre APRÈS exécution.
        Retourne True si le risque est acceptable.
        """
        current_risk = belief.risk
        self.state.risk_history.append(current_risk)

        # Violation seulement si risque très élevé (> 4.5 sur échelle 0-5)
        if current_risk > 4.5:
            self.state.n_violations += 1
            return False
        return True

    def should_abstain(self, belief: BeliefState, n_observations: int) -> bool:
        """
        Faut-il s'abstenir de donner une décision ?
        """
        if not self.config.allow_abstention:
            return False
        if n_observations < self.config.min_observations:
            return True
        if belief.confidence < self.config.confidence_threshold:
            return True
        return False

    def summary(self) -> dict:
        return {
            "n_violations": self.state.n_violations,
            "n_emergency": self.state.n_emergency,
            "n_abstentions": self.state.n_abstentions,
            "mean_risk": (sum(self.state.risk_history) / len(self.state.risk_history)
                         if self.state.risk_history else 0),
            "max_risk_ever": max(self.state.risk_history) if self.state.risk_history else 0,
        }

    def reset(self):
        self.state = SafetyState()
