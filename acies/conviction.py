"""
ACIES — Conviction Mechanism

Gère la zone de "flottement" quand le contrôleur est proche du seuil
de confiance mais pas encore sûr. Anti-oscillation.

Quand confidence ∈ [zone_start, threshold]:
  - On est dans la "conviction zone"
  - Les actions à haute clarté reçoivent un bonus
  - Les actions "cheap mais uninformative" sont pénalisées
"""

from dataclasses import dataclass, field
from typing import List, Optional
from .actions import Action


@dataclass
class ConvictionConfig:
    """Configuration du mécanisme de conviction."""
    zone_start: float = 0.90      # Fraction du threshold pour entrer dans la zone
    bonus_high_clarity: float = 1.3   # Bonus pour actions clarifiées dans la zone
    penalty_low_clarity: float = 0.5  # Pénalité pour actions peu claires dans la zone
    oscillation_threshold: int = 4    # Steps max dans la zone avant alerte
    clarity_threshold: float = 0.8    # Clarté min pour considérer une action "clarifiée"


@dataclass
class ConvictionState:
    """État du mécanisme de conviction."""
    in_zone: bool = False          # Sommes-nous dans la zone ?
    zone_steps: int = 0            # Combien de steps dans la zone
    zone_entries: int = 0          # Nombre total d'entrées dans la zone
    last_confidence: float = 0.0   # Dernière confiance observée
    oscillation_count: int = 0     # Nombre d'oscillations détectées
    belief_direction: float = 0.0  # Direction du belief (+ = vers 1, - = vers 0)
    belief_history: List[float] = field(default_factory=list)


class Conviction:
    """
    Mécanisme de conviction pour le contrôleur APC.

    Détecte quand on est dans la zone d'hésitation et ajuste les scores
    pour forcer une décision plus rapidement.

    Usage:
        conviction = Conviction(config=ConvictionConfig(zone_start=0.85))
        # Dans la boucle de contrôle :
        conviction.update(belief.confidence)
        if conviction.state.in_zone:
            scores = conviction.adjust_scores(scores, clarity_estimates)
        if conviction.should_force_commit():
            action = conviction.force_commit_action(candidates)
    """

    def __init__(self, config: ConvictionConfig = None):
        self.config = config or ConvictionConfig()
        self.state = ConvictionState()

    def reset(self):
        """Remet l'état pour une nouvelle tâche."""
        self.state = ConvictionState()

    def update(self, confidence: float):
        """
        Met à jour l'état de conviction avec la confiance courante.
        À appeler à chaque step du contrôleur.
        """
        threshold = 1.0  # Le seuil est toujours 1.0 (normalisé)
        zone_threshold = threshold * self.config.zone_start

        was_in_zone = self.state.in_zone
        self.state.in_zone = confidence >= zone_threshold
        self.state.last_confidence = confidence

        # Tracker la direction du belief
        if len(self.state.belief_history) >= 2:
            prev = self.state.belief_history[-1]
            self.state.belief_direction = confidence - prev
        self.state.belief_history.append(confidence)

        if self.state.in_zone:
            self.state.zone_steps += 1

            if not was_in_zone:
                # Entrée dans la zone
                self.state.zone_entries += 1

            # Détecter oscillation : confiance qui monte puis descend
            if len(self.state.belief_history) >= 3:
                h = self.state.belief_history
                if h[-1] < h[-2] and h[-2] > h[-3]:
                    self.state.oscillation_count += 1
        else:
            # Sortie de la zone → reset le compteur
            if was_in_zone:
                self.state.zone_steps = 0

    def adjust_scores(
        self,
        scores: list,
        clarity_estimates: dict,
    ) -> list:
        """
        Ajuste les scores ΔR/C quand on est dans la zone de conviction.

        Args:
            scores: Liste de (action, score, clarity)
            clarity_estimates: {action_id: estimated_clarity}

        Returns:
            Liste ajustée de (action, adjusted_score, clarity)
        """
        if not self.state.in_zone:
            return scores

        adjusted = []
        for action, score, clarity in scores:
            clarity_est = clarity_estimates.get(action.id, clarity)

            if clarity_est >= self.config.clarity_threshold:
                # Action clarifiée → bonus dans la zone
                bonus = self.config.bonus_high_clarity
                # Plus on est longtemps dans la zone, plus le bonus augmente
                zone_pressure = min(self.state.zone_steps / self.config.oscillation_threshold, 1.0)
                adjusted_score = score * (1.0 + (bonus - 1.0) * zone_pressure)
            else:
                # Action peu claire → pénalité dans la zone
                penalty = self.config.penalty_low_clarity
                adjusted_score = score * penalty

            adjusted.append((action, adjusted_score, clarity))

        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def summary(self) -> dict:
        return {
            "in_zone": self.state.in_zone,
            "zone_steps": self.state.zone_steps,
            "zone_entries": self.state.zone_entries,
            "oscillation_count": self.state.oscillation_count,
            "last_confidence": round(self.state.last_confidence, 4),
        }
