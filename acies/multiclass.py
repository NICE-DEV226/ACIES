"""
ACIES — Multi-Class Belief Tracker

Maintient P(Y=k | observations) pour K classes.
Utilise un modèle de Dirichlet pour les probabilités.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MultiClassBelief:
    """
    Filtre bayésien multi-class.

    Maintient P(Y=k | O₁:t) avec Dirichlet prior.
    """
    n_classes: int = 10
    class_names: List[str] = field(default_factory=list)
    prior_alpha: float = 1.0      # Dirichlet prior (uniforme)
    temperature: float = 1.0
    min_prob: float = 0.001

    # État
    alpha: List[float] = field(default_factory=list)  # Dirichlet parameters
    history: List[Dict[str, float]] = field(default_factory=list)
    n_updates: int = 0

    def __post_init__(self):
        if not self.class_names:
            self.class_names = [str(i) for i in range(self.n_classes)]
        self.n_classes = len(self.class_names)
        self.alpha = [self.prior_alpha] * self.n_classes
        self.history = [self.probs.copy()]

    @property
    def probs(self) -> Dict[str, float]:
        """Current probability for each class."""
        total = sum(self.alpha)
        return {
            self.class_names[i]: self.alpha[i] / total
            for i in range(self.n_classes)
        }

    @property
    def prediction(self) -> str:
        """Most likely class."""
        p = self.probs
        return max(p, key=p.get)

    @property
    def confidence(self) -> float:
        """Confidence = max probability."""
        return max(self.probs.values())

    @property
    def entropy(self) -> float:
        """Entropy of the distribution (higher = more uncertain)."""
        p = self.probs
        h = 0.0
        for v in p.values():
            if v > 0:
                h -= v * math.log(v)
        return h

    def update(self, observed_class: str, clarity: float):
        """
        Update belief with an observation.

        Args:
            observed_class: The class that was observed
            clarity: P(observed_class is correct) — how clear the observation is
        """
        if observed_class not in self.class_names:
            return

        idx = self.class_names.index(observed_class)

        # Apply temperature scaling
        p_correct = clarity
        if self.temperature != 1.0:
            logit = math.log(max(p_correct / (1 - p_correct), 1e-10))
            logit_scaled = logit / self.temperature
            p_correct = 1.0 / (1.0 + math.exp(-logit_scaled))

        # Update Dirichlet parameters
        for i in range(self.n_classes):
            if i == idx:
                self.alpha[i] += p_correct
            else:
                self.alpha[i] += (1 - p_correct) / (self.n_classes - 1)

        # Apply min probability
        total = sum(self.alpha)
        self.alpha = [max(a, self.min_prob * total) for a in self.alpha]

        self.history.append(self.probs.copy())
        self.n_updates += 1

    def reset(self):
        """Reset to prior."""
        self.alpha = [self.prior_alpha] * self.n_classes
        self.history = [self.probs.copy()]
        self.n_updates = 0

    def top_k(self, k: int = 3) -> List[tuple]:
        """Top-k predictions with probabilities."""
        p = self.probs
        sorted_items = sorted(p.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    def summary(self) -> dict:
        return {
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "entropy": round(self.entropy, 4),
            "top3": [(name, round(prob, 4)) for name, prob in self.top_k(3)],
            "n_updates": self.n_updates,
        }
