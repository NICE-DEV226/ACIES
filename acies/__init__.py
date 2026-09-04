"""
ACIES — Adaptive Perception Control

A decision-theoretic framework for adaptively controlling visual perception
to minimize computational cost while maintaining target decision risk.

Usage:
    from acies import APCController, APCConfig, HardwareProfile

    apc = APCController(APCConfig(
        confidence_threshold=0.92,
        hardware=HardwareProfile.jetson_orin(),
    ))
    result = apc.run(true_class=1, clarity_fn=my_clarity_fn)
"""

from .actions import Action, ActionType, HardwareProfile, build_standard_actions
from .belief import BeliefState
from .clarity_learner import ClarityLearner, BetaPosterior
from .safety import SafetyLayer, SafetyConfig, SafetyState
from .conviction import Conviction, ConvictionConfig, ConvictionState
from .change_point import ChangePointDetector, ChangePointConfig
from .controller import APCController, APCConfig, APCResult, APCStep

__all__ = [
    "Action", "ActionType", "HardwareProfile", "build_standard_actions",
    "BeliefState",
    "ClarityLearner", "BetaPosterior",
    "SafetyLayer", "SafetyConfig", "SafetyState",
    "Conviction", "ConvictionConfig", "ConvictionState",
    "ChangePointDetector", "ChangePointConfig",
    "APCController", "APCConfig", "APCResult", "APCStep",
]

__version__ = "0.1.0"
