"""
ACIES C++ Accelerator

Wrapper Python via ctypes pour la bibliothèque C++ native.
Fournit les mêmes interfaces que les modules Python pur,
mais avec des opérations critiques en C++ (belief, Thompson Sampling).

Usage:
    from acies加速 import BeliefState, ClarityLearner
    b = BeliefState(prior=0.5)
    b.update(obs=1, clarity=0.85)
"""

import ctypes
import os
import math

# Charger la bibliothèque partagée
_LIB_PATH = os.path.join(os.path.dirname(__file__), "..", "cpp", "libacies.so")
_lib = ctypes.CDLL(_LIB_PATH)

# ---- BeliefState ----

_lib.acies_belief_create.restype = ctypes.c_void_p
_lib.acies_belief_create.argtypes = [ctypes.c_double, ctypes.c_double]
_lib.acies_belief_destroy.restype = None
_lib.acies_belief_destroy.argtypes = [ctypes.c_void_p]
_lib.acies_belief_reset.restype = None
_lib.acies_belief_reset.argtypes = [ctypes.c_void_p]
_lib.acies_belief_update.restype = None
_lib.acies_belief_update.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
_lib.acies_belief_risk.restype = ctypes.c_double
_lib.acies_belief_risk.argtypes = [ctypes.c_void_p]
_lib.acies_belief_confidence.restype = ctypes.c_double
_lib.acies_belief_confidence.argtypes = [ctypes.c_void_p]
_lib.acies_belief_decision.restype = ctypes.c_int
_lib.acies_belief_decision.argtypes = [ctypes.c_void_p]
_lib.acies_belief_delta_risk.restype = ctypes.c_double
_lib.acies_belief_delta_risk.argtypes = [ctypes.c_void_p, ctypes.c_double]
_lib.acies_belief_delta_risk_efficiency.restype = ctypes.c_double
_lib.acies_belief_delta_risk_efficiency.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]

# ---- ClarityLearner ----

_lib.acies_learner_create.restype = ctypes.c_void_p
_lib.acies_learner_create.argtypes = [ctypes.c_int]
_lib.acies_learner_destroy.restype = None
_lib.acies_learner_destroy.argtypes = [ctypes.c_void_p]
_lib.acies_learner_reset.restype = None
_lib.acies_learner_reset.argtypes = [ctypes.c_void_p]
_lib.acies_learner_reset_posterior.restype = None
_lib.acies_learner_reset_posterior.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.acies_learner_sample.restype = ctypes.c_double
_lib.acies_learner_sample.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.acies_learner_update.restype = None
_lib.acies_learner_update.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.acies_learner_mean.restype = ctypes.c_double
_lib.acies_learner_mean.argtypes = [ctypes.c_void_p, ctypes.c_int]


class BeliefState:
    """Wrapper C++ pour BeliefState."""

    def __init__(self, prior=0.5, temperature=1.0):
        self._ptr = _lib.acies_belief_create(prior, temperature)

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.acies_belief_destroy(self._ptr)

    def reset(self):
        _lib.acies_belief_reset(self._ptr)

    def update(self, obs, clarity):
        _lib.acies_belief_update(self._ptr, obs, clarity)

    @property
    def risk(self):
        return _lib.acies_belief_risk(self._ptr)

    @property
    def confidence(self):
        return _lib.acies_belief_confidence(self._ptr)

    @property
    def decision(self):
        return _lib.acies_belief_decision(self._ptr)

    def delta_risk(self, clarity):
        return _lib.acies_belief_delta_risk(self._ptr, clarity)

    def delta_risk_efficiency(self, clarity, cost):
        return _lib.acies_belief_delta_risk_efficiency(self._ptr, clarity, cost)


class ClarityLearner:
    """Wrapper C++ pour ClarityLearner."""

    def __init__(self, n_actions):
        self._ptr = _lib.acies_learner_create(n_actions)
        self.n_actions = n_actions

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.acies_learner_destroy(self._ptr)

    def reset(self):
        _lib.acies_learner_reset(self._ptr)

    def reset_posterior(self, action_id):
        _lib.acies_learner_reset_posterior(self._ptr, action_id)

    def sample(self, action_id):
        return _lib.acies_learner_sample(self._ptr, action_id)

    def update(self, action_id, correct):
        _lib.acies_learner_update(self._ptr, action_id, 1 if correct else 0)

    def mean(self, action_id):
        return _lib.acies_learner_mean(self._ptr, action_id)
