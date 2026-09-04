"""
ACIES — Action Space

Définit les actions perceptuelles hétérogènes avec leurs propriétés.
Chaque action est un tuple (type, paramètres, coût estimé).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Tuple
import math


class ActionType(Enum):
    RESOLUTION = auto()   # Changer la résolution globale
    CROP = auto()         # Zoom sur une région
    FRAME = auto()        # Prendre une frame supplémentaire
    LAYERS = auto()       # Activer/désactiver des couches
    TOKENS = auto()       # Nombre de tokens à traiter
    MODALITY = auto()     # Activer une modalité supplémentaire


@dataclass(frozen=True)
class Action:
    """
    Une action perceptuelle immutable.

    Attributes:
        id: Identifiant unique (index dans l'espace d'actions)
        name: Nom lisible
        action_type: Type d'action
        params: Paramètres spécifiques (résolution, coordonnées crop, etc.)
        base_latency_ms: Latence de base sur hardware de référence
        base_energy_mJ: Énergie de base
        base_memory_MB: Mémoire de base
        pixel_ratio: Ratio de pixels traités (0.0-1.0) par rapport au max
    """
    id: int
    name: str
    action_type: ActionType
    params: Dict[str, float] = field(default_factory=dict)
    base_latency_ms: float = 0.0
    base_energy_mJ: float = 0.0
    base_memory_MB: float = 0.0
    pixel_ratio: float = 1.0

    @property
    def resolution(self) -> int:
        return int(self.params.get("resolution", 224))

    @property
    def crop_area_ratio(self) -> float:
        """Ratio de la zone cropée par rapport à l'image entière."""
        return self.params.get("crop_area_ratio", 1.0)

    def cost(self, profile: "HardwareProfile") -> float:
        """Coût composite sur un hardware donné."""
        return (profile.latency_weight * self.base_latency_ms * profile.latency_scale +
                profile.energy_weight * self.base_energy_mJ * profile.energy_scale +
                profile.memory_weight * self.base_memory_MB * profile.memory_scale)


@dataclass(frozen=True)
class HardwareProfile:
    """
    Profil hardware — définit les poids et échelles de coût.
    Permet d'adapter le contrôleur à différents devices.
    """
    name: str
    latency_weight: float = 0.4
    energy_weight: float = 0.4
    memory_weight: float = 0.2
    latency_scale: float = 1.0    # Multiplicateur (GPU rapide = 0.5, lent = 2.0)
    energy_scale: float = 1.0
    memory_scale: float = 1.0

    @staticmethod
    def jetson_orin() -> "HardwareProfile":
        return HardwareProfile(
            name="Jetson Orin Nano",
            latency_weight=0.5, energy_weight=0.3, memory_weight=0.2,
            latency_scale=0.6, energy_scale=0.8, memory_scale=1.0,
        )

    @staticmethod
    def raspberry_pi5() -> "HardwareProfile":
        return HardwareProfile(
            name="Raspberry Pi 5",
            latency_weight=0.3, energy_weight=0.5, memory_weight=0.2,
            latency_scale=2.5, energy_scale=0.4, memory_scale=0.8,
        )

    @staticmethod
    def desktop_gpu() -> "HardwareProfile":
        return HardwareProfile(
            name="Desktop GPU (RTX 4090)",
            latency_weight=0.6, energy_weight=0.2, memory_weight=0.2,
            latency_scale=0.1, energy_scale=3.0, memory_scale=2.0,
        )

    @staticmethod
    def edge_tpu() -> "HardwareProfile":
        return HardwareProfile(
            name="Edge TPU (Coral)",
            latency_weight=0.4, energy_weight=0.5, memory_weight=0.1,
            latency_scale=0.3, energy_scale=0.1, memory_scale=0.3,
        )

    @staticmethod
    def default() -> "HardwareProfile":
        return HardwareProfile(name="Default")


# ============================================================
# Espace d'actions standard
# ============================================================

def build_standard_actions() -> list:
    """
    Espace d'actions standard pour la classification visuelle.
    9 actions hétérogènes : résolutions + crops.
    """
    actions = []
    idx = 0

    # Résolutions globales
    for res, latency, energy, memory in [
        (64,   2,   0.5,  8),
        (128,  5,   2,   16),
        (224,  12,  6,   32),
        (320,  25,  13,  64),
        (512,  60,  35,  128),
        (1024, 200, 140, 256),
    ]:
        pixel_ratio = (res / 1024) ** 2
        actions.append(Action(
            id=idx, name=f"{res}p", action_type=ActionType.RESOLUTION,
            params={"resolution": float(res), "crop_area_ratio": 1.0},
            base_latency_ms=latency, base_energy_mJ=energy,
            base_memory_MB=memory, pixel_ratio=pixel_ratio,
        ))
        idx += 1

    # Crops (haute résolution sur zone locale)
    for crop_res, crop_area, latency, energy, memory in [
        (224, 0.05, 8,   4,   24),   # crop_128 : zone 5% de l'image
        (320, 0.08, 15,  8,   40),   # crop_320 : zone 8%
        (512, 0.12, 35,  20,  80),   # crop_512 : zone 12%
    ]:
        pixel_ratio = crop_area * (crop_res / 1024) ** 2
        actions.append(Action(
            id=idx, name=f"crop_{crop_res}", action_type=ActionType.CROP,
            params={"resolution": float(crop_res), "crop_area_ratio": crop_area},
            base_latency_ms=latency, base_energy_mJ=energy,
            base_memory_MB=memory, pixel_ratio=pixel_ratio,
        ))
        idx += 1

    return actions
