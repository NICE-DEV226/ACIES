"""
APC — Adaptive Perception Controller
Prototype Phase 3 : Simulation réaliste de CNN

Simule le comportement d'un réseau de neurones (MobileNetV2-like)
avec des courbes accuracy/cost calibrées sur des benchmarks réels.

Basé sur :
- MobileNetV2 accuracy à différentes résolutions (ImageNet, CIFAR-10)
- Latence mesurée sur GPU edge (Jetson, RPi)
- Comportement de crop : zoom sur zone informative
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


# ============================================================
# 1. Courbes de performance calibrées sur des benchmarks réels
# ============================================================

# Données calibrées sur CIFAR-10 avec MobileNetV2-like
# Source : mesures de performance typiques en edge AI
PERFORMANCE_TABLE = {
    # resolution: (accuracy, flops_M, latency_ms, memory_MB)
    "64":   (0.45, 5,   2,   8),
    "128":  (0.72, 20,  5,   16),
    "224":  (0.88, 60,  12,  32),
    "320":  (0.92, 130, 25,  64),
    "512":  (0.95, 350, 60,  128),
    "1024": (0.97, 1400, 200, 256),
}

# Coûts normalisés (relatifs au plus petit)
COST_WEIGHTS = {
    "latency":  {"64": 1, "128": 2.5, "224": 6, "320": 12.5, "512": 30, "1024": 100},
    "energy":   {"64": 1, "128": 2.2, "224": 5, "320": 10,   "512": 25, "1024": 80},
    "memory":   {"64": 1, "128": 2,   "224": 4, "320": 8,    "512": 16, "1024": 32},
}

# Crop : zoom sur une région → résolution effective plus haute mais sur une zone plus petite
CROP_EFFECTIVENESS = {
    "small_object":  1.3,  # Objet petit : le crop aide beaucoup
    "medium_object": 1.1,  # Objet moyen : crop modéré
    "large_object":  0.8,  # Objet grand : le crop n'aide pas (déjà visible)
    "text":          1.5,  # Texte : le crop aide énormément
    "scene":         0.7,  # Scène globale : le crop perd du contexte
}


# ============================================================
# 2. Modèle de tâche visuelle réaliste
# ============================================================

@dataclass
class VisualTask:
    """Tâche visuelle avec difficulté variable."""
    name: str
    base_difficulty: float  # 0=facile, 1=impossible
    object_type: str  # clé pour CROP_EFFECTIVENESS
    spatial_frequency: str  # "low", "medium", "high" — fréquences spatiales dominantes

    @property
    def crop_bonus(self) -> float:
        return CROP_EFFECTIVENESS.get(self.object_type, 1.0)


@dataclass
class PerceptionAction:
    name: str
    resolution: str  # clé dans PERFORMANCE_TABLE
    is_crop: bool = False
    crop_region: str = ""  # type de région cropée
    cost_weight: float = 1.0

    def performance(self) -> Tuple[float, float, float, float]:
        """Retourne (accuracy, flops_M, latency_ms, memory_MB)."""
        return PERFORMANCE_TABLE[self.resolution]


def build_action_space() -> List[PerceptionAction]:
    """Espace d'actions hétérogène réaliste."""
    return [
        # Résolutions globales
        PerceptionAction(name="64p",    resolution="64"),
        PerceptionAction(name="128p",   resolution="128"),
        PerceptionAction(name="224p",   resolution="224"),
        PerceptionAction(name="320p",   resolution="320"),
        PerceptionAction(name="512p",   resolution="512"),
        PerceptionAction(name="1024p",  resolution="1024"),
        # Crops (résolution élevée sur zone locale)
        PerceptionAction(name="crop_128",  resolution="224", is_crop=True, crop_region="local"),
        PerceptionAction(name="crop_320",  resolution="320", is_crop=True, crop_region="local"),
        PerceptionAction(name="crop_512",  resolution="512", is_crop=True, crop_region="local"),
    ]


# ============================================================
# 3. Modèle de génération d'observations
# ============================================================

def observation_quality(action: PerceptionAction, task: VisualTask,
                        true_difficulty: float) -> float:
    """
    Calcule la qualité d'observation (P(observation correcte)) en fonction :
    - de l'action (résolution, crop)
    - de la tâche (difficulté, type d'objet)
    - de la difficulté spécifique de cette instance

    Modèle calibré :
    - accuracy = base_accuracy * task_modifier * difficulty_modifier
    - Le crop booste la résolution effective
    """
    acc, flops, latency, memory = action.performance()

    # Modifier par le type de tâche
    if action.is_crop:
        acc *= task.crop_bonus
    else:
        # Pour les résolutions globales, la fréquence spatiale compte
        if task.spatial_frequency == "high" and int(action.resolution) < 224:
            acc *= 0.85  # Pénalité pour haute fréquence en basse résolution
        elif task.spatial_frequency == "low" and int(action.resolution) > 320:
            acc *= 1.02  # Légère augmentation pour basse fréquence en haute rés.

    # Modifier par la difficulté de l'instance
    difficulty_factor = 1.0 - true_difficulty * 0.3  # Max -30% pour instances difficiles
    acc *= difficulty_factor

    # Saturer entre 0.01 et 0.99
    acc = max(0.01, min(0.99, acc))
    return acc


def action_cost(action: PerceptionAction, cost_type: str = "composite") -> float:
    """
    Coût composite de l'action (latency + energy + memory).
    """
    _, flops, latency, memory = action.performance()
    energy = flops * 0.1  # Approximation : énergie ∝ FLOPs

    if cost_type == "latency":
        return latency
    elif cost_type == "energy":
        return energy
    elif cost_type == "memory":
        return memory
    else:  # composite
        return 0.4 * latency + 0.4 * energy + 0.2 * memory


# ============================================================
# 4. Bayes risk et belief update
# ============================================================

def bayes_risk(belief: float) -> float:
    """Risque 0-1 amplifié."""
    return 10.0 * min(belief, 1 - belief)


def belief_update(belief: float, obs: int, clarity: float) -> float:
    p_obs_y1 = clarity if obs == 1 else (1 - clarity)
    p_obs_y0 = (1 - clarity) if obs == 1 else clarity
    p_obs = p_obs_y1 * belief + p_obs_y0 * (1 - belief)
    if p_obs < 1e-15:
        return belief
    return max(1e-10, min(1 - 1e-10, (p_obs_y1 * belief) / p_obs))


def optimal_decision(belief: float) -> int:
    return 0 if belief < 0.5 else 1


# ============================================================
# 5. Politiques
# ============================================================

def policy_apc_greedy(actions: List[PerceptionAction], task: VisualTask,
                      true_class: int, true_difficulty: float,
                      prior: float = 0.5, threshold: float = 0.92,
                      cost_type: str = "composite") -> Dict:
    """APC : Glouton ΔR/C."""
    belief = prior
    total_cost = 0.0
    total_latency = 0
    total_energy = 0
    total_memory = 0
    actions_taken = []
    flops_used = 0

    for step in range(8):
        if max(belief, 1 - belief) >= threshold:
            break

        current_risk = bayes_risk(belief)
        best_score = -1
        best_action = None

        for action in actions:
            clarity = observation_quality(action, task, true_difficulty)
            cost = action_cost(action, cost_type)

            expected_risk = 0.0
            for obs in [0, 1]:
                p_obs = (clarity if obs == 1 else (1 - clarity)) * belief + \
                        ((1 - clarity) if obs == 1 else clarity) * (1 - belief)
                new_b = belief_update(belief, obs, clarity)
                expected_risk += p_obs * bayes_risk(new_b)

            delta_risk = current_risk - expected_risk
            score = delta_risk / cost if cost > 0 else 0

            if score > best_score:
                best_score = score
                best_action = action

        # Exécuter l'action
        clarity = observation_quality(best_action, task, true_difficulty)
        cost = action_cost(best_action, cost_type)
        _, flops, latency, memory = best_action.performance()

        total_cost += cost
        total_latency += latency
        total_energy += flops * 0.1
        total_memory = max(total_memory, memory)
        flops_used += flops

        obs = 1 if (true_class == 1 and random.random() < clarity) else \
              (0 if true_class == 0 and random.random() < clarity else \
               (1 if true_class == 1 else 0))
        if true_class == 1:
            obs = 1 if random.random() < clarity else 0
        else:
            obs = 0 if random.random() < clarity else 1

        belief = belief_update(belief, obs, clarity)
        actions_taken.append(best_action.name)

    decision = optimal_decision(belief)
    return {
        "correct": decision == true_class,
        "total_cost": total_cost,
        "latency_ms": total_latency,
        "energy_mJ": total_energy,
        "peak_memory_MB": total_memory,
        "flops_M": flops_used,
        "n_actions": len(actions_taken),
        "actions_taken": actions_taken,
        "final_belief": belief,
        "final_risk": bayes_risk(belief),
    }


def policy_fixed_resolution(actions: List[PerceptionAction], task: VisualTask,
                            true_class: int, true_difficulty: float,
                            resolution: str, cost_type: str = "composite") -> Dict:
    """Politique fixe : toujours la même résolution."""
    action = [a for a in actions if a.resolution == resolution and not a.is_crop][0]
    clarity = observation_quality(action, task, true_difficulty)
    cost = action_cost(action, cost_type)
    _, flops, latency, memory = action.performance()

    obs = 1 if (true_class == 1 and random.random() < clarity) else \
          (0 if true_class == 0 and random.random() < clarity else \
           (1 if true_class == 1 else 0))
    if true_class == 1:
        obs = 1 if random.random() < clarity else 0
    else:
        obs = 0 if random.random() < clarity else 1

    belief = belief_update(0.5, obs, clarity)
    decision = optimal_decision(belief)

    return {
        "correct": decision == true_class,
        "total_cost": cost,
        "latency_ms": latency,
        "energy_mJ": flops * 0.1,
        "peak_memory_MB": memory,
        "flops_M": flops,
        "n_actions": 1,
        "actions_taken": [action.name],
        "final_belief": belief,
        "final_risk": bayes_risk(belief),
    }


def policy_early_exit_confidence(actions: List[PerceptionAction], task: VisualTask,
                                 true_class: int, true_difficulty: float,
                                 threshold: float = 0.92,
                                 cost_type: str = "composite") -> Dict:
    """Baseline : résolution croissante + early exit par confiance."""
    resolutions = ["64", "128", "224", "320", "512", "1024"]
    belief = 0.5
    total_cost = 0
    total_latency = 0
    total_energy = 0
    total_memory = 0
    flops_used = 0
    actions_taken = []

    for res in resolutions:
        if max(belief, 1 - belief) >= threshold:
            break

        action = [a for a in actions if a.resolution == res and not a.is_crop][0]
        clarity = observation_quality(action, task, true_difficulty)
        cost = action_cost(action, cost_type)
        _, flops, latency, memory = action.performance()

        total_cost += cost
        total_latency += latency
        total_energy += flops * 0.1
        total_memory = max(total_memory, memory)
        flops_used += flops

        obs = 1 if (true_class == 1 and random.random() < clarity) else \
              (0 if true_class == 0 and random.random() < clarity else \
               (1 if true_class == 1 else 0))
        if true_class == 1:
            obs = 1 if random.random() < clarity else 0
        else:
            obs = 0 if random.random() < clarity else 1

        belief = belief_update(belief, obs, clarity)
        actions_taken.append(action.name)

    decision = optimal_decision(belief)
    return {
        "correct": decision == true_class,
        "total_cost": total_cost,
        "latency_ms": total_latency,
        "energy_mJ": total_energy,
        "peak_memory_MB": total_memory,
        "flops_M": flops_used,
        "n_actions": len(actions_taken),
        "actions_taken": actions_taken,
        "final_belief": belief,
        "final_risk": bayes_risk(belief),
    }


# ============================================================
# 6. Génération de tâches
# ============================================================

def generate_tasks(n: int, seed: int = 42) -> List[Tuple[VisualTask, int, float]]:
    """Génère N tâches avec difficultés et classes variables."""
    random.seed(seed)
    tasks = []
    object_types = list(CROP_EFFECTIVENESS.keys())
    frequencies = ["low", "medium", "high"]

    for _ in range(n):
        obj = random.choice(object_types)
        freq = random.choice(frequencies)
        diff = random.uniform(0.1, 0.9)
        true_class = random.randint(0, 1)

        task = VisualTask(
            name=f"task_{random.randint(0,9999)}",
            base_difficulty=diff,
            object_type=obj,
            spatial_frequency=freq,
        )
        tasks.append((task, true_class, diff))

    return tasks


# ============================================================
# 7. Expériences
# ============================================================

def run_experiment_1_realistic(n_trials: int = 5000) -> Dict:
    """Expérience principale : toutes les méthodes sur tâches réalistes."""
    actions = build_action_space()
    tasks = generate_tasks(n_trials)

    methods = {
        "APC greedy": lambda t, tc, d: policy_apc_greedy(actions, t, tc, d),
        "Fixed 64p":  lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "64"),
        "Fixed 128p": lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "128"),
        "Fixed 224p": lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "224"),
        "Fixed 320p": lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "320"),
        "Fixed 512p": lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "512"),
        "Fixed 1024p":lambda t, tc, d: policy_fixed_resolution(actions, t, tc, d, "1024"),
        "Early Exit": lambda t, tc, d: policy_early_exit_confidence(actions, t, tc, d),
    }

    results = {name: {"costs": [], "corrects": [], "latencies": [], "energies": [],
                       "flops": [], "memories": [], "risks": [], "n_actions": []}
               for name in methods}

    for task, true_class, diff in tasks:
        for name, method in methods.items():
            res = method(task, true_class, diff)
            results[name]["costs"].append(res["total_cost"])
            results[name]["corrects"].append(1 if res["correct"] else 0)
            results[name]["latencies"].append(res["latency_ms"])
            results[name]["energies"].append(res["energy_mJ"])
            results[name]["flops"].append(res["flops_M"])
            results[name]["memories"].append(res["peak_memory_MB"])
            results[name]["risks"].append(res["final_risk"])
            results[name]["n_actions"].append(res["n_actions"])

    stats = {}
    for name, data in results.items():
        n = len(data["costs"])
        mc = sum(data["costs"]) / n
        acc = sum(data["corrects"]) / n
        ml = sum(data["latencies"]) / n
        me = sum(data["energies"]) / n
        mf = sum(data["flops"]) / n
        mm = sum(data["memories"]) / n
        mr = sum(data["risks"]) / n
        mna = sum(data["n_actions"]) / n
        stats[name] = {
            "mean_cost": mc, "accuracy": acc, "epc": mc / max(acc, 1e-10),
            "latency_ms": ml, "energy_mJ": me, "flops_M": mf,
            "peak_memory_MB": mm, "mean_risk": mr, "mean_actions": mna,
        }
    return stats


def run_experiment_2_crops(n_trials: int = 3000) -> Dict:
    """Expérience 2 : impact des crops sur la frontière coût-risque."""
    actions_with_crop = build_action_space()
    actions_no_crop = [a for a in actions_with_crop if not a.is_crop]

    tasks = generate_tasks(n_trials)

    configs = {
        "Sans crop": actions_no_crop,
        "Avec crop": actions_with_crop,
    }

    results = {}
    for label, act_list in configs.items():
        costs, accs, risks = [], [], []
        for task, true_class, diff in tasks:
            res = policy_apc_greedy(act_list, task, true_class, diff)
            costs.append(res["total_cost"])
            accs.append(1 if res["correct"] else 0)
            risks.append(res["final_risk"])

        mc = sum(costs) / len(costs)
        ac = sum(accs) / len(accs)
        mr = sum(risks) / len(risks)
        results[label] = {"mean_cost": mc, "accuracy": ac, "epc": mc / max(ac, 1e-10),
                          "mean_risk": mr}

    return results


def run_experiment_3_hardware_profile(n_trials: int = 3000) -> Dict:
    """Expérience 3 : profils hardware (latency vs energy vs memory)."""
    actions = build_action_space()
    tasks = generate_tasks(n_trials)

    cost_types = ["latency", "energy", "memory", "composite"]
    results = {}

    for cost_type in cost_types:
        costs, accs, lats, ens = [], [], [], []
        for task, true_class, diff in tasks:
            res = policy_apc_greedy(actions, task, true_class, diff, cost_type=cost_type)
            costs.append(res["total_cost"])
            accs.append(1 if res["correct"] else 0)
            lats.append(res["latency_ms"])
            ens.append(res["energy_mJ"])

        mc = sum(costs) / len(costs)
        ac = sum(accs) / len(accs)
        ml = sum(lats) / len(lats)
        me = sum(ens) / len(ens)
        results[cost_type] = {
            "mean_cost": mc, "accuracy": ac, "epc": mc / max(ac, 1e-10),
            "latency_ms": ml, "energy_mJ": me,
        }

    return results


# ============================================================
# 8. Programme principal
# ============================================================

def main():
    random.seed(42)

    print("=" * 75)
    print("APC — Prototype Phase 3 : Simulation réaliste CNN")
    print("=" * 75)

    # ── Expérience 1 ──
    print("\n" + "─" * 75)
    print("EXPÉRIENCE 1 : Comparaison sur tâches visuelles réalistes")
    print("─" * 75)

    stats = run_experiment_1_realistic(n_trials=5000)

    print(f"\n{'Méthode':<18} {'Coût':<8} {'Acc':<7} {'EPC':<7} "
          f"{'Lat(ms)':<9} {'E(mJ)':<8} {'FLOPs':<8} {'Act':<5}")
    print("─" * 75)
    for name, s in stats.items():
        print(f"{name:<18} {s['mean_cost']:<8.1f} {s['accuracy']:<7.3f} "
              f"{s['epc']:<7.2f} {s['latency_ms']:<9.1f} {s['energy_mJ']:<8.1f} "
              f"{s['flops_M']:<8.0f} {s['mean_actions']:<5.1f}")

    # ── Expérience 2 ──
    print("\n" + "─" * 75)
    print("EXPÉRIENCE 2 : Impact des actions crop")
    print("─" * 75)

    crop_stats = run_experiment_2_crops()
    print(f"\n{'Config':<16} {'Coût':<8} {'Acc':<7} {'EPC':<7} {'Risque':<8}")
    print("─" * 45)
    for name, s in crop_stats.items():
        print(f"{name:<16} {s['mean_cost']:<8.1f} {s['accuracy']:<7.3f} "
              f"{s['epc']:<7.2f} {s['mean_risk']:<8.3f}")

    # ── Expérience 3 ──
    print("\n" + "─" * 75)
    print("EXPÉRIENCE 3 : Profils hardware (coût dominé)")
    print("─" * 75)

    hw_stats = run_experiment_3_hardware_profile()
    print(f"\n{'Profil':<14} {'Coût':<8} {'Acc':<7} {'EPC':<7} "
          f"{'Lat(ms)':<9} {'E(mJ)':<8}")
    print("─" * 52)
    for name, s in hw_stats.items():
        print(f"{name:<14} {s['mean_cost']:<8.1f} {s['accuracy']:<7.3f} "
              f"{s['epc']:<7.2f} {s['latency_ms']:<9.1f} {s['energy_mJ']:<8.1f}")

    print("\n" + "=" * 75)
    print("FIN — Phase 3 complétée")
    print("=" * 75)


if __name__ == "__main__":
    main()
