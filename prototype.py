"""
APC — Adaptive Perception Controller
Prototype minimal : Classification binaire, 3 résolutions, programmation dynamique

Phase 1 du programme de recherche.
Utilise uniquement la bibliothèque standard Python (math, random, json).
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ============================================================
# 1. Modèle de problème
# ============================================================

@dataclass
class PerceptionAction:
    name: str
    cost: float
    clarity: float  # P(observation correcte | action)


@dataclass
class ToyProblem:
    prior: float
    actions: List[PerceptionAction]
    max_steps: int = 10

    @property
    def n_actions(self) -> int:
        return len(self.actions)


def default_problem() -> ToyProblem:
    """
    Problème jouet : « Le panneau affiche 50 ou 80 ? »

    Perte symétrique 0-1, mais avec un facteur d'amplification :
    - Erreur = 10 (coût de mauvaise décision)
    - Correct = 0
    - Bayes risk max = 5 (à prior=0.5)

    Actions avec coûts hétérogènes :
    - 320p : coût 0.5, clarity 0.70
    - 640p : coût 1.5, clarity 0.88
    - 1280p : coût 4.0, clarity 0.98
    """
    actions = [
        PerceptionAction(name="320p", cost=0.5, clarity=0.70),
        PerceptionAction(name="640p", cost=1.5, clarity=0.88),
        PerceptionAction(name="1280p", cost=4.0, clarity=0.98),
    ]
    return ToyProblem(prior=0.5, actions=actions, max_steps=8)


# ============================================================
# 2. Croyance et Bayes risk
# ============================================================

def belief_update(belief: float, obs: int, action: PerceptionAction) -> float:
    p_obs_y1 = action.clarity if obs == 1 else (1 - action.clarity)
    p_obs_y0 = (1 - action.clarity) if obs == 1 else action.clarity
    p_obs = p_obs_y1 * belief + p_obs_y0 * (1 - belief)
    if p_obs < 1e-15:
        return belief
    return max(1e-10, min(1 - 1e-10, (p_obs_y1 * belief) / p_obs))


def bayes_risk(belief: float) -> float:
    """
    Bayes risk avec perte 0-1 amplifiée :
    - Erreur = 10, Correct = 0
    - R(B) = 10 * min(B, 1-B)
    """
    return 10.0 * min(belief, 1 - belief)


def optimal_decision(belief: float) -> int:
    return 0 if belief < 0.5 else 1


# ============================================================
# 3. Programmation dynamique
# ============================================================

def solve_dp(problem: ToyProblem, n_bins: int = 200) -> Dict:
    beliefs = [0.01 + i * (0.98) / (n_bins - 1) for i in range(n_bins)]
    T = problem.max_steps
    n_a = problem.n_actions

    V = [[0.0] * n_bins for _ in range(T + 1)]
    policy = [[-1] * n_bins for _ in range(T)]

    for b_idx in range(n_bins):
        V[T][b_idx] = bayes_risk(beliefs[b_idx])

    for t in range(T - 1, -1, -1):
        for b_idx in range(n_bins):
            b = beliefs[b_idx]
            best_cost = bayes_risk(b)
            best_action = -1

            for a_idx, action in enumerate(problem.actions):
                expected_future = 0.0
                for obs in [0, 1]:
                    p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * b + \
                            ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - b)
                    new_b = belief_update(b, obs, action)
                    new_b_idx = min(range(n_bins), key=lambda i: abs(beliefs[i] - new_b))
                    expected_future += p_obs * V[t + 1][new_b_idx]

                total_cost = action.cost + expected_future
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_action = a_idx

            V[t][b_idx] = best_cost
            policy[t][b_idx] = best_action

    return {"V": V, "policy": policy, "beliefs": beliefs, "n_bins": n_bins}


def get_dp_action(sol: Dict, belief: float, t: int) -> int:
    b_idx = min(range(sol["n_bins"]), key=lambda i: abs(sol["beliefs"][i] - belief))
    return sol["policy"][t][b_idx]


def simulate_dp(sol: Dict, problem: ToyProblem, true_class: int) -> Dict:
    belief = problem.prior
    total_cost = 0.0
    actions_taken = []

    for t in range(problem.max_steps):
        a_idx = get_dp_action(sol, belief, t)
        if a_idx == -1:
            break
        action = problem.actions[a_idx]
        total_cost += action.cost
        obs = 1 if (true_class == 1 and random.random() < action.clarity) else \
              (0 if true_class == 0 and random.random() < action.clarity else \
               (1 if true_class == 1 else 0))
        if true_class == 1:
            obs = 1 if random.random() < action.clarity else 0
        else:
            obs = 0 if random.random() < action.clarity else 1
        belief = belief_update(belief, obs, action)
        actions_taken.append(action.name)

    decision = optimal_decision(belief)
    return {
        "correct": decision == true_class,
        "total_cost": total_cost,
        "final_belief": belief,
        "final_risk": bayes_risk(belief),
        "n_actions": len(actions_taken),
        "actions_taken": actions_taken,
    }


# ============================================================
# 4. Baselines
# ============================================================

def simulate_always_high(problem: ToyProblem, true_class: int) -> Dict:
    action = problem.actions[-1]
    if true_class == 1:
        obs = 1 if random.random() < action.clarity else 0
    else:
        obs = 0 if random.random() < action.clarity else 1
    belief = belief_update(problem.prior, obs, action)
    return {"correct": optimal_decision(belief) == true_class, "total_cost": action.cost,
            "final_belief": belief, "final_risk": bayes_risk(belief), "n_actions": 1,
            "actions_taken": [action.name]}


def simulate_always_low(problem: ToyProblem, true_class: int) -> Dict:
    action = problem.actions[0]
    if true_class == 1:
        obs = 1 if random.random() < action.clarity else 0
    else:
        obs = 0 if random.random() < action.clarity else 1
    belief = belief_update(problem.prior, obs, action)
    return {"correct": optimal_decision(belief) == true_class, "total_cost": action.cost,
            "final_belief": belief, "final_risk": bayes_risk(belief), "n_actions": 1,
            "actions_taken": [action.name]}


def simulate_confidence_threshold(problem: ToyProblem, true_class: int, threshold: float = 0.9) -> Dict:
    belief = problem.prior
    total_cost = 0.0
    actions_taken = []
    for _ in range(problem.max_steps):
        if max(belief, 1 - belief) >= threshold:
            break
        action = problem.actions[0]
        total_cost += action.cost
        if true_class == 1:
            obs = 1 if random.random() < action.clarity else 0
        else:
            obs = 0 if random.random() < action.clarity else 1
        belief = belief_update(belief, obs, action)
        actions_taken.append(action.name)
    decision = optimal_decision(belief)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


def simulate_info_gain(problem: ToyProblem, true_class: int, threshold: float = 0.9) -> Dict:
    belief = problem.prior
    total_cost = 0.0
    actions_taken = []
    for _ in range(problem.max_steps):
        if max(belief, 1 - belief) >= threshold:
            break
        best_ig = -1
        best_action = None
        for action in problem.actions:
            h_before = entropy(belief)
            ig = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * belief + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - belief)
                new_b = belief_update(belief, obs, action)
                ig += p_obs * (h_before - entropy(new_b))
            if ig > best_ig:
                best_ig = ig
                best_action = action
        total_cost += best_action.cost
        if true_class == 1:
            obs = 1 if random.random() < best_action.clarity else 0
        else:
            obs = 0 if random.random() < best_action.clarity else 1
        belief = belief_update(belief, obs, best_action)
        actions_taken.append(best_action.name)
    decision = optimal_decision(belief)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


def simulate_apc_greedy(problem: ToyProblem, true_class: int, threshold: float = 0.9) -> Dict:
    """APC : Glouton sur ΔR/C (réduction de risque par unité de coût)."""
    belief = problem.prior
    total_cost = 0.0
    actions_taken = []
    for _ in range(problem.max_steps):
        if max(belief, 1 - belief) >= threshold:
            break
        current_risk = bayes_risk(belief)
        best_score = -1
        best_action = None
        for action in problem.actions:
            expected_risk = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * belief + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - belief)
                new_b = belief_update(belief, obs, action)
                expected_risk += p_obs * bayes_risk(new_b)
            delta_risk = current_risk - expected_risk
            score = delta_risk / action.cost if action.cost > 0 else 0
            if score > best_score:
                best_score = score
                best_action = action
        total_cost += best_action.cost
        if true_class == 1:
            obs = 1 if random.random() < best_action.clarity else 0
        else:
            obs = 0 if random.random() < best_action.clarity else 1
        belief = belief_update(belief, obs, best_action)
        actions_taken.append(best_action.name)
    decision = optimal_decision(belief)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


def entropy(p: float) -> float:
    p = max(1e-10, min(1 - 1e-10, p))
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ============================================================
# 5. Mesures
# ============================================================

def compute_delta_risk_curve(problem: ToyProblem, n_bins: int = 100) -> Dict:
    beliefs = [0.01 + i * 0.98 / (n_bins - 1) for i in range(n_bins)]
    results = {}
    for action in problem.actions:
        delta_risks = []
        for b in beliefs:
            current_risk = bayes_risk(b)
            expected_risk = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * b + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - b)
                new_b = belief_update(b, obs, action)
                expected_risk += p_obs * bayes_risk(new_b)
            delta_risks.append(current_risk - expected_risk)
        dr = delta_risks
        results[action.name] = {
            "beliefs": beliefs,
            "delta_risks": dr,
            "cost": action.cost,
            "efficiency": [d / action.cost for d in dr],
        }
    return results


def run_experiment(n_trials: int = 10000, seed: int = 42) -> Tuple[Dict, Dict, ToyProblem]:
    random.seed(seed)
    problem = default_problem()

    print("Résolution par programmation dynamique...")
    dp_sol = solve_dp(problem, n_bins=200)
    print("  OK.")

    methods = {
        "DP (optimal)":     lambda tc: simulate_dp(dp_sol, problem, tc),
        "Always 1280p":     lambda tc: simulate_always_high(problem, tc),
        "Always 320p":      lambda tc: simulate_always_low(problem, tc),
        "Confidence (0.9)": lambda tc: simulate_confidence_threshold(problem, tc, 0.9),
        "Info-Gain":        lambda tc: simulate_info_gain(problem, tc, 0.9),
        "APC (ΔR/C)":       lambda tc: simulate_apc_greedy(problem, tc, 0.9),
    }

    results = {name: {"costs": [], "corrects": [], "risks": []} for name in methods}

    print(f"Exécution de {n_trials} essais par méthode...")
    for trial in range(n_trials):
        true_class = 1 if random.random() < problem.prior else 0
        for name, method in methods.items():
            res = method(true_class)
            results[name]["costs"].append(res["total_cost"])
            results[name]["corrects"].append(1 if res["correct"] else 0)
            results[name]["risks"].append(res["final_risk"])

    stats = {}
    for name, data in results.items():
        mc = sum(data["costs"]) / len(data["costs"])
        acc = sum(data["corrects"]) / len(data["corrects"])
        mr = sum(data["risks"]) / len(data["risks"])
        stats[name] = {
            "mean_cost": mc,
            "accuracy": acc,
            "mean_risk": mr,
            "epc": mc / max(acc, 1e-10),
        }
    return stats, dp_sol, problem


# ============================================================
# 6. Programme principal
# ============================================================

def main():
    print("=" * 65)
    print("APC — Adaptive Perception Controller — Prototype Phase 1")
    print("=" * 65)
    print()

    stats, dp_sol, problem = run_experiment(n_trials=10000)

    print()
    print("RÉSULTATS")
    print("-" * 65)
    print(f"{'Méthode':<22} {'Coût moy.':<10} {'Accuracy':<10} {'EPC':<10}")
    print("-" * 65)
    for name, s in stats.items():
        print(f"{name:<22} {s['mean_cost']:<10.2f} {s['accuracy']:<10.4f} {s['epc']:<10.2f}")

    print()
    print("ANALYSE ΔR(a|B_t) — Vérification de la submodularité")
    print("-" * 65)
    delta_curve = compute_delta_risk_curve(problem)
    for action_name, data in delta_curve.items():
        dr = data["delta_risks"]
        n = len(dr)
        mid = n // 2
        is_decreasing = all(dr[i] >= dr[i+1] - 1e-8 for i in range(mid, n-1))
        print(f"\nAction : {action_name} (coût = {data['cost']})")
        print(f"  ΔR max     : {max(dr):.6f}")
        print(f"  ΔR/C max   : {max(data['efficiency']):.6f}")
        print(f"  B à ΔR max : {data['beliefs'][dr.index(max(dr))]:.3f}")
        print(f"  ΔR(b=0.5)  : {dr[mid]:.6f}")
        print(f"  ΔR(b=0.1)  : {dr[0]:.6f}")
        print(f"  ΔR(b=0.9)  : {dr[-1]:.6f}")
        print(f"  Décroissant pour b > 0.5 : {'OUI' if is_decreasing else 'NON'}")

    # Frontière coût-risque
    print()
    print("FRONTIÈRE COÛT-RISQUE (seuil de confiance variable)")
    print("-" * 65)
    print(f"{'Seuil':<8} {'Coût':<10} {'Accuracy':<10} {'Risque':<10}")
    for thr_int in range(55, 100, 3):
        thr = thr_int / 100.0
        costs, accs, rsk = [], [], []
        for _ in range(3000):
            tc = 1 if random.random() < problem.prior else 0
            res = simulate_apc_greedy(problem, tc, thr)
            costs.append(res["total_cost"])
            accs.append(1 if res["correct"] else 0)
            rsk.append(res["final_risk"])
        mc = sum(costs) / len(costs)
        ac = sum(accs) / len(accs)
        mr = sum(rsk) / len(rsk)
        print(f"{thr:<8.2f} {mc:<10.2f} {ac:<10.4f} {mr:<10.6f}")

    print()
    print("=" * 65)
    print("FIN DU PROTOTYPE")
    print("=" * 65)


if __name__ == "__main__":
    main()
