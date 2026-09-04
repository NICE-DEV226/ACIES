"""
APC — Adaptive Perception Controller
Prototype Phase 2 : Loss functions, actions crop, bornes théoriques

Extensions :
- Comparaison 0-1 loss vs squared error vs log loss
- Actions crop ciblées
- Prior variable (quantifier le gain d'adaptativité)
- Borne de Chernoff vs optimal DP
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ============================================================
# 1. Loss functions
# ============================================================

def loss_01(y: int, delta: int) -> float:
    """0-1 loss amplifiée."""
    return 10.0 if y != delta else 0.0

def loss_squared(y: int, delta: int, belief: float) -> float:
    """
    Erreur quadratique : L(y, δ) = (y - δ)²
    Bayes risk : R(B) = B(1-B) — concave, submodulaire
    """
    return (y - delta) ** 2

def loss_log(y: int, delta: int, belief: float) -> float:
    """
    Log loss (cross-entropy) : L(y, δ) = -log(P(y|δ))
    Bayes risk : h₂(B) — entropie binaire, concave, submodulaire
    """
    p = max(1e-10, min(1 - 1e-10, belief if delta == 1 else 1 - belief))
    return -math.log(p)


def bayes_risk(belief: float, loss_type: str = "01") -> float:
    """Bayes risk pour chaque type de loss, tous sur échelle comparable."""
    if loss_type == "01":
        return 10.0 * min(belief, 1 - belief)  # max = 5.0
    elif loss_type == "squared":
        return 20.0 * belief * (1 - belief)  # max = 5.0 (×20 pour même échelle)
    elif loss_type == "log":
        p = max(1e-10, min(1 - 1e-10, belief))
        h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        return 10.0 * h / math.log(2)  # max = 5.0 (×10/log2 pour même échelle)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


# ============================================================
# 2. Modèle de problème
# ============================================================

@dataclass
class PerceptionAction:
    name: str
    cost: float
    clarity: float
    action_type: str = "resolution"  # "resolution" ou "crop"


def default_actions() -> List[PerceptionAction]:
    """Actions résolution + crop."""
    return [
        PerceptionAction(name="320p", cost=0.5, clarity=0.70, action_type="resolution"),
        PerceptionAction(name="640p", cost=1.5, clarity=0.88, action_type="resolution"),
        PerceptionAction(name="1280p", cost=4.0, clarity=0.98, action_type="resolution"),
        PerceptionAction(name="crop_320p", cost=0.8, clarity=0.92, action_type="crop"),
        PerceptionAction(name="crop_640p", cost=2.0, clarity=0.96, action_type="crop"),
    ]


def belief_update(belief: float, obs: int, clarity: float) -> float:
    p_obs_y1 = clarity if obs == 1 else (1 - clarity)
    p_obs_y0 = (1 - clarity) if obs == 1 else clarity
    p_obs = p_obs_y1 * belief + p_obs_y0 * (1 - belief)
    if p_obs < 1e-15:
        return belief
    return max(1e-10, min(1 - 1e-10, (p_obs_y1 * belief) / p_obs))


def optimal_decision(belief: float, loss_type: str) -> int:
    """Décision optimale selon la loss."""
    if loss_type == "01":
        return 0 if belief < 0.5 else 1
    elif loss_type == "squared":
        # Minimiser E[(Y-δ)²] = B si δ=0, 1-B si δ=1
        return 0 if belief < 0.5 else 1
    elif loss_type == "log":
        return 0 if belief < 0.5 else 1
    return 0 if belief < 0.5 else 1


# ============================================================
# 3. Programmation dynamique
# ============================================================

def solve_dp(actions: List[PerceptionAction], prior: float, loss_type: str,
             max_steps: int = 8, n_bins: int = 200) -> Dict:
    beliefs = [0.01 + i * 0.98 / (n_bins - 1) for i in range(n_bins)]
    T = max_steps

    V = [[0.0] * n_bins for _ in range(T + 1)]
    policy = [[-1] * n_bins for _ in range(T)]

    for b_idx in range(n_bins):
        V[T][b_idx] = bayes_risk(beliefs[b_idx], loss_type)

    for t in range(T - 1, -1, -1):
        for b_idx in range(n_bins):
            b = beliefs[b_idx]
            best_cost = bayes_risk(b, loss_type)
            best_action = -1

            for a_idx, action in enumerate(actions):
                expected_future = 0.0
                for obs in [0, 1]:
                    p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * b + \
                            ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - b)
                    new_b = belief_update(b, obs, action.clarity)
                    new_b_idx = min(range(n_bins), key=lambda i: abs(beliefs[i] - new_b))
                    expected_future += p_obs * V[t + 1][new_b_idx]

                total_cost = action.cost + expected_future
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_action = a_idx

            V[t][b_idx] = best_cost
            policy[t][b_idx] = best_action

    return {"V": V, "policy": policy, "beliefs": beliefs, "n_bins": n_bins}


def simulate_dp(sol: Dict, actions: List[PerceptionAction], prior: float,
                true_class: int, loss_type: str) -> Dict:
    belief = prior
    total_cost = 0.0
    actions_taken = []

    for t in range(len(sol["V"]) - 1):
        b_idx = min(range(sol["n_bins"]), key=lambda i: abs(sol["beliefs"][i] - belief))
        a_idx = sol["policy"][t][b_idx]
        if a_idx == -1:
            break
        action = actions[a_idx]
        total_cost += action.cost
        obs = 1 if (true_class == 1 and random.random() < action.clarity) else \
              (0 if true_class == 0 and random.random() < action.clarity else \
               (1 if true_class == 1 else 0))
        if true_class == 1:
            obs = 1 if random.random() < action.clarity else 0
        else:
            obs = 0 if random.random() < action.clarity else 1
        belief = belief_update(belief, obs, action.clarity)
        actions_taken.append(action.name)

    decision = optimal_decision(belief, loss_type)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief, loss_type),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


# ============================================================
# 4. Baselines
# ============================================================

def simulate_always_best(actions: List[PerceptionAction], prior: float,
                         true_class: int, loss_type: str) -> Dict:
    action = max(actions, key=lambda a: a.clarity)
    obs = 1 if (true_class == 1 and random.random() < action.clarity) else \
          (0 if true_class == 0 and random.random() < action.clarity else \
           (1 if true_class == 1 else 0))
    if true_class == 1:
        obs = 1 if random.random() < action.clarity else 0
    else:
        obs = 0 if random.random() < action.clarity else 1
    belief = belief_update(prior, obs, action.clarity)
    return {"correct": optimal_decision(belief, loss_type) == true_class,
            "total_cost": action.cost, "final_risk": bayes_risk(belief, loss_type),
            "final_belief": belief, "n_actions": 1, "actions_taken": [action.name]}


def simulate_always_cheapest(actions: List[PerceptionAction], prior: float,
                             true_class: int, loss_type: str) -> Dict:
    action = min(actions, key=lambda a: a.cost)
    obs = 1 if (true_class == 1 and random.random() < action.clarity) else \
          (0 if true_class == 0 and random.random() < action.clarity else \
           (1 if true_class == 1 else 0))
    if true_class == 1:
        obs = 1 if random.random() < action.clarity else 0
    else:
        obs = 0 if random.random() < action.clarity else 1
    belief = belief_update(prior, obs, action.clarity)
    return {"correct": optimal_decision(belief, loss_type) == true_class,
            "total_cost": action.cost, "final_risk": bayes_risk(belief, loss_type),
            "final_belief": belief, "n_actions": 1, "actions_taken": [action.name]}


def simulate_apc_greedy(actions: List[PerceptionAction], prior: float,
                        true_class: int, loss_type: str, threshold: float = 0.9) -> Dict:
    belief = prior
    total_cost = 0.0
    actions_taken = []
    max_steps = 8
    for _ in range(max_steps):
        if max(belief, 1 - belief) >= threshold:
            break
        current_risk = bayes_risk(belief, loss_type)
        best_score = -1
        best_action = None
        for action in actions:
            expected_risk = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * belief + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - belief)
                new_b = belief_update(belief, obs, action.clarity)
                expected_risk += p_obs * bayes_risk(new_b, loss_type)
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
        belief = belief_update(belief, obs, best_action.clarity)
        actions_taken.append(best_action.name)
    decision = optimal_decision(belief, loss_type)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief, loss_type),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


def simulate_info_gain(actions: List[PerceptionAction], prior: float,
                       true_class: int, loss_type: str, threshold: float = 0.9) -> Dict:
    belief = prior
    total_cost = 0.0
    actions_taken = []
    for _ in range(8):
        if max(belief, 1 - belief) >= threshold:
            break
        best_ig = -1
        best_action = None
        for action in actions:
            h_before = -(belief * math.log(belief + 1e-10) + (1 - belief) * math.log(1 - belief + 1e-10))
            ig = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * belief + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - belief)
                new_b = belief_update(belief, obs, action.clarity)
                h_after = -(new_b * math.log(new_b + 1e-10) + (1 - new_b) * math.log(1 - new_b + 1e-10))
                ig += p_obs * (h_before - h_after)
            if ig > best_ig:
                best_ig = ig
                best_action = action
        total_cost += best_action.cost
        if true_class == 1:
            obs = 1 if random.random() < best_action.clarity else 0
        else:
            obs = 0 if random.random() < best_action.clarity else 1
        belief = belief_update(belief, obs, best_action.clarity)
        actions_taken.append(best_action.name)
    decision = optimal_decision(belief, loss_type)
    return {"correct": decision == true_class, "total_cost": total_cost,
            "final_belief": belief, "final_risk": bayes_risk(belief, loss_type),
            "n_actions": len(actions_taken), "actions_taken": actions_taken}


# ============================================================
# 5. Borne théorique
# ============================================================

def chernoff_lower_bound(actions: List[PerceptionAction], epsilon: float) -> float:
    """Borne de Chernoff : C*(ε) ≥ log((1-ε)/ε) / η*"""
    eta_star = 0.0
    for a in actions:
        p = a.clarity
        d_kl = p * math.log(p / (1 - p + 1e-10) + 1e-10) + \
               (1 - p) * math.log((1 - p) / (p + 1e-10) + 1e-10)
        d_kl = max(0, d_kl)
        eta = d_kl / a.cost if a.cost > 0 else 0
        if eta > eta_star:
            eta_star = eta
    if eta_star < 1e-15:
        return float('inf')
    return math.log((1 - epsilon) / epsilon) / eta_star


# ============================================================
# 6. Mesures de submodularité
# ============================================================

def compute_delta_risk_curve(actions: List[PerceptionAction], loss_type: str,
                             n_bins: int = 100) -> Dict:
    beliefs = [0.01 + i * 0.98 / (n_bins - 1) for i in range(n_bins)]
    results = {}
    for action in actions:
        delta_risks = []
        for b in beliefs:
            current_risk = bayes_risk(b, loss_type)
            expected_risk = 0.0
            for obs in [0, 1]:
                p_obs = (action.clarity if obs == 1 else (1 - action.clarity)) * b + \
                        ((1 - action.clarity) if obs == 1 else action.clarity) * (1 - b)
                new_b = belief_update(b, obs, action.clarity)
                expected_risk += p_obs * bayes_risk(new_b, loss_type)
            delta_risks.append(current_risk - expected_risk)
        results[action.name] = {
            "beliefs": beliefs, "delta_risks": delta_risks,
            "cost": action.cost, "efficiency": [d / action.cost for d in delta_risks],
        }
    return results


# ============================================================
# 7. Expérience : prior variable
# ============================================================

def experiment_variable_prior(actions: List[PerceptionAction], loss_type: str,
                              n_trials: int = 5000) -> Dict:
    """Teste l'adaptativité sur des priors variables (Théorème 3)."""
    # Distribution de priors : mélange de faciles et difficiles
    priors = []
    for _ in range(n_trials):
        r = random.random()
        if r < 0.3:
            priors.append(random.uniform(0.1, 0.3))  # Facile (Y=0 probable)
        elif r < 0.6:
            priors.append(random.uniform(0.7, 0.9))  # Facile (Y=1 probable)
        else:
            priors.append(random.uniform(0.3, 0.7))  # Difficile

    # Comparer adaptatif vs fixe
    results_adaptive = {"costs": [], "corrects": []}
    results_fixed_320 = {"costs": [], "corrects": []}
    results_fixed_640 = {"costs": [], "corrects": []}
    results_fixed_1280 = {"costs": [], "corrects": []}

    for prior in priors:
        true_class = 1 if random.random() < prior else 0

        # Adaptatif (APC greedy)
        res = simulate_apc_greedy(actions, prior, true_class, loss_type, 0.9)
        results_adaptive["costs"].append(res["total_cost"])
        results_adaptive["corrects"].append(1 if res["correct"] else 0)

        # Fixe 320p
        res = simulate_always_cheapest(actions, prior, true_class, loss_type)
        results_fixed_320["costs"].append(res["total_cost"])
        results_fixed_320["corrects"].append(1 if res["correct"] else 0)

        # Fixe 640p
        a640 = [a for a in actions if a.name == "640p"][0]
        obs = 1 if (true_class == 1 and random.random() < a640.clarity) else \
              (0 if true_class == 0 and random.random() < a640.clarity else \
               (1 if true_class == 1 else 0))
        if true_class == 1:
            obs = 1 if random.random() < a640.clarity else 0
        else:
            obs = 0 if random.random() < a640.clarity else 1
        b = belief_update(prior, obs, a640.clarity)
        results_fixed_640["costs"].append(a640.cost)
        results_fixed_640["corrects"].append(1 if optimal_decision(b, loss_type) == true_class else 0)

        # Fixe 1280p
        res = simulate_always_best(actions, prior, true_class, loss_type)
        results_fixed_1280["costs"].append(res["total_cost"])
        results_fixed_1280["corrects"].append(1 if res["correct"] else 0)

    def summarize(data):
        costs = data["costs"]
        corrects = data["corrects"]
        mc = sum(costs) / len(costs)
        acc = sum(corrects) / len(corrects)
        return {"mean_cost": mc, "accuracy": acc, "epc": mc / max(acc, 1e-10)}

    return {
        "adaptive": summarize(results_adaptive),
        "fixed_320p": summarize(results_fixed_320),
        "fixed_640p": summarize(results_fixed_640),
        "fixed_1280p": summarize(results_fixed_1280),
    }


# ============================================================
# 8. Programme principal
# ============================================================

def main():
    random.seed(42)
    actions = default_actions()

    print("=" * 70)
    print("APC — Prototype Phase 2 : Loss functions, crops, bornes théoriques")
    print("=" * 70)

    # ── Expérience 1 : Comparaison des loss functions ──
    print("\n" + "─" * 70)
    print("EXPÉRIENCE 1 : Impact de la loss function")
    print("─" * 70)

    for loss_type in ["01", "squared", "log"]:
        print(f"\n▸ Loss = {loss_type}")
        dp_sol = solve_dp(actions, 0.5, loss_type, max_steps=8, n_bins=200)

        methods = {
            "DP": lambda tc: simulate_dp(dp_sol, actions, 0.5, tc, loss_type),
            "Always best": lambda tc: simulate_always_best(actions, 0.5, tc, loss_type),
            "Always cheap": lambda tc: simulate_always_cheapest(actions, 0.5, tc, loss_type),
            "APC ΔR/C": lambda tc: simulate_apc_greedy(actions, 0.5, tc, loss_type, 0.9),
            "Info-Gain": lambda tc: simulate_info_gain(actions, 0.5, tc, loss_type, 0.9),
        }

        print(f"  {'Méthode':<18} {'Coût':<8} {'Acc':<8} {'EPC':<8}")
        print(f"  {'─'*42}")
        for name, method in methods.items():
            costs, accs = [], []
            for _ in range(3000):
                tc = 1 if random.random() < 0.5 else 0
                res = method(tc)
                costs.append(res["total_cost"])
                accs.append(1 if res["correct"] else 0)
            mc = sum(costs) / len(costs)
            ac = sum(accs) / len(accs)
            epc = mc / max(ac, 1e-10)
            print(f"  {name:<18} {mc:<8.2f} {ac:<8.4f} {epc:<8.2f}")

        # Borne de Chernoff
        lb = chernoff_lower_bound(actions, 0.05)
        print(f"  Borne Chernoff (ε=0.05) : {lb:.2f}")

    # ── Expérience 2 : Submodularité par loss ──
    print("\n" + "─" * 70)
    print("EXPÉRIENCE 2 : Submodularité de ΔR par loss function")
    print("─" * 70)

    for loss_type in ["01", "squared", "log"]:
        print(f"\n▸ Loss = {loss_type}")
        delta = compute_delta_risk_curve(actions, loss_type)
        for aname, data in delta.items():
            dr = data["delta_risks"]
            n = len(dr)
            mid = n // 2
            # Vérifier concavité (submodularité)
            is_concave = all(dr[i] >= dr[i+1] - 1e-8 for i in range(mid, n-1))
            max_eff = max(data["efficiency"])
            b_max = data["beliefs"][data["efficiency"].index(max_eff)]
            print(f"  {aname:<12} ΔR/C max = {max_eff:.3f} (b={b_max:.2f}) "
                  f"{'✓ concave' if is_concave else '✗ non-concave'}")

    # ── Expérience 3 : Actions crop ──
    print("\n" + "─" * 70)
    print("EXPÉRIENCE 3 : Impact des actions crop")
    print("─" * 70)

    actions_no_crop = [a for a in actions if a.action_type == "resolution"]
    actions_with_crop = actions

    for label, act_list in [("Sans crop", actions_no_crop), ("Avec crop", actions_with_crop)]:
        print(f"\n▸ {label} ({len(act_list)} actions)")
        dp_sol = solve_dp(act_list, 0.5, "01", max_steps=8, n_bins=200)

        methods = {
            "DP": lambda tc: simulate_dp(dp_sol, act_list, 0.5, tc, "01"),
            "APC ΔR/C": lambda tc: simulate_apc_greedy(act_list, 0.5, tc, "01", 0.9),
        }

        for name, method in methods.items():
            costs, accs = [], []
            for _ in range(3000):
                tc = 1 if random.random() < 0.5 else 0
                res = method(tc)
                costs.append(res["total_cost"])
                accs.append(1 if res["correct"] else 0)
            mc = sum(costs) / len(costs)
            ac = sum(accs) / len(accs)
            epc = mc / max(ac, 1e-10)
            print(f"    {name:<18} Coût={mc:.2f}  Acc={ac:.4f}  EPC={epc:.2f}")

    # ── Expérience 4 : Prior variable ──
    print("\n" + "─" * 70)
    print("EXPÉRIENCE 4 : Prior variable (gain d'adaptativité)")
    print("─" * 70)

    for loss_type in ["01", "squared"]:
        print(f"\n▸ Loss = {loss_type}")
        res = experiment_variable_prior(actions, loss_type, n_trials=5000)
        print(f"  {'Méthode':<18} {'Coût':<8} {'Acc':<8} {'EPC':<8}")
        print(f"  {'─'*42}")
        for name, data in res.items():
            print(f"  {name:<18} {data['mean_cost']:<8.2f} "
                  f"{data['accuracy']:<8.4f} {data['epc']:<8.2f}")

        # Gap adaptatif vs fixe
        adapt = res["adaptive"]["epc"]
        fixed = min(res["fixed_320p"]["epc"], res["fixed_640p"]["epc"],
                    res["fixed_1280p"]["epc"])
        gap = fixed / adapt if adapt > 0 else 0
        print(f"  Gap adaptatif/meilleur fixe : {gap:.2f}x")

    print("\n" + "=" * 70)
    print("FIN — Phase 2 complétée")
    print("=" * 70)


if __name__ == "__main__":
    main()
