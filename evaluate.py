"""
evaluate.py — Simulation and metric aggregation (mirrors src/utils.jl).

For each policy we run N seeded episodes of up to max_steps decisions, summing
undiscounted per-step rewards (matching the Julia HistoryRecorder convention),
then aggregate the same metrics reported in the paper.
"""

from __future__ import annotations
import numpy as np
from energy_mdp import EnergyMDP, State, do_nothing


def simulate_episode(mdp: EnergyMDP, policy, seed: int, max_steps: int):
    rng = np.random.default_rng(seed)
    # policies that carry their own rng use it for tie-breaking; give them this one
    if hasattr(policy, "rng"):
        policy.rng = rng
    s = mdp.initial_state(rng)
    total_reward = 0.0
    for _ in range(max_steps):
        a = policy.action(mdp, s)
        total_reward += mdp.reward(s, a)
        s = mdp.transition(s, a)
        if mdp.is_terminal(s):
            break
    return s, total_reward


def episode_metrics(mdp: EnergyMDP, final_state: State, total_reward: float):
    cities = final_state.cities
    total_re = sum(c.re_supply for c in cities)
    total_supply = sum(c.re_supply + c.nre_supply for c in cities)
    re_pct = 100.0 * total_re / total_supply if total_supply > 0 else 0.0

    budget_used = mdp.initial_budget - final_state.b

    low_cities = high_cities = 0
    low_pop = high_pop = 0.0
    for c in cities:
        unmet_ratio = max(0.0, c.demand - (c.re_supply + c.nre_supply)) / c.demand
        if unmet_ratio > 0.1:
            if c.income == 0:
                low_cities += 1
                low_pop += c.population * unmet_ratio
            else:
                high_cities += 1
                high_pop += c.population * unmet_ratio

    return {
        "total_reward": total_reward,
        "re_pct": re_pct,
        "budget_used": budget_used,
        "low_income_cities": float(low_cities),
        "high_income_cities": float(high_cities),
        "low_income_pop": low_pop,
        "high_income_pop": high_pop,
    }


def evaluate_policy(mdp: EnergyMDP, policy, n_sims=30, max_steps=12, base_seed=1234):
    rows = []
    for i in range(1, n_sims + 1):
        fs, tr = simulate_episode(mdp, policy, base_seed + i, max_steps)
        rows.append(episode_metrics(mdp, fs, tr))
    keys = rows[0].keys()
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        agg[k + "_mean"] = float(vals.mean())
        agg[k + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return agg


def format_table(results: dict) -> str:
    order = sorted(results.items(), key=lambda kv: kv[1]["total_reward_mean"], reverse=True)
    cols = [
        ("total_reward", "Avg Reward", 0, 1.0),
        ("re_pct",       "RE %",       1, 1.0),
        ("budget_used",  "Budget Used",0, 1.0),
        ("low_income_cities", "Low-Inc Cities", 1, 1.0),
        ("high_income_cities","High-Inc Cities",1, 1.0),
        ("low_income_pop",   "Low-Inc Pop(M)", 2, 1e-6),
        ("high_income_pop",  "High-Inc Pop(M)",2, 1e-6),
    ]
    hdr = f"{'Policy':<16}" + "".join(f" | {lbl:>15}" for _, lbl, _, _ in cols)
    lines = [hdr, "-" * len(hdr)]
    for name, m in order:
        row = f"{name:<16}"
        for base, _, prec, scale in cols:
            mu = m[base + "_mean"] * scale
            sd = m[base + "_std"] * scale
            row += f" | {mu:>7.{prec}f}±{sd:<6.{prec}f}"
        lines.append(row)
    return "\n".join(lines)
