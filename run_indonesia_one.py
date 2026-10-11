"""
run_indonesia_one.py — evaluate ONE West Java policy, append to results_id.json.
Usage: python3 run_indonesia_one.py <PolicyName> <n_sims> <max_steps>
Names: Random, UrbanFirst, AccessFirst, Lookahead, MCTS
"""
import json, os, sys, time
import numpy as np
from indonesia_mdp import IndonesiaEnergyMDP, in_nothing
from indonesia_policies import INRandom, INUrbanFirst, INAccessFirst, INLookahead, INMCTS

NAME = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs("results", exist_ok=True)
OUT = os.path.join("results", "results_id.json")

mdp = IndonesiaEnergyMDP()
rng = np.random.default_rng(1234)
registry = {
    "Random":      (lambda: INRandom(rng),      "Random"),
    "UrbanFirst":  (lambda: INUrbanFirst(),     "Urban-First"),
    "AccessFirst": (lambda: INAccessFirst(),    "Access-First"),
    "Lookahead":   (lambda: INLookahead(depth=3),"MDP (Lookahead)"),
    "MCTS":        (lambda: INMCTS(rng, 200, 12, 12.0), "MDP (MCTS)"),
}
make, label = registry[NAME]
pol = make()


def metrics(mdp, s, total):
    regions = s.regions
    total_re = sum(r.re_installed for r in regions)
    total_supply = sum(r.re_installed + r.nre_installed for r in regions)
    re_pct = 100.0 * total_re / total_supply if total_supply > 0 else 0.0
    budget_used = mdp.initial_budget - s.b
    low_regions = 0
    low_pop = 0.0
    for r in regions:
        unmet = max(0.0, r.demand - (r.re_installed + r.nre_installed)) / r.demand
        if unmet > 0.1 and r.low_access == 1:
            low_regions += 1
            low_pop += r.population * unmet
    return {
        "total_reward": total,
        "re_pct": re_pct,
        "budget_used": budget_used,
        "low_access_regions": float(low_regions),
        "low_access_pop": low_pop,
        "private_leverage": s.private_leverage,
    }


t = time.time()
rows = []
for i in range(1, N + 1):
    r = np.random.default_rng(1234 + i)
    if hasattr(pol, "rng"):
        pol.rng = r
    s = mdp.initial_state(r)
    tot = 0.0
    for _ in range(STEPS):
        a = pol.action(mdp, s)
        tot += mdp.reward(s, a)
        s = mdp.transition(s, a)
        if mdp.is_terminal(s):
            break
    rows.append(metrics(mdp, s, tot))
agg = {}
for k in rows[0]:
    v = np.array([x[k] for x in rows], float)
    agg[k + "_mean"] = float(v.mean()); agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
dt = time.time() - t

data = {}
if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
data[label] = agg
with open(OUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"{label}: reward={agg['total_reward_mean']:.0f}±{agg['total_reward_std']:.0f}  "
      f"RE%={agg['re_pct_mean']:.1f}  lowpop={agg['low_access_pop_mean']/1e6:.2f}M  "
      f"leverage={agg['private_leverage_mean']:.0f}  ({dt:.1f}s)")
