"""
run_one.py — Evaluate ONE policy and append to results_us.json (resumable).

Usage: python3 run_one.py <policy_name> <n_sims> <max_steps>
Policy names: Random, Expert, GreedyRE, ValueIteration, MCTS, EquityMCTS, MCTSRE
"""
import json, os, sys, time
import numpy as np
from energy_mdp import EnergyMDP, do_nothing
from policies import (RandomPolicy, ExpertPolicy, GreedyREPolicy,
                      MCTSPolicy, EquityMCTSPolicy, MCTSREPolicy, ValueIterationPolicy)
from evaluate import evaluate_policy, episode_metrics

NAME = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
os.makedirs("results", exist_ok=True)
OUT = os.path.join("results", "results_us.json")

mdp = EnergyMDP()
mdp_re = EnergyMDP(w_action_cost=0.005, w_lowincome_served=160.0, w_pop_with_re=220.0)
rng = np.random.default_rng(1234)

registry = {
    "Random":        lambda: RandomPolicy(rng),
    "Expert":        lambda: ExpertPolicy(),
    "GreedyRE":      lambda: GreedyREPolicy(),
    "ValueIteration":lambda: ValueIterationPolicy(depth=2),
    "MCTS":          lambda: MCTSPolicy(rng, n_iterations=300, depth=15, c_uct=15.0),
    "EquityMCTS":    lambda: EquityMCTSPolicy(rng, n_iterations=300, depth=20, c_uct=12.0),
}

label = {"MCTS": "MCTS Base", "EquityMCTS": "Equity-MCTS",
         "ValueIteration": "Value Iteration", "MCTSRE": "MCTS RE"}.get(NAME, NAME)

t = time.time()
if NAME == "MCTSRE":
    # solve under RE-emphasis weights, score under base reward
    pol = MCTSREPolicy(rng, n_iterations=300, depth=15, c_uct=10.0)
    rows = []
    for i in range(1, N + 1):
        r = np.random.default_rng(1234 + i)
        pol.rng = r
        s = mdp.initial_state(r)
        tot = 0.0
        for _ in range(STEPS):
            a = pol.action(mdp_re, s)
            if a not in mdp.valid_actions(s):
                a = do_nothing()
            tot += mdp.reward(s, a)
            s = mdp.transition(s, a)
            if mdp.is_terminal(s):
                break
        rows.append(episode_metrics(mdp, s, tot))
    agg = {}
    for k in rows[0]:
        v = np.array([x[k] for x in rows], float)
        agg[k + "_mean"] = float(v.mean()); agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    result = agg
else:
    result = evaluate_policy(mdp, registry[NAME](), N, STEPS)

dt = time.time() - t

data = {}
if os.path.exists(OUT):
    with open(OUT) as f:
        data = json.load(f)
data[label] = result
with open(OUT, "w") as f:
    json.dump(data, f, indent=2)

print(f"{label}: reward={result['total_reward_mean']:.0f}±{result['total_reward_std']:.0f}  "
      f"RE%={result['re_pct_mean']:.1f}  lowpop={result['low_income_pop_mean']/1e6:.2f}M  ({dt:.1f}s)")
