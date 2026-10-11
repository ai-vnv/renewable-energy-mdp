"""
run_mcts_accum.py — accumulate MCTS episodes across calls (checkpointed).

Stores per-episode metric rows in a .rows.json cache keyed by policy label so
that slow MCTS runs can be built up to N episodes over several invocations, then
aggregated into results_us.json.

Usage: python3 run_mcts_accum.py <MCTS|EquityMCTS|MCTSRE> <target_N> <max_steps> <time_budget_s>
"""
import json, os, sys, time
import numpy as np
from energy_mdp import EnergyMDP, do_nothing
from policies import MCTSPolicy, EquityMCTSPolicy, MCTSREPolicy
from evaluate import episode_metrics

NAME = sys.argv[1]
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 30
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BUDGET = float(sys.argv[4]) if len(sys.argv) > 4 else 38.0

label = {"MCTS": "MCTS Base", "EquityMCTS": "Equity-MCTS", "MCTSRE": "MCTS RE"}[NAME]
CACHE = f".rows_{NAME}.json"

mdp = EnergyMDP()
mdp_re = EnergyMDP(w_action_cost=0.005, w_lowincome_served=160.0, w_pop_with_re=220.0)

rows = []
if os.path.exists(CACHE):
    with open(CACHE) as f:
        rows = json.load(f)
done = len(rows)

t0 = time.time()
i = done
while i < TARGET and (time.time() - t0) < BUDGET:
    seed = 1234 + i + 1
    r = np.random.default_rng(seed)
    if NAME == "MCTS":
        pol = MCTSPolicy(r, n_iterations=300, depth=15, c_uct=15.0)
        solve = mdp
    elif NAME == "EquityMCTS":
        pol = EquityMCTSPolicy(r, n_iterations=300, depth=20, c_uct=12.0)
        solve = mdp
    else:  # MCTSRE
        pol = MCTSREPolicy(r, n_iterations=300, depth=15, c_uct=10.0)
        solve = mdp_re
    s = mdp.initial_state(r)
    tot = 0.0
    for _ in range(STEPS):
        a = pol.action(solve, s)
        if a not in mdp.valid_actions(s):
            a = do_nothing()
        tot += mdp.reward(s, a)
        s = mdp.transition(s, a)
        if mdp.is_terminal(s):
            break
    rows.append(episode_metrics(mdp, s, tot))
    i += 1

with open(CACHE, "w") as f:
    json.dump(rows, f)

# if reached target, aggregate into results_us.json
if len(rows) >= TARGET:
    agg = {}
    for k in rows[0]:
        v = np.array([x[k] for x in rows[:TARGET]], float)
        agg[k + "_mean"] = float(v.mean()); agg[k + "_std"] = float(v.std(ddof=1))
    data = {}
    if os.path.exists("results_us.json"):
        with open("results_us.json") as f:
            data = json.load(f)
    data[label] = agg
    with open("results_us.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[DONE] {label}: {len(rows)}/{TARGET} eps  reward={agg['total_reward_mean']:.0f}±{agg['total_reward_std']:.0f}  "
          f"RE%={agg['re_pct_mean']:.1f}  lowpop={agg['low_income_pop_mean']/1e6:.2f}M")
else:
    print(f"[PROGRESS] {label}: {len(rows)}/{TARGET} eps ({time.time()-t0:.0f}s this call)")
