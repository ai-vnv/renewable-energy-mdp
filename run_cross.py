"""
run_cross.py — Cross-deployment robustness (transfer) study.

Online policies carry a transferable inductive bias: the OBJECTIVE they plan
toward. We take the bias that is optimal on a *source* problem and deploy the
policy on a *target* problem, measuring the performance retained.

Two bias families, each realized as a greedy heuristic and an MCTS planner:
  * equity bias  — target the priority (low-income / low-access) group. This is
                   the bias that wins on the U.S. problem.
  * market bias  — target the cheap / market-attractive option. The naive
                   "chase attractive markets" default.

For each (bias, deploy-problem) cell we report reward and priority-group
population left underserved (M). The diagonal is the bias that matches the
problem's own best policy; the off-diagonal is the transferred bias.

Usage: python3 run_cross.py <family: greedy|mcts> <bias: equity|market> <deploy: US|ID> <n> <steps> [budget_s]
Writes results/results_cross.json.
"""
import os, sys, json, time
import numpy as np

from energy_mdp import EnergyMDP
from indonesia_mdp import IndonesiaEnergyMDP
from solvers import (EquityGreedySolver, MarketGreedySolver,
                     EquityMCTSSolver, MarketMCTSSolver)


def metrics(mdp, prob, s, total):
    """Shared metric set for both problems (priority = low-income / low-access)."""
    if prob == "US":
        units = s.cities
        total_supply = [c.re_supply + c.nre_supply for c in units]
        re_installed = [c.re_supply for c in units]
        priority = [c.income == 0 for c in units]
        demand = [c.demand for c in units]
        pop = [c.population for c in units]
        leverage = 0.0
    else:
        units = s.regions
        total_supply = [r.re_installed + r.nre_installed for r in units]
        re_installed = [r.re_installed for r in units]
        priority = [r.low_access == 1 for r in units]
        demand = [r.demand for r in units]
        pop = [r.population for r in units]
        leverage = s.private_leverage
    tot_re, tot_sup = sum(re_installed), sum(total_supply)
    re_pct = 100.0 * tot_re / tot_sup if tot_sup > 0 else 0.0
    prio_pop = 0.0
    for i in range(len(units)):
        unmet = max(0.0, demand[i] - total_supply[i]) / demand[i]
        if unmet > 0.1 and priority[i]:
            prio_pop += pop[i] * unmet
    return {"total_reward": total, "re_pct": re_pct,
            "priority_pop": prio_pop, "leverage": leverage}

FAMILY = sys.argv[1]      # greedy | mcts
BIAS = sys.argv[2]        # equity | market
DEPLOY = sys.argv[3]      # US | ID
N = int(sys.argv[4]) if len(sys.argv) > 4 else 30
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
BUDGET = float(sys.argv[6]) if len(sys.argv) > 6 else 1e9

os.makedirs("results", exist_ok=True)
OUT = os.path.join("results", "results_cross.json")
CACHE = os.path.join("results", f".xrows_{FAMILY}_{BIAS}_{DEPLOY}.json")
rng = np.random.default_rng(1234)


def make_solver():
    if FAMILY == "greedy":
        return EquityGreedySolver() if BIAS == "equity" else MarketGreedySolver()
    return (EquityMCTSSolver(rng, 300, 20, 12.0) if BIAS == "equity"
            else MarketMCTSSolver(rng, 300, 20, 12.0))


def make_mdp():
    return EnergyMDP() if DEPLOY == "US" else IndonesiaEnergyMDP()


rows = []
if os.path.exists(CACHE):
    with open(CACHE) as f:
        rows = json.load(f)

mdp = make_mdp()
t0 = time.time()
i = len(rows)
while i < N and (time.time() - t0) < BUDGET:
    r = np.random.default_rng(1234 + i + 1)
    solver = make_solver()
    if hasattr(solver, "rng"):
        solver.rng = r
    s = mdp.initial_state(r)
    tot = 0.0
    for _ in range(STEPS):
        a = solver.action(mdp, s)
        tot += mdp.reward(s, a)
        s = mdp.transition(s, a)
        if mdp.is_terminal(s):
            break
    rows.append(metrics(mdp, DEPLOY, s, tot))
    i += 1

with open(CACHE, "w") as f:
    json.dump(rows, f)

key = f"{FAMILY}"
subkey = f"{BIAS}->{DEPLOY}"
if len(rows) >= N:
    agg = {}
    for k in ("total_reward", "priority_pop", "re_pct", "leverage"):
        v = np.array([x[k] for x in rows[:N]], float)
        agg[k + "_mean"] = float(v.mean())
        agg[k + "_std"] = float(v.std(ddof=1))
    data = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            data = json.load(f)
    data.setdefault(key, {})[subkey] = agg
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[DONE] {FAMILY} {BIAS}->{DEPLOY}: reward={agg['total_reward_mean']:.0f}±{agg['total_reward_std']:.0f} "
          f"prio_pop={agg['priority_pop_mean']/1e6:.2f}M")
else:
    print(f"[PROGRESS] {FAMILY} {BIAS}->{DEPLOY}: {len(rows)}/{N} ({time.time()-t0:.0f}s)")
