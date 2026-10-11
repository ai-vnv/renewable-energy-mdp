"""
run_unified.py — Consolidated evaluation of ONE shared policy set on BOTH problems.

The same six problem-agnostic policies (solvers.py) are evaluated on the U.S.
single-planner MDP and the Indonesia incentive/quota MDP. This produces the two
per-problem tables in the paper (US = Table IV, Indonesia = Table V) from a
single shared roster, and also the cross-deployment robustness matrix.

Usage:
    python3 run_unified.py <Solver> <problem: US|ID> <n_sims> <max_steps> [time_budget_s]

Solver in: Random, MarketGreedy, EquityGreedy, ValueIteration, MCTS, EquityMCTS

Results are written to results/results_unified.json as {solver: {"US":..., "ID":...}}.
Episodes accumulate across calls (per solver+problem) so slow MCTS runs can be
built up within short time budgets. See run_cross.py for the transfer matrix.
"""
import os, sys, json, time
import numpy as np

from energy_mdp import EnergyMDP
from indonesia_mdp import IndonesiaEnergyMDP
from solvers import (RandomSolver, MarketGreedySolver, EquityGreedySolver,
                     ValueIterationSolver, MCTSSolver, EquityMCTSSolver)

# Arg parsing is deferred so this module can be imported (e.g. for `metrics`)
# without requiring command-line arguments.
SOLVER = sys.argv[1] if len(sys.argv) > 1 else "ValueIteration"
PROB = sys.argv[2] if len(sys.argv) > 2 else "US"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 30
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
BUDGET = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9

OUT = os.path.join("results", "results_unified.json")
CACHE = os.path.join("results", f".rows_{SOLVER}_{PROB}.json")

rng = np.random.default_rng(1234)


def make_solver():
    return {
        "Random":         lambda: RandomSolver(rng),
        "MarketGreedy":   lambda: MarketGreedySolver(),
        "EquityGreedy":   lambda: EquityGreedySolver(),
        "ValueIteration": lambda: ValueIterationSolver(depth=2),
        "MCTS":           lambda: MCTSSolver(rng, n_iterations=300, depth=15, c_uct=15.0),
        "EquityMCTS":     lambda: EquityMCTSSolver(rng, n_iterations=300, depth=20, c_uct=12.0),
    }[SOLVER]()


def make_mdp(prob):
    return EnergyMDP() if prob == "US" else IndonesiaEnergyMDP()


def metrics(mdp, prob, s, total):
    """Shared metric set for both problems (priority group = low-income cities /
    low-access regions)."""
    if prob == "US":
        units = s.cities
        re_installed = [c.re_supply for c in units]
        total_supply = [c.re_supply + c.nre_supply for c in units]
        priority = [c.income == 0 for c in units]
        demand = [c.demand for c in units]
        pop = [c.population for c in units]
        leverage = 0.0
    else:
        units = s.regions
        re_installed = [r.re_installed for r in units]
        total_supply = [r.re_installed + r.nre_installed for r in units]
        priority = [r.low_access == 1 for r in units]
        demand = [r.demand for r in units]
        pop = [r.population for r in units]
        leverage = s.private_leverage

    tot_re = sum(re_installed)
    tot_sup = sum(total_supply)
    re_pct = 100.0 * tot_re / tot_sup if tot_sup > 0 else 0.0
    budget_used = mdp.initial_budget - s.b

    prio_units = prio_pop = 0.0
    for i in range(len(units)):
        unmet = max(0.0, demand[i] - total_supply[i]) / demand[i]
        if unmet > 0.1 and priority[i]:
            prio_units += 1
            prio_pop += pop[i] * unmet

    return {
        "total_reward": total, "re_pct": re_pct, "budget_used": budget_used,
        "priority_units": prio_units, "priority_pop": prio_pop, "leverage": leverage,
    }


def run_episode(mdp, prob, solver, seed, steps):
    r = np.random.default_rng(seed)
    if hasattr(solver, "rng"):
        solver.rng = r
    s = mdp.initial_state(r)
    tot = 0.0
    for _ in range(steps):
        a = solver.action(mdp, s)
        tot += mdp.reward(s, a)
        s = mdp.transition(s, a)
        if mdp.is_terminal(s):
            break
    return metrics(mdp, prob, s, tot)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            rows = json.load(f)

    mdp = make_mdp(PROB)
    t0 = time.time()
    i = len(rows)
    while i < N and (time.time() - t0) < BUDGET:
        rows.append(run_episode(mdp, PROB, make_solver(), 1234 + i + 1, STEPS))
        i += 1

    with open(CACHE, "w") as f:
        json.dump(rows, f)

    if len(rows) >= N:
        agg = {}
        for k in rows[0]:
            v = np.array([x[k] for x in rows[:N]], float)
            agg[k + "_mean"] = float(v.mean())
            agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        data = {}
        if os.path.exists(OUT):
            with open(OUT) as f:
                data = json.load(f)
        data.setdefault(SOLVER, {})[PROB] = agg
        with open(OUT, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DONE] {SOLVER}/{PROB}: reward={agg['total_reward_mean']:.0f}±{agg['total_reward_std']:.0f} "
              f"RE%={agg['re_pct_mean']:.1f} prio_pop={agg['priority_pop_mean']/1e6:.2f}M "
              f"lev={agg['leverage_mean']:.0f}")
    else:
        print(f"[PROGRESS] {SOLVER}/{PROB}: {len(rows)}/{N} eps ({time.time()-t0:.0f}s)")
