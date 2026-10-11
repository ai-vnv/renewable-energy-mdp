"""
run_sensitivity.py — One-at-a-time sensitivity of the policy comparison to the
reward weights (added for the camera-ready revision, Reviewer 2).

Each reward weight is scaled by 0.5 and 1.5 while the other weights stay at the
values used in the paper. This gives 6 settings for the U.S. case (3 weights)
and 10 settings for the West Java case (5 weights), plus the two baselines.
The same six policies, the same 30 seeds and the same 20-step horizon as
run_unified.py are used, so the baseline rows reproduce the two policy tables
of the paper exactly.

Usage:
    python3 run_sensitivity.py [n_processes]     # default 2; about 45 min on 2 cores

Output:
    results/results_sensitivity.json
        {"US": {"<weight>x<factor>": {"<Solver>": {"total_reward_mean": ..., ...}}}, "ID": {...}}
    Per-job episode rows are cached in results/.sens_*.json so the run can be resumed.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

_argv, sys.argv = sys.argv, [sys.argv[0]]      # run_unified reads sys.argv at import
import run_unified as ru                        # noqa: E402
sys.argv = _argv

from energy_mdp import EnergyMDP                # noqa: E402
from indonesia_mdp import IndonesiaEnergyMDP    # noqa: E402
from solvers import (RandomSolver, MarketGreedySolver, EquityGreedySolver,       # noqa: E402
                     ValueIterationSolver, MCTSSolver, EquityMCTSSolver)

WEIGHTS = {
    "US": ["w_action_cost", "w_lowincome_served", "w_pop_with_re"],
    "ID": ["w_cost", "w_access_coverage", "w_re_coverage", "w_leverage", "w_deadweight"],
}
FACTORS = (0.5, 1.5)
SOLVERS = ["Random", "MarketGreedy", "EquityGreedy", "ValueIteration", "MCTS", "EquityMCTS"]
N_SIMS, STEPS = 30, 20
OUT_DIR = "results"


def make_solver(name, rng):
    return {
        "Random":         lambda: RandomSolver(rng),
        "MarketGreedy":   lambda: MarketGreedySolver(),
        "EquityGreedy":   lambda: EquityGreedySolver(),
        "ValueIteration": lambda: ValueIterationSolver(depth=2),
        "MCTS":           lambda: MCTSSolver(rng, n_iterations=300, depth=15, c_uct=15.0),
        "EquityMCTS":     lambda: EquityMCTSSolver(rng, n_iterations=300, depth=20, c_uct=12.0),
    }[name]()


def setting_name(weight, factor):
    return "baseline" if weight is None else f"{weight}x{factor}"


def run_job(job):
    prob, weight, factor, solver = job
    cache = os.path.join(OUT_DIR, f".sens_{prob}_{setting_name(weight, factor)}_{solver}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return job, json.load(f)
    mdp = EnergyMDP() if prob == "US" else IndonesiaEnergyMDP()
    if weight is not None:
        setattr(mdp, weight, getattr(mdp, weight) * factor)
    rng = np.random.default_rng(1234)
    rows = [ru.run_episode(mdp, prob, make_solver(solver, rng), 1234 + i + 1, STEPS)
            for i in range(N_SIMS)]
    with open(cache, "w") as f:
        json.dump(rows, f)
    return job, rows


def aggregate(rows):
    agg = {}
    for k in rows[0]:
        v = np.array([x[k] for x in rows], float)
        agg[k + "_mean"] = float(v.mean())
        agg[k + "_std"] = float(v.std(ddof=1))
    return agg


def all_jobs():
    settings = [(p, None, 1.0) for p in WEIGHTS]
    settings += [(p, w, f) for p in WEIGHTS for w in WEIGHTS[p] for f in FACTORS]
    return [(p, w, f, s) for (p, w, f) in settings for s in SOLVERS]


if __name__ == "__main__":
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {p: {} for p in WEIGHTS}
    with Pool(n_proc) as pool:
        for (prob, weight, factor, solver), rows in pool.imap_unordered(run_job, all_jobs(), chunksize=1):
            results[prob].setdefault(setting_name(weight, factor), {})[solver] = aggregate(rows)
            print(f"[DONE] {prob} {setting_name(weight, factor)} {solver}", flush=True)
    with open(os.path.join(OUT_DIR, "results_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    for prob in results:
        print(f"\n{prob}: policies ranked by mean reward")
        for setting in sorted(results[prob]):
            r = results[prob][setting]
            order = sorted(r, key=lambda s: -r[s]["total_reward_mean"])
            print(f"  {setting:28s} " + " > ".join(order))
