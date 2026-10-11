"""
run_all.py — reproduce every result in the paper end to end.

    python3 run_all.py            # 30 reps, 20 steps (paper settings)
    python3 run_all.py 10 15      # quick smoke run

Produces results/results_us.json, results/results_id.json and (if matplotlib is
installed) the result_*.png figures. Fast policies run directly; the Monte Carlo
Tree Search policies are the slowest — for very tight time budgets use the
checkpointed run_mcts_accum.py / run_indonesia_one.py drivers instead.
"""
import os, sys, json, subprocess

N = sys.argv[1] if len(sys.argv) > 1 else "30"
STEPS = sys.argv[2] if len(sys.argv) > 2 else "20"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)


def run(script, *args):
    print(f"\n>>> {script} {' '.join(args)}")
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True, cwd=HERE)


# --- U.S. single-planner model -------------------------------------------------
for pol in ["Random", "Expert", "GreedyRE", "ValueIteration", "MCTS", "EquityMCTS", "MCTSRE"]:
    run("run_one.py", pol, N, STEPS)

# --- Indonesia (West Java) incentive/quota model ------------------------------
for pol in ["Random", "UrbanFirst", "AccessFirst", "Lookahead", "MCTS"]:
    run("run_indonesia_one.py", pol, N, STEPS)

# --- Figures ------------------------------------------------------------------
try:
    run("make_figures.py")
except Exception as e:  # matplotlib optional
    print("Figure generation skipped:", e)

print("\nAll experiments complete. See results/ and result_*.png")
