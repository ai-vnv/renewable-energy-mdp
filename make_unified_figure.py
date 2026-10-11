"""make_unified_figure.py — combined US-vs-Indonesia figure for the shared solver set."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_R = "results"
d = json.load(open(os.path.join(_R, "results_unified.json")))

ORDER = ["ValueIteration", "EquityMCTS", "EquityGreedy", "MCTS", "Random"]
LABEL = {"ValueIteration": "Value Iteration", "EquityMCTS": "Equity-MCTS",
         "EquityGreedy": "Equity-Greedy", "MCTS": "MCTS", "Random": "Random"}
names = [LABEL[k] for k in ORDER]
x = np.arange(len(names))
w = 0.38

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# Panel 1: reward, grouped US vs ID
us_r = [d[k]["US"]["total_reward_mean"] for k in ORDER]
us_e = [d[k]["US"]["total_reward_std"] for k in ORDER]
id_r = [d[k]["ID"]["total_reward_mean"] for k in ORDER]
id_e = [d[k]["ID"]["total_reward_std"] for k in ORDER]
ax = axes[0]
ax.bar(x - w/2, us_r, w, yerr=us_e, capsize=3, label="US (build)", color="#1f77b4", edgecolor="black", lw=0.4)
ax.bar(x + w/2, id_r, w, yerr=id_e, capsize=3, label="Indonesia (incentivize)", color="#ff7f0e", edgecolor="black", lw=0.4)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Average reward"); ax.set_title("Same solvers, both problems: average reward")
ax.legend(fontsize=8)

# Panel 2: priority-group population left underserved (equity), grouped
us_p = [d[k]["US"]["priority_pop_mean"] / 1e6 for k in ORDER]
id_p = [d[k]["ID"]["priority_pop_mean"] / 1e6 for k in ORDER]
ax = axes[1]
ax.bar(x - w/2, us_p, w, label="US: low-income", color="#1f77b4", edgecolor="black", lw=0.4)
ax.bar(x + w/2, id_p, w, label="Indonesia: low-access", color="#ff7f0e", edgecolor="black", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Priority population underserved (M)")
ax.set_title("Equity: priority population left underserved (lower better)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("result_unified.png", dpi=150)
plt.close()
print("Saved result_unified.png")
