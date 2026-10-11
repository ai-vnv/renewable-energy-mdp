"""make_figures.py — regenerate paper result figures from results/results_unified.json.

result_1.png            US average reward by policy
result_2.png            US per-metric grid
result_indonesia.png    West Java reward / access-equity / leverage by policy
All use the shared six-policy roster (solvers.py) so figures match Tables IV & V.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_R = "results" if os.path.exists(os.path.join("results", "results_unified.json")) else "."
D = json.load(open(os.path.join(_R, "results_unified.json")))

ORDER = ["ValueIteration", "EquityMCTS", "EquityGreedy", "MarketGreedy", "MCTS", "Random"]
LABEL = {"ValueIteration": "Value Iteration", "EquityMCTS": "Equity-MCTS",
         "EquityGreedy": "Equity-Greedy", "MarketGreedy": "Market-Greedy",
         "MCTS": "MCTS", "Random": "Random"}
names = [LABEL[k] for k in ORDER]
colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#7f7f7f", "#d62728"]


def bars(ax, m, s, title, ylabel):
    ax.bar(range(len(names)), m, yerr=s, color=colors, capsize=3, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color="black", lw=0.6)


def col(prob, key, scale=1.0):
    m = [D[k][prob][key + "_mean"] * scale for k in ORDER]
    s = [D[k][prob][key + "_std"] * scale for k in ORDER]
    return m, s


# ---- Figure 1: US average reward ----
fig, ax = plt.subplots(figsize=(7, 4))
m, s = col("US", "total_reward")
bars(ax, m, s, "US: Average Discounted Reward by Policy", "Reward")
plt.tight_layout(); plt.savefig("result_1.png", dpi=150); plt.close()

# ---- Figure 2: US metrics grid ----
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
specs = [
    ("total_reward", "Avg Reward (higher better)", "Reward", 1.0),
    ("re_pct", "RE % (higher better)", "%", 1.0),
    ("budget_used", "Budget Used ($M)", "$M", 1.0),
    ("priority_pop", "Low-Income Pop Unserved (lower better)", "Millions", 1e-6),
]
for ax, (key, title, ylab, scale) in zip(axes.flat, specs):
    m, s = col("US", key, scale)
    bars(ax, m, s, title, ylab)
plt.tight_layout(); plt.savefig("result_2.png", dpi=150); plt.close()

# ---- Figure 3: Indonesia ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
idspecs = [
    ("total_reward", "West Java: Avg Reward (higher better)", "Reward", 1.0),
    ("priority_pop", "Low-Access Pop Unserved (lower better)", "Millions", 1e-6),
    ("leverage", "Private Capital Crowded In (higher better)", "Index", 1.0),
]
for ax, (key, title, ylab, scale) in zip(axes, idspecs):
    m, s = col("ID", key, scale)
    bars(ax, m, s, title, ylab)
plt.tight_layout(); plt.savefig("result_indonesia.png", dpi=150); plt.close()

print("Saved result_1.png, result_2.png, result_indonesia.png")
