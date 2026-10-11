"""
export_csv.py — dump results JSON into tidy CSVs for pgfplots.

Outputs (into ../figs/data/ so they sit next to the figure .tex files):
    us_reward.csv        policy,reward,rstd
    us_metrics.csv       policy,reward,rstd,re,restd,budget,bstd,unserved,ustd
    indonesia.csv        policy,reward,rstd,unserved,ustd,leverage,levstd
    cross_greedy.csv     bias,us_reward,us_unserved,id_reward,id_unserved
    cross_mcts.csv       bias,us_reward,us_unserved,id_reward,id_unserved

Column conventions: rewards are absolute; *_unserved are in millions; RE in %.
Row order is fixed (best-to-worst by US reward) so bar charts read consistently.
"""
import os, json, csv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.abspath(os.path.join(HERE, "..", "figs", "data"))
os.makedirs(OUT, exist_ok=True)

U = json.load(open(os.path.join(RES, "results_unified.json")))
C = json.load(open(os.path.join(RES, "results_cross.json")))

# fixed order and short labels (pgfplots symbolic x ticks)
ORDER = ["ValueIteration", "EquityMCTS", "MCTS", "EquityGreedy", "MarketGreedy", "Random"]
LABEL = {"ValueIteration": "VI", "EquityMCTS": "Eq-MCTS", "MCTS": "MCTS",
         "EquityGreedy": "Eq-Greedy", "MarketGreedy": "Mk-Greedy", "Random": "Random"}


def m(key, prob, stat="mean", scale=1.0):
    return {k: U[k][prob][key + "_" + stat] * scale for k in ORDER}


def w(name, header, rows):
    p = os.path.join(OUT, name)
    with open(p, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print("wrote", p)


# --- US reward ---
w("us_reward.csv", ["policy", "reward", "rstd"],
  [[LABEL[k], round(U[k]["US"]["total_reward_mean"], 1), round(U[k]["US"]["total_reward_std"], 1)] for k in ORDER])

# --- US metrics grid ---
rows = []
for k in ORDER:
    d = U[k]["US"]
    rows.append([LABEL[k],
                 round(d["total_reward_mean"], 1), round(d["total_reward_std"], 1),
                 round(d["re_pct_mean"], 2), round(d["re_pct_std"], 2),
                 round(d["budget_used_mean"], 1), round(d["budget_used_std"], 1),
                 round(d["priority_pop_mean"] / 1e6, 3), round(d["priority_pop_std"] / 1e6, 3)])
w("us_metrics.csv", ["policy", "reward", "rstd", "re", "restd", "budget", "bstd", "unserved", "ustd"], rows)

# --- Indonesia ---
rows = []
for k in ORDER:
    d = U[k]["ID"]
    rows.append([LABEL[k],
                 round(d["total_reward_mean"], 1), round(d["total_reward_std"], 1),
                 round(d["priority_pop_mean"] / 1e6, 3), round(d["priority_pop_std"] / 1e6, 3),
                 round(d["leverage_mean"], 1), round(d["leverage_std"], 1)])
w("indonesia.csv", ["policy", "reward", "rstd", "unserved", "ustd", "leverage", "levstd"], rows)

# --- Combined US | West Java (comparable metrics, each keeps its own scale) ---
rows = []
for k in ORDER:
    u = U[k]["US"]; i = U[k]["ID"]
    rows.append([LABEL[k],
                 round(u["total_reward_mean"], 1), round(u["total_reward_std"], 1),
                 round(u["re_pct_mean"], 2), round(u["re_pct_std"], 2),
                 round(u["priority_pop_mean"] / 1e6, 3), round(u["priority_pop_std"] / 1e6, 3),
                 round(i["total_reward_mean"], 1), round(i["total_reward_std"], 1),
                 round(i["re_pct_mean"], 2), round(i["re_pct_std"], 2),
                 round(i["priority_pop_mean"] / 1e6, 3), round(i["priority_pop_std"] / 1e6, 3)])
w("combined.csv",
  ["policy",
   "us_reward", "us_rstd", "us_re", "us_restd", "us_unserved", "us_ustd",
   "id_reward", "id_rstd", "id_re", "id_restd", "id_unserved", "id_ustd"],
  rows)

# --- Cross-deployment (one file per family) ---
for fam in ("greedy", "mcts"):
    rows = []
    for bias in ("equity", "market"):
        us = C[fam][f"{bias}->US"]; idn = C[fam][f"{bias}->ID"]
        rows.append([bias.capitalize(),
                     round(us["total_reward_mean"], 1), round(us["priority_pop_mean"] / 1e6, 3),
                     round(idn["total_reward_mean"], 1), round(idn["priority_pop_mean"] / 1e6, 3)])
    w(f"cross_{fam}.csv", ["bias", "us_reward", "us_unserved", "id_reward", "id_unserved"], rows)

print("done")
