"""indonesia_policies.py — policies for the West Java incentive/quota MDP."""
from __future__ import annotations
import math
import numpy as np
from indonesia_mdp import IndonesiaEnergyMDP, INState, INAction, in_nothing


class INRandom:
    name = "Random"
    def __init__(self, rng): self.rng = rng
    def action(self, mdp, s):
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]


class INUrbanFirst:
    """Naive policy: subsidize the most market-attractive (urban) clusters first.
    Mirrors the common failure mode — incentives flow where the market already
    invests, producing high deadweight and little additionality."""
    name = "Urban-First"
    def action(self, mdp, s):
        best, best_val = in_nothing(), -math.inf
        for a in mdp.valid_actions(s):
            if a.kind != "incentive":
                continue
            r = s.regions[a.region_idx]
            if r.market_attractiveness > best_val:
                best_val, best = r.market_attractiveness, a
        return best


class INAccessFirst:
    """Equity heuristic: prioritize low-access regions; use off-grid for rural,
    incentives for on-grid low-access clusters."""
    name = "Access-First"
    def action(self, mdp, s):
        # rural low-access with off-grid available -> off-grid
        cand = [a for a in mdp.valid_actions(s)
                if a.kind == "offgrid" and s.regions[a.region_idx].low_access == 1]
        if cand:
            cand.sort(key=lambda a: -max(0.0, s.regions[a.region_idx].demand
                       - (s.regions[a.region_idx].re_installed + s.regions[a.region_idx].nre_installed)))
            return cand[0]
        # else incentivize low-access on-grid regions with largest shortfall
        cand = [a for a in mdp.valid_actions(s)
                if a.kind == "incentive" and s.regions[a.region_idx].low_access == 1]
        if cand:
            cand.sort(key=lambda a: -max(0.0, s.regions[a.region_idx].demand
                       - (s.regions[a.region_idx].re_installed + s.regions[a.region_idx].nre_installed)))
            return cand[0]
        # else release quota to a low-access region for future private RE
        cand = [a for a in mdp.valid_actions(s)
                if a.kind == "quota" and s.regions[a.region_idx].low_access == 1]
        if cand:
            return cand[0]
        return in_nothing()


# ---------------------------------------------------------------------------
# Receding-horizon lookahead ("value iteration" analogue) for the incentive MDP
# ---------------------------------------------------------------------------
class INLookahead:
    name = "MDP (Lookahead)"
    def __init__(self, depth=3):
        self.depth = depth

    def _value(self, mdp, s, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, in_nothing())
        best = -math.inf
        for a in mdp.valid_actions(s):
            v = mdp.reward(s, a) + mdp.discount * self._value(mdp, mdp.transition(s, a), depth - 1)
            if v > best:
                best = v
        return best

    def action(self, mdp, s):
        best_a, best_v = in_nothing(), -math.inf
        for a in mdp.valid_actions(s):
            v = mdp.reward(s, a) + mdp.discount * self._value(mdp, mdp.transition(s, a), self.depth - 1)
            if v > best_v:
                best_v, best_a = v, a
        return best_a


# ---------------------------------------------------------------------------
# MCTS (UCT) for the incentive MDP
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("N", "children", "Q")
    def __init__(self):
        self.N = 0; self.children = {}; self.Q = {}


def _key(s: INState):
    return (round(s.b, 1),
            tuple((round(r.re_installed, 1), round(r.pln_quota, 1)) for r in s.regions))


class INMCTS:
    name = "MDP (MCTS)"
    def __init__(self, rng, n_iterations=400, depth=20, c_uct=12.0, informed=True):
        self.rng = rng; self.n_iterations = n_iterations
        self.depth = depth; self.c_uct = c_uct
        self.informed = informed  # informed rollout prevents collapse to inaction

    def _rollout_action(self, mdp, s):
        if self.informed:
            # target low-access regions with largest shortfall (off-grid for
            # rural, incentives for on-grid), matching the Access-First heuristic
            best, best_val = None, 0.0
            for a in mdp.valid_actions(s):
                if a.kind == "nothing":
                    continue
                r = s.regions[a.region_idx]
                if r.low_access != 1:
                    continue
                short = max(0.0, r.demand - (r.re_installed + r.nre_installed))
                pref = 1.5 if a.kind == "offgrid" and r.rural == 1 else 1.0
                val = short * pref
                if val > best_val:
                    best_val, best = val, a
            if best is not None:
                return best
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]

    def _rollout(self, mdp, s, depth):
        total, disc, d = 0.0, 1.0, 0
        while d < depth and not mdp.is_terminal(s):
            a = self._rollout_action(mdp, s)
            total += disc * mdp.reward(s, a)
            s = mdp.transition(s, a); disc *= mdp.discount; d += 1
        if not mdp.is_terminal(s):
            total += disc * mdp.reward(s, in_nothing())
        return total

    def _simulate(self, mdp, s, table, k, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, in_nothing())
        if k not in table:
            table[k] = _Node()
            return self._rollout(mdp, s, depth)
        node = table[k]
        best_a, best_ucb = None, -math.inf
        for a in mdp.valid_actions(s):
            n_sa = node.children.get(a, 0); q_sa = node.Q.get(a, 0.0)
            ucb = math.inf if n_sa == 0 else q_sa + self.c_uct * math.sqrt(math.log(node.N + 1) / n_sa)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        a = best_a
        r = mdp.reward(s, a); sp = mdp.transition(s, a)
        q = r + mdp.discount * self._simulate(mdp, sp, table, _key(sp), depth - 1)
        node.N += 1
        n_sa = node.children.get(a, 0) + 1
        node.children[a] = n_sa
        node.Q[a] = node.Q.get(a, 0.0) + (q - node.Q.get(a, 0.0)) / n_sa
        return q

    def action(self, mdp, s):
        table = {}; k = _key(s)
        for _ in range(self.n_iterations):
            self._simulate(mdp, s, table, k, self.depth)
        node = table.get(k)
        if not node or not node.Q:
            va = mdp.valid_actions(s); return va[self.rng.integers(len(va))]
        return max(node.Q.items(), key=lambda kv: kv[1])[0]
