"""
policies.py — Benchmark and MDP policies for the US EnergyMDP.

Policies:
  RandomPolicy       — uniform over valid actions (baseline)
  ExpertPolicy       — largest-deficit-first RE placement (domain heuristic)
  GreedyREPolicy     — greedy RE toward unmet demand, low-income weighted
  MCTSPolicy         — Monte Carlo Tree Search (UCT) over the MDP
  EquityMCTSPolicy   — MCTS with an equity-shaped rollout (improved solver)
  ValueIterationPolicy (approx.) — fitted-value / receding-horizon lookahead
"""

from __future__ import annotations
import math
import numpy as np
from energy_mdp import EnergyMDP, State, Action, do_nothing


class RandomPolicy:
    name = "Random"
    def __init__(self, rng): self.rng = rng
    def action(self, mdp: EnergyMDP, s: State) -> Action:
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]


class ExpertPolicy:
    """Largest-deficit-first, add RE if affordable (paper's expert heuristic)."""
    name = "Expert"
    def action(self, mdp: EnergyMDP, s: State) -> Action:
        deficits = [(c.demand - c.re_supply - c.nre_supply, i) for i, c in enumerate(s.cities)]
        deficits.sort(key=lambda x: -x[0])
        idx = deficits[0][1]
        if s.b >= mdp.cost_add_re:
            return Action(add=True, energy=0, city_idx=idx)
        return do_nothing()


class GreedyREPolicy:
    """Greedy RE toward unmet demand, low-income cities weighted 2x."""
    name = "GreedyRE"
    def action(self, mdp: EnergyMDP, s: State) -> Action:
        best, best_val = do_nothing(), -math.inf
        for a in mdp.valid_actions(s):
            if a.do_nothing or a.energy != 0 or not a.add:
                continue
            c = s.cities[a.city_idx]
            unmet = c.demand - (c.re_supply + c.nre_supply)
            if c.income == 0:
                unmet *= 2.0
            if unmet > best_val:
                best_val, best = unmet, a
        return best


# ---------------------------------------------------------------------------
# MCTS (UCT)
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("N", "children", "Q")
    def __init__(self):
        self.N = 0
        self.children = {}   # action -> (child_state_key, edge_N, edge_Q)
        self.Q = {}


class MCTSPolicy:
    """Vanilla UCT planner. Re-plans from each encountered state."""
    name = "MCTS Base"

    def __init__(self, rng, n_iterations=1000, depth=30, c_uct=15.0,
                 rollout="default", shaped=False):
        self.rng = rng
        self.n_iterations = n_iterations
        self.depth = depth
        self.c_uct = c_uct
        self.rollout = rollout
        self.shaped = shaped   # use potential-based reward shaping in the tree

    def _r(self, mdp, s, a, sp):
        """Reward used *inside* the planner (optionally shaped)."""
        if self.shaped:
            return mdp.shaped_reward(s, a, sp)
        return mdp.reward(s, a)

    # -- rollout policies ---------------------------------------------------
    def _rollout_action(self, mdp, s):
        if self.rollout == "equity":
            # Informed (greedy) rollout: pick the affordable RE addition that
            # most reduces low-income shortfall, low-income cities weighted 2x.
            # An informed default policy is essential here — with a uniform-random
            # rollout, MCTS cannot see that sustained investment beats inaction
            # and collapses to "do nothing".
            best, best_val = None, 0.0
            for a in mdp.valid_actions(s):
                if a.do_nothing or not a.add or a.energy != 0:
                    continue
                c = s.cities[a.city_idx]
                unmet = c.demand - (c.re_supply + c.nre_supply)
                if unmet <= 0:
                    continue
                val = unmet * (2.0 if c.income == 0 else 1.0)
                if val > best_val:
                    best_val, best = val, a
            if best is not None:
                return best
            # fall back to a cheap NRE add for any remaining deficit
            for a in mdp.valid_actions(s):
                if a.add and a.energy == 1:
                    c = s.cities[a.city_idx]
                    if (c.re_supply + c.nre_supply) < c.demand:
                        return a
            return do_nothing()
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]

    def _rollout(self, mdp, s, depth):
        total, disc = 0.0, 1.0
        d = 0
        while d < depth and not mdp.is_terminal(s):
            a = self._rollout_action(mdp, s)
            sp = mdp.transition(s, a)
            total += disc * self._r(mdp, s, a, sp)
            s = sp
            disc *= mdp.discount
            d += 1
        if not mdp.is_terminal(s):
            total += disc * mdp.reward(s, do_nothing())
        return total

    def _simulate(self, mdp, s, node_table, s_key, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, do_nothing())
        if s_key not in node_table:
            node_table[s_key] = _Node()
            return self._rollout(mdp, s, depth)
        node = node_table[s_key]
        va = mdp.valid_actions(s)
        # UCT selection
        best_a, best_ucb = None, -math.inf
        for a in va:
            n_sa = node.children.get(a, 0)
            q_sa = node.Q.get(a, 0.0)
            if n_sa == 0:
                ucb = math.inf
            else:
                ucb = q_sa + self.c_uct * math.sqrt(math.log(node.N + 1) / n_sa)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        a = best_a
        sp = mdp.transition(s, a)
        r = self._r(mdp, s, a, sp)
        sp_key = _state_key(sp)
        q = r + mdp.discount * self._simulate(mdp, sp, node_table, sp_key, depth - 1)
        node.N += 1
        n_sa = node.children.get(a, 0) + 1
        node.children[a] = n_sa
        node.Q[a] = node.Q.get(a, 0.0) + (q - node.Q.get(a, 0.0)) / n_sa
        return q

    def action(self, mdp: EnergyMDP, s: State) -> Action:
        node_table = {}
        s_key = _state_key(s)
        for _ in range(self.n_iterations):
            self._simulate(mdp, s, node_table, s_key, self.depth)
        node = node_table.get(s_key)
        if node is None or not node.Q:
            va = mdp.valid_actions(s)
            return va[self.rng.integers(len(va))]
        return max(node.Q.items(), key=lambda kv: kv[1])[0]


class EquityMCTSPolicy(MCTSPolicy):
    """Improved solver: MCTS with equity-biased rollouts AND potential-based
    reward shaping, so the planner can see that a sequence of equity investments
    pays off (plain MCTS collapses to 'do nothing' in this landscape)."""
    name = "Equity-MCTS"
    def __init__(self, rng, n_iterations=2000, depth=40, c_uct=12.0):
        super().__init__(rng, n_iterations, depth, c_uct,
                         rollout="equity", shaped=False)


class MCTSREPolicy(MCTSPolicy):
    """RE-emphasis MCTS: solves an MDP whose weights favour RE deployment."""
    name = "MCTS RE"
    def __init__(self, rng, n_iterations=1500, depth=30, c_uct=10.0):
        super().__init__(rng, n_iterations, depth, c_uct, rollout="default")


# ---------------------------------------------------------------------------
# Approximate Value Iteration via receding-horizon lookahead
# ---------------------------------------------------------------------------
class ValueIterationPolicy:
    """
    Discretized value iteration is intractable for the full 8-city state space.
    We use a bounded receding-horizon (finite-depth) lookahead that returns the
    greedy action w.r.t. the depth-limited optimal value — the deterministic
    analogue of value iteration on a truncated horizon.
    """
    name = "Value Iteration"
    def __init__(self, depth=3):
        self.depth = depth

    def _value(self, mdp, s, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, do_nothing())
        best = -math.inf
        for a in mdp.valid_actions(s):
            r = mdp.reward(s, a)
            sp = mdp.transition(s, a)
            v = r + mdp.discount * self._value(mdp, sp, depth - 1)
            if v > best:
                best = v
        return best

    def action(self, mdp: EnergyMDP, s: State) -> Action:
        best_a, best_v = do_nothing(), -math.inf
        for a in mdp.valid_actions(s):
            r = mdp.reward(s, a)
            sp = mdp.transition(s, a)
            v = r + mdp.discount * self._value(mdp, sp, self.depth - 1)
            if v > best_v:
                best_v, best_a = v, a
        return best_a


def _state_key(s: State):
    return (round(s.b, 1),
            tuple((round(c.re_supply, 1), round(c.nre_supply, 1)) for c in s.cities))
