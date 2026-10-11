"""
solvers.py — Problem-agnostic MDP solvers.

The SAME solver code runs on both the U.S. single-planner MDP (energy_mdp.py)
and the Indonesia incentive/quota MDP (indonesia_mdp.py). A solver only relies on
the common MDP interface:

    mdp.valid_actions(s) -> list[action]
    mdp.transition(s, a) -> s'
    mdp.reward(s, a)     -> float
    mdp.is_terminal(s)   -> bool
    mdp.discount         -> float
    mdp.initial_state(rng) -> s
    mdp.noop()           -> the do-nothing action
    mdp.state_key(s)     -> hashable key for the tree cache
    mdp.equity_score(s, a) -> float   # >0 if a reduces the priority group's
                                      #   energy shortfall the most (used only by
                                      #   the informed rollout / equity heuristic)

Because the interface is identical, the policies below are defined once and
evaluated unchanged on both problems.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# 1. Random
# ---------------------------------------------------------------------------
class RandomSolver:
    name = "Random"
    def __init__(self, rng):
        self.rng = rng
    def action(self, mdp, s):
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]


# ---------------------------------------------------------------------------
# 2. Equity-first greedy heuristic (generic)
#    Picks the affordable action with the highest equity_score; falls back to
#    do-nothing. Plays the role that GreedyRE / Access-First played before, but
#    is now a single implementation shared by both problems.
# ---------------------------------------------------------------------------
class EquityGreedySolver:
    name = "Equity-Greedy"
    def action(self, mdp, s):
        best, best_val = mdp.noop(), 0.0
        for a in mdp.valid_actions(s):
            v = mdp.equity_score(s, a)
            if v > best_val:
                best_val, best = v, a
        return best


# ---------------------------------------------------------------------------
# 2b. Market-greedy heuristic (generic) — the naive "chase the easy/attractive
#     target" policy. Ignores equity and pursues cheap capacity or
#     market-attractive clusters first. This is the common failure mode
#     (Urban-First in the incentive setting): high output, poor equity.
# ---------------------------------------------------------------------------
class MarketGreedySolver:
    name = "Market-Greedy"
    def action(self, mdp, s):
        best, best_val = mdp.noop(), 0.0
        for a in mdp.valid_actions(s):
            v = mdp.market_score(s, a)
            if v > best_val:
                best_val, best = v, a
        return best


# ---------------------------------------------------------------------------
# 3. Receding-horizon Value Iteration (generic)
#    Depth-limited optimal value; greedy action w.r.t. it. Tractable analogue of
#    value iteration on a truncated horizon for large factored state spaces.
# ---------------------------------------------------------------------------
class ValueIterationSolver:
    name = "Value Iteration"
    def __init__(self, depth=2):
        self.depth = depth

    def _value(self, mdp, s, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, mdp.noop())
        best = -math.inf
        for a in mdp.valid_actions(s):
            v = mdp.reward(s, a) + mdp.discount * self._value(mdp, mdp.transition(s, a), depth - 1)
            if v > best:
                best = v
        return best

    def action(self, mdp, s):
        best_a, best_v = mdp.noop(), -math.inf
        for a in mdp.valid_actions(s):
            v = mdp.reward(s, a) + mdp.discount * self._value(mdp, mdp.transition(s, a), self.depth - 1)
            if v > best_v:
                best_v, best_a = v, a
        return best_a


# ---------------------------------------------------------------------------
# 4. MCTS / UCT (generic), with optional informed (equity) rollout
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("N", "children", "Q")
    def __init__(self):
        self.N = 0
        self.children = {}
        self.Q = {}


class MCTSSolver:
    name = "MCTS"
    def __init__(self, rng, n_iterations=300, depth=15, c_uct=15.0,
                 informed=False, bias="equity"):
        self.rng = rng
        self.n_iterations = n_iterations
        self.depth = depth
        self.c_uct = c_uct
        self.informed = informed
        self.bias = bias   # "equity" or "market" — the transferable inductive bias

    def _heuristic_action(self, mdp, s):
        """Action the informed rollout / tree prior takes under the configured bias."""
        score = mdp.market_score if self.bias == "market" else mdp.equity_score
        best, best_val = None, 0.0
        for a in mdp.valid_actions(s):
            v = score(s, a)
            if v > best_val:
                best_val, best = v, a
        return best

    def _rollout_action(self, mdp, s):
        if self.informed:
            score = mdp.market_score if self.bias == "market" else mdp.equity_score
            best, best_val = None, 0.0
            for a in mdp.valid_actions(s):
                v = score(s, a)
                if v > best_val:
                    best_val, best = v, a
            if best is not None:
                return best
        va = mdp.valid_actions(s)
        return va[self.rng.integers(len(va))]

    def _rollout(self, mdp, s, depth):
        total, disc, d = 0.0, 1.0, 0
        while d < depth and not mdp.is_terminal(s):
            a = self._rollout_action(mdp, s)
            total += disc * mdp.reward(s, a)
            s = mdp.transition(s, a)
            disc *= mdp.discount
            d += 1
        if not mdp.is_terminal(s):
            total += disc * mdp.reward(s, mdp.noop())
        return total

    def _simulate(self, mdp, s, table, key, depth):
        if depth == 0 or mdp.is_terminal(s):
            return mdp.reward(s, mdp.noop())
        if key not in table:
            node = table[key] = _Node()
            # Informed tree prior: give the equity-recommended action one virtual
            # visit initialized with a one-step lookahead value. This biases UCT
            # toward socially-beneficial investment whose payoff is delayed, and
            # is what makes the shared Equity-MCTS work uniformly on BOTH problems
            # (plain MCTS collapses to inaction on each).
            if self.informed:
                ha = self._heuristic_action(mdp, s)
                if ha is not None:
                    sp = mdp.transition(s, ha)
                    q0 = mdp.reward(s, ha) + mdp.discount * self._rollout(mdp, sp, depth - 1)
                    node.children[ha] = 1
                    node.Q[ha] = q0
                    node.N = 1
            return self._rollout(mdp, s, depth)
        node = table[key]
        best_a, best_ucb = None, -math.inf
        for a in mdp.valid_actions(s):
            n_sa = node.children.get(a, 0)
            q_sa = node.Q.get(a, 0.0)
            ucb = math.inf if n_sa == 0 else q_sa + self.c_uct * math.sqrt(math.log(node.N + 1) / n_sa)
            if ucb > best_ucb:
                best_ucb, best_a = ucb, a
        a = best_a
        sp = mdp.transition(s, a)
        r = mdp.reward(s, a)
        q = r + mdp.discount * self._simulate(mdp, sp, table, mdp.state_key(sp), depth - 1)
        node.N += 1
        n_sa = node.children.get(a, 0) + 1
        node.children[a] = n_sa
        node.Q[a] = node.Q.get(a, 0.0) + (q - node.Q.get(a, 0.0)) / n_sa
        return q

    def action(self, mdp, s):
        table = {}
        key = mdp.state_key(s)
        for _ in range(self.n_iterations):
            self._simulate(mdp, s, table, key, self.depth)
        node = table.get(key)
        if not node or not node.Q:
            va = mdp.valid_actions(s)
            return va[self.rng.integers(len(va))]
        return max(node.Q.items(), key=lambda kv: kv[1])[0]


class EquityMCTSSolver(MCTSSolver):
    """MCTS with an equity-biased (informed) rollout policy + tree prior."""
    name = "Equity-MCTS"
    def __init__(self, rng, n_iterations=300, depth=20, c_uct=12.0):
        super().__init__(rng, n_iterations, depth, c_uct, informed=True, bias="equity")


class MarketMCTSSolver(MCTSSolver):
    """MCTS whose informed bias targets the easy/market-attractive option instead
    of equity. Used in the cross-deployment study to test whether a policy's
    inductive bias transfers between institutional settings."""
    name = "Market-MCTS"
    def __init__(self, rng, n_iterations=300, depth=20, c_uct=12.0):
        super().__init__(rng, n_iterations, depth, c_uct, informed=True, bias="market")
