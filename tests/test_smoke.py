"""Smoke tests — every solver must run on every MDP through the shared interface.

Fast by design (small MCTS budgets, short horizons) so CI finishes in seconds:

    python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_mdp import EnergyMDP
from indonesia_mdp import IndonesiaEnergyMDP
from solvers import (RandomSolver, MarketGreedySolver, EquityGreedySolver,
                     ValueIterationSolver, MCTSSolver, EquityMCTSSolver,
                     MarketMCTSSolver)

STEPS = 4


def mdps():
    return [("US", EnergyMDP()), ("ID", IndonesiaEnergyMDP())]


def solvers(rng):
    return [
        ("Random", RandomSolver(rng)),
        ("MarketGreedy", MarketGreedySolver()),
        ("EquityGreedy", EquityGreedySolver()),
        ("ValueIteration", ValueIterationSolver(depth=1)),
        ("MCTS", MCTSSolver(rng, n_iterations=12, depth=3, c_uct=15.0)),
        ("EquityMCTS", EquityMCTSSolver(rng, n_iterations=12, depth=3, c_uct=12.0)),
        ("MarketMCTS", MarketMCTSSolver(rng, n_iterations=12, depth=3, c_uct=12.0)),
    ]


class TestSharedInterface(unittest.TestCase):
    """Both MDPs must expose the interface solvers.py programs against."""

    def test_interface_is_complete(self):
        for name, mdp in mdps():
            with self.subTest(problem=name):
                for attr in ("valid_actions", "transition", "reward", "is_terminal",
                             "initial_state", "noop", "state_key",
                             "equity_score", "market_score"):
                    self.assertTrue(hasattr(mdp, attr), f"{name} missing {attr}")
                self.assertGreater(mdp.discount, 0.0)
                self.assertLessEqual(mdp.discount, 1.0)

    def test_initial_state_is_non_terminal_with_actions(self):
        for name, mdp in mdps():
            with self.subTest(problem=name):
                s = mdp.initial_state(np.random.default_rng(0))
                self.assertFalse(mdp.is_terminal(s))
                self.assertGreater(len(mdp.valid_actions(s)), 0)
                self.assertIn(mdp.noop(), mdp.valid_actions(s))

    def test_initial_state_is_seed_deterministic(self):
        for name, mdp in mdps():
            with self.subTest(problem=name):
                a = mdp.state_key(mdp.initial_state(np.random.default_rng(7)))
                b = mdp.state_key(mdp.initial_state(np.random.default_rng(7)))
                self.assertEqual(a, b)


class TestSolverRollouts(unittest.TestCase):
    """Every solver runs on every problem and returns a finite return."""

    def test_rollout(self):
        for prob, mdp in mdps():
            for name, solver in solvers(np.random.default_rng(1234)):
                with self.subTest(problem=prob, solver=name):
                    s = mdp.initial_state(np.random.default_rng(1234))
                    total = 0.0
                    for _ in range(STEPS):
                        a = solver.action(mdp, s)
                        self.assertIn(a, mdp.valid_actions(s))
                        total += mdp.reward(s, a)
                        s = mdp.transition(s, a)
                        if mdp.is_terminal(s):
                            break
                    self.assertTrue(np.isfinite(total))
                    self.assertGreaterEqual(s.b, 0.0, "budget went negative")

    def test_noop_is_free_and_budget_preserving(self):
        for prob, mdp in mdps():
            with self.subTest(problem=prob):
                s = mdp.initial_state(np.random.default_rng(3))
                sp = mdp.transition(s, mdp.noop())
                self.assertLessEqual(sp.b, s.b + 1e-9)

    def test_seeded_rollouts_reproduce(self):
        """Same seed, same policy -> identical return (paper reproducibility)."""
        def run(prob, mdp):
            solver = MCTSSolver(np.random.default_rng(1234), n_iterations=12,
                                depth=3, c_uct=15.0)
            s = mdp.initial_state(np.random.default_rng(1234))
            total = 0.0
            for _ in range(STEPS):
                a = solver.action(mdp, s)
                total += mdp.reward(s, a)
                s = mdp.transition(s, a)
                if mdp.is_terminal(s):
                    break
            return total

        for prob, _ in mdps():
            with self.subTest(problem=prob):
                mdp_a = EnergyMDP() if prob == "US" else IndonesiaEnergyMDP()
                mdp_b = EnergyMDP() if prob == "US" else IndonesiaEnergyMDP()
                self.assertAlmostEqual(run(prob, mdp_a), run(prob, mdp_b), places=9)


class TestCachedResults(unittest.TestCase):
    """The committed results JSON must stay loadable and cover the paper tables."""

    def test_unified_results_cover_every_solver_and_problem(self):
        import json
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(here, "results", "results_unified.json")
        with open(path) as f:
            data = json.load(f)
        for solver in ("Random", "MarketGreedy", "EquityGreedy",
                       "ValueIteration", "MCTS", "EquityMCTS"):
            self.assertIn(solver, data)
            for prob in ("US", "ID"):
                self.assertIn("total_reward_mean", data[solver][prob])


if __name__ == "__main__":
    unittest.main()
