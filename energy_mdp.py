"""
energy_mdp.py — Faithful Python reimplementation of the EnergyProjectMDP (US case).

Mirrors the dynamics in the Julia package (src/mdp.jl, src/functions.jl) so the
paper's numerical experiments can be reproduced without a Julia toolchain.

State  s = (b, [ (demand, re, nre, pop, income) per city ])
Action a = do-nothing, or (add/remove) x (RE/NRE) x city_index
"""

from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class City:
    name: str
    demand: float
    re_supply: float
    nre_supply: float
    population: float
    income: int          # 1 = high/medium income, 0 = low income


@dataclass(frozen=True)
class State:
    b: float
    cities: Tuple[City, ...]
    total_demand: float


@dataclass(frozen=True)
class Action:
    add: Optional[bool] = None
    energy: Optional[int] = None  # 0 RE, 1 NRE
    city_idx: Optional[int] = None
    do_nothing: bool = False

    def __repr__(self):
        if self.do_nothing:
            return "do_nothing"
        e = "RE" if self.energy == 0 else "NRE"
        v = "add" if self.add else "rem"
        return f"{v}_{e}@{self.city_idx}"


def do_nothing() -> Action:
    return Action(do_nothing=True)


# Canonical U.S. cities (Table II of the paper — synthetic but realistic).
# Low-income classification: >= 25% low-income population -> income = 0.
# Demand, RE and NRE supply are expressed in consistent capacity units (GW-scale)
# so that the 10-GW capacity increment and the $3,000M budget can meaningfully
# serve a city over the planning horizon. City demands preserve the relative
# ordering of the eight U.S. metros in the paper.
US_CITIES = [
    City("Atlanta",        58.0, 14.0, 19.0,   498_000, 0),
    City("New York City", 105.0, 26.0, 24.0, 8_468_000, 0),
    City("Houston",        72.0, 13.0, 17.0, 2_305_000, 1),
    City("Phoenix",        60.0, 10.0, 18.0, 1_608_000, 0),
    City("Denver",         42.0, 11.0, 10.0,   727_000, 1),
    City("Memphis",        38.0,  8.0, 15.0,   628_000, 0),
    City("Seattle",        44.0, 12.0,  8.0,   753_000, 1),
    City("San Antonio",    63.0,  9.0, 16.0, 1_452_000, 0),
]


@dataclass
class EnergyMDP:
    """US renewable-energy planning MDP (single-planner, government-builds model)."""
    cities: List[City] = field(default_factory=lambda: list(US_CITIES))

    cost_add_re: float = 180.0
    cost_add_nre: float = 120.0
    cost_rem_re: float = 120.0
    cost_rem_nre: float = 180.0

    op_re:  List[float] = field(default_factory=lambda: [0.010, 0.013, 0.009, 0.008, 0.011, 0.014, 0.009, 0.011])
    op_nre: List[float] = field(default_factory=lambda: [0.038, 0.048, 0.035, 0.036, 0.042, 0.045, 0.040, 0.037])

    supply_re: float = 10.0
    supply_nre: float = 10.0

    # Reward weights.
    #
    # The original formulation added a per-step term proportional to the *raw*
    # remaining budget (0.15 * b).  Because that term is collected at every step,
    # a "do nothing" policy that simply preserves budget accumulated the largest
    # return, making inaction reward-optimal and inverting the intended policy
    # ranking.  We instead charge budget through an explicit *action cost* (money
    # actually spent this step) and reward the equity and decarbonization
    # objectives on the resulting state.  The discounted return is then aligned
    # with the planner's true objective: serve low-income demand with renewables
    # at low cost.
    w_action_cost: float = 0.02       # penalty per $M spent this step
    w_lowincome_served: float = 200.0    # bonus on low-income coverage served (0..1)
    w_pop_with_re: float = 80.0       # bonus on population fully served by RE

    discount: float = 0.95
    initial_budget: float = 3000.0
    max_energy_per_city: float = 300.0

    @property
    def n_cities(self) -> int:
        return len(self.cities)

    # ----- common MDP interface (shared solvers in solvers.py) -------------
    def noop(self) -> Action:
        return do_nothing()

    def state_key(self, s: State):
        return _state_key(s)

    def equity_score(self, s: State, a: Action) -> float:
        """>0 for an affordable RE addition that most reduces low-income
        shortfall (low-income cities weighted 2x); 0 otherwise. Used by the
        generic equity heuristic and the informed MCTS rollout."""
        if a.do_nothing or not a.add or a.energy != 0:
            return 0.0
        c = s.cities[a.city_idx]
        unmet = c.demand - (c.re_supply + c.nre_supply)
        if unmet <= 0:
            return 0.0
        return unmet * (2.0 if c.income == 0 else 1.0)

    def market_score(self, s: State, a: Action) -> float:
        """Naive 'easy win' signal used by the Market-Greedy heuristic: prefer the
        cheapest capacity addition to the largest raw deficit, ignoring equity.
        Here this favors adding cheaper NRE to high-demand (often high-income)
        cities. >0 for such an action, 0 otherwise."""
        if a.do_nothing or not a.add:
            return 0.0
        c = s.cities[a.city_idx]
        unmet = c.demand - (c.re_supply + c.nre_supply)
        if unmet <= 0:
            return 0.0
        # cheaper option (NRE) preferred; scale by raw deficit size
        cost = self.cost_add_nre if a.energy == 1 else self.cost_add_re
        return unmet * (self.cost_add_re / cost)

    def all_actions(self) -> List[Action]:
        acts = [do_nothing()]
        for i in range(self.n_cities):
            acts.append(Action(add=True,  energy=0, city_idx=i))
            acts.append(Action(add=False, energy=0, city_idx=i))
            acts.append(Action(add=True,  energy=1, city_idx=i))
            acts.append(Action(add=False, energy=1, city_idx=i))
        return acts

    def valid_actions(self, s: State) -> List[Action]:
        va = [do_nothing()]
        for i, c in enumerate(s.cities):
            if s.b >= self.cost_add_re and c.re_supply + self.supply_re <= self.max_energy_per_city:
                va.append(Action(add=True, energy=0, city_idx=i))
            if c.re_supply >= self.supply_re:
                va.append(Action(add=False, energy=0, city_idx=i))
            if s.b >= self.cost_add_nre and c.nre_supply + self.supply_nre <= self.max_energy_per_city:
                va.append(Action(add=True, energy=1, city_idx=i))
            if c.nre_supply >= self.supply_nre:
                va.append(Action(add=False, energy=1, city_idx=i))
        return va

    def transition(self, s: State, a: Action) -> State:
        if a.do_nothing:
            return State(b=s.b, cities=s.cities, total_demand=s.total_demand)

        i = a.city_idx
        cities = list(s.cities)
        c = cities[i]

        if a.energy == 0 and a.add:
            add_op = self.supply_re * self.op_re[i]
            bp = s.b - self.cost_add_re - add_op
            cities[i] = replace(c, re_supply=c.re_supply + self.supply_re)
        elif a.energy == 1 and a.add:
            add_op = self.supply_nre * self.op_nre[i]
            bp = s.b - self.cost_add_nre - add_op
            cities[i] = replace(c, nre_supply=c.nre_supply + self.supply_nre)
        elif a.energy == 0 and not a.add:
            op_red = self.supply_re * self.op_re[i]
            bp = min(s.b - self.cost_rem_re + op_red, self.initial_budget)
            cities[i] = replace(c, re_supply=c.re_supply - self.supply_re)
        else:
            op_red = self.supply_nre * self.op_nre[i]
            bp = min(s.b - self.cost_rem_nre + op_red, self.initial_budget)
            cities[i] = replace(c, nre_supply=c.nre_supply - self.supply_nre)

        return State(b=bp, cities=tuple(cities), total_demand=s.total_demand)

    def potential(self, s: State) -> float:
        """
        Potential Phi(s) for potential-based reward shaping (Ng et al., 1999).
        Higher when more low-income and total population is served. Shaping with
        F = gamma*Phi(s') - Phi(s) densifies the reward for planners (MCTS) that
        otherwise cannot see that a *sequence* of investments pays off, WITHOUT
        changing the optimal policy.
        """
        pop_low = sum(c.population for c in s.cities if c.income == 0) + 1e-6
        pop_tot = sum(c.population for c in s.cities) + 1e-6
        low_served = sum(c.population * min(1.0, (c.re_supply + c.nre_supply) / c.demand)
                         for c in s.cities if c.income == 0)
        re_served = sum(c.population * min(1.0, c.re_supply / c.demand) for c in s.cities)
        return 150.0 * (low_served / pop_low) + 60.0 * (re_served / pop_tot)

    def shaped_reward(self, s: State, a: Action, sp: State) -> float:
        """Reward plus potential-based shaping term (policy-invariant)."""
        return self.reward(s, a) + self.discount * self.potential(sp) - self.potential(s)

    def action_cost(self, a: Action) -> float:
        """Money (in $M) spent by an action this step (0 for removals / nothing)."""
        if a.do_nothing:
            return 0.0
        if a.add and a.energy == 0:
            return self.cost_add_re + self.supply_re * self.op_re[a.city_idx]
        if a.add and a.energy == 1:
            return self.cost_add_nre + self.supply_nre * self.op_nre[a.city_idx]
        return 0.0  # removals do not count as investment spend

    def reward(self, s: State, a: Action) -> float:
        """
        Reward for taking action a in state s.  Equity and decarbonization terms
        are evaluated on the *resulting* state s' = T(s, a) so that the benefit of
        an investment is credited in the same step it is made (the standard
        R(s, a, s') convention).  This removes the one-step delay that made
        myopic per-step reward under-value investment in the original model.
        """
        sp = self.transition(s, a)
        r = 0.0
        # (1) cost efficiency: penalize money actually invested this step
        r -= self.action_cost(a) * self.w_action_cost

        # (2) equity: population-weighted coverage *served* among low-income
        #     cities on the resulting state (continuous, 1 = all met, 0 = none).
        #     Rewarding coverage served (rather than penalizing the shortfall)
        #     puts the objective on a positive scale while preserving the policy
        #     ranking, since served = 1 - shortfall up to a constant.
        pop_low = sum(c.population for c in sp.cities if c.income == 0)
        low_served = 0.0
        for c in sp.cities:
            if c.income == 0:
                coverage = min(1.0, (c.re_supply + c.nre_supply) / c.demand)
                low_served += c.population * coverage
        r += (low_served / (pop_low + 1e-6)) * self.w_lowincome_served

        # (3) decarbonization: population-weighted RE coverage fraction on s'.
        pop_total = sum(c.population for c in sp.cities)
        re_cov = 0.0
        for c in sp.cities:
            re_cov += c.population * min(1.0, c.re_supply / c.demand)
        r += (re_cov / (pop_total + 1e-6)) * self.w_pop_with_re
        return r

    def is_terminal(self, s: State) -> bool:
        if s.b < 0:
            return True
        return all((c.re_supply + c.nre_supply) >= c.demand for c in s.cities)

    def initial_state(self, rng: np.random.Generator) -> State:
        budget = self.initial_budget * (1.0 + (rng.random() - 0.5) * 0.1)
        cities = []
        for c in self.cities:
            varied = c.demand * (1.0 + (rng.random() - 0.5) * 0.2)
            cities.append(replace(c, demand=varied))
        total_demand = sum(c.demand for c in cities)
        return State(b=budget, cities=tuple(cities), total_demand=total_demand)


def _state_key(s: State):
    return (round(s.b, 1),
            tuple((round(c.re_supply, 1), round(c.nre_supply, 1)) for c in s.cities))
