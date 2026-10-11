"""
indonesia_mdp.py — West Java (Jawa Barat) renewable-energy incentive/quota MDP.

Contrast with the U.S. model (energy_mdp.py). In Indonesia most renewable
capacity (PLTS, PLTA, PLTP, PLTB) is built by private developers / IPPs that
sell power to PLN, not by the government. The provincial government acts as a
*regulator and incentive provider*, not a builder. This module reformulates the
MDP accordingly:

  * Action space: the planner allocates INCENTIVE / QUOTA to a region, or funds
    an OFF-GRID communal project, or does nothing — it does NOT build plants.
  * On-grid deployment is realized by private developers and is capped by a
    per-cluster PLN QUOTA set roughly every 5 years via the RUPTL. Since the
    removal of net-metering (Permen ESDM No. 2/2024), rooftop/on-grid PLTS is
    governed by a regional quota system rather than 1:1 export credits.
  * Off-grid / communal deployment serves remote areas where grid extension is
    expensive (Jabar Selatan: Sukabumi, Garut, Tasikmalaya) and is realized
    directly by government-funded projects (no private crowd-in there).
  * Reward adds a LEVERAGE / CROWDING-IN term: an incentive is "good" when it
    pulls private capital into regions the market would otherwise ignore
    (poor / remote), and is wasteful when it merely subsidizes projects that
    would have proceeded anyway in attractive urban clusters.

All figures are DUMMY but calibrated to be plausible for West Java.
"""

from __future__ import annotations
from dataclasses import dataclass, replace, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class Region:
    name: str
    demand: float          # electricity demand (arb. capacity units)
    re_installed: float    # RE already installed (private + gov)
    nre_installed: float   # non-RE installed
    population: float
    rural: int             # 1 = remote/rural (grid extension costly), 0 = urban
    low_access: int        # 1 = electrification / access gap, 0 = well served
    market_attractiveness: float  # 0..1 private-investor appetite (natural pull)
    pln_quota: float       # remaining on-grid quota for this cluster (RUPTL cap)


@dataclass(frozen=True)
class INState:
    b: float                        # remaining incentive budget (IDR bn, dummy)
    regions: Tuple[Region, ...]
    private_leverage: float         # cumulative private capital crowded in


@dataclass(frozen=True)
class INAction:
    kind: str                       # "incentive" | "offgrid" | "quota" | "nothing"
    region_idx: Optional[int] = None

    def __repr__(self):
        if self.kind == "nothing":
            return "nothing"
        return f"{self.kind}@{self.region_idx}"


def in_nothing() -> INAction:
    return INAction(kind="nothing")


# ---------------------------------------------------------------------------
# West Java regions (dummy but plausible).  Jabar Selatan (Sukabumi, Garut,
# Tasikmalaya) is remote with lower market appetite; Bandung Raya / Bekasi /
# Bogor are attractive urban clusters the market already serves.
# ---------------------------------------------------------------------------
WEST_JAVA = [
    #        name          demand  re    nre   pop        rural low  mkt   quota
    Region("Bandung Raya",  120.0, 12.0, 40.0, 8_600_000,  0,   0,  0.85, 30.0),
    Region("Bekasi",        140.0,  8.0, 55.0, 3_100_000,  0,   0,  0.90, 25.0),
    Region("Bogor",         110.0,  9.0, 38.0, 5_500_000,  0,   0,  0.80, 25.0),
    Region("Karawang",       95.0,  6.0, 42.0, 2_400_000,  0,   0,  0.75, 20.0),
    Region("Sukabumi",       55.0,  5.0, 12.0, 2_700_000,  1,   1,  0.35, 15.0),
    Region("Garut",          48.0,  4.0,  9.0, 2_600_000,  1,   1,  0.28, 12.0),
    Region("Tasikmalaya",    50.0,  4.0, 10.0, 1_900_000,  1,   1,  0.30, 12.0),
    Region("Cirebon",        70.0,  5.0, 28.0, 2_200_000,  0,   1,  0.55, 18.0),
]


@dataclass
class IndonesiaEnergyMDP:
    """West Java incentive/quota MDP (government as regulator, not builder)."""
    regions: List[Region] = field(default_factory=lambda: [replace(r) for r in WEST_JAVA])

    # Incentive mechanics
    incentive_cost: float = 40.0        # cost to place one incentive package
    offgrid_cost: float = 90.0          # cost of a government off-grid communal project
    quota_cost: float = 5.0             # admin cost of releasing extra on-grid quota

    # Capacity realized per action. Sized (together with the budget below) so
    # that an effective incentive/off-grid program can substantially close the
    # low-access gap over the planning horizon; otherwise the access objective is
    # unreachable and every policy is penalized regardless of quality.
    incentive_capacity: float = 20.0    # RE capacity a private developer adds if incentivized
    offgrid_capacity: float = 18.0      # RE capacity delivered by a communal project
    quota_release: float = 20.0         # extra on-grid quota released to a cluster

    # How strongly private capital responds. In attractive markets, an incentive
    # crowds in a lot of private money (high leverage) but is partly deadweight
    # (they'd have built anyway). In remote/poor markets, private appetite is low
    # so the *additional* capacity is fully attributable to the incentive.
    base_leverage: float = 3.0          # IDR of private capital per IDR incentive (max)

    # Reward weights.  The access term rewards coverage *served* among low-access
    # regions (positive, 0..1), mirroring the U.S. model, rather than penalizing
    # the shortfall; this puts the objective on a positive scale while preserving
    # the policy ranking.
    w_cost: float = -0.02               # penalty per unit incentive budget spent
    w_access_coverage: float = 180.0    # bonus on low-access coverage served (0..1)
    w_re_coverage: float = 70.0         # bonus on RE coverage of demand
    w_leverage: float = 25.0            # bonus on ADDITIONAL private capital crowded in
    w_deadweight: float = -20.0         # penalty for subsidizing already-attractive markets

    discount: float = 0.95
    initial_budget: float = 1000.0      # sufficient to make the access gap closeable
    max_re_per_region: float = 200.0

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    # ----- common MDP interface (shared solvers in solvers.py) -------------
    def noop(self) -> INAction:
        return in_nothing()

    def state_key(self, s: INState):
        return (round(s.b, 1),
                tuple((round(r.re_installed, 1), round(r.pln_quota, 1)) for r in s.regions))

    def equity_score(self, s: INState, a: INAction) -> float:
        """>0 for an incentive/off-grid action that most reduces the shortfall of
        a low-access region (off-grid to rural weighted higher); 0 otherwise.
        Used by the generic equity heuristic and the informed MCTS rollout."""
        if a.kind == "nothing":
            return 0.0
        r = s.regions[a.region_idx]
        if r.low_access != 1:
            return 0.0
        short = max(0.0, r.demand - (r.re_installed + r.nre_installed))
        if short <= 0:
            return 0.0
        pref = 1.5 if (a.kind == "offgrid" and r.rural == 1) else 1.0
        return short * pref

    def market_score(self, s: INState, a: INAction) -> float:
        """Naive 'easy win' signal used by the Market-Greedy heuristic: subsidize
        the most market-attractive (urban) clusters first, where private uptake
        is easiest, ignoring access equity. >0 for an on-grid incentive to such a
        cluster, 0 otherwise."""
        if a.kind != "incentive":
            return 0.0
        return s.regions[a.region_idx].market_attractiveness

    # -- actions ------------------------------------------------------------
    def all_actions(self) -> List[INAction]:
        acts = [in_nothing()]
        for i in range(self.n_regions):
            acts.append(INAction("incentive", i))
            acts.append(INAction("offgrid", i))
            acts.append(INAction("quota", i))
        return acts

    def valid_actions(self, s: INState) -> List[INAction]:
        va = [in_nothing()]
        for i, r in enumerate(s.regions):
            # on-grid incentive requires remaining PLN quota and budget
            if s.b >= self.incentive_cost and r.pln_quota >= self.incentive_capacity \
               and r.re_installed + self.incentive_capacity <= self.max_re_per_region:
                va.append(INAction("incentive", i))
            # off-grid communal project: for rural regions, no quota needed
            if s.b >= self.offgrid_cost and r.rural == 1 \
               and r.re_installed + self.offgrid_capacity <= self.max_re_per_region:
                va.append(INAction("offgrid", i))
            # release extra on-grid quota to a cluster (enables future private RE)
            if s.b >= self.quota_cost:
                va.append(INAction("quota", i))
        return va

    # -- leverage model -----------------------------------------------------
    def _crowd_in(self, r: Region) -> Tuple[float, float, float]:
        """
        Returns (realized_capacity, additional_private_capital, deadweight).
        Additional capital is what the incentive genuinely *added*; deadweight is
        the part that would have happened anyway (scales with market appetite).
        """
        realized = self.incentive_capacity * (0.4 + 0.6 * r.market_attractiveness)
        private_capital = self.base_leverage * self.incentive_cost * r.market_attractiveness
        # additionality is high where the market would NOT have acted (low appetite)
        additionality = 1.0 - r.market_attractiveness
        additional = private_capital * additionality
        deadweight = private_capital * r.market_attractiveness
        return realized, additional, deadweight

    # -- transition ---------------------------------------------------------
    def transition(self, s: INState, a: INAction) -> INState:
        if a.kind == "nothing":
            return INState(b=s.b, regions=s.regions, private_leverage=s.private_leverage)

        i = a.region_idx
        regions = list(s.regions)
        r = regions[i]
        b = s.b
        lev = s.private_leverage

        if a.kind == "incentive":
            realized, additional, _dead = self._crowd_in(r)
            b -= self.incentive_cost
            regions[i] = replace(r,
                                 re_installed=r.re_installed + realized,
                                 pln_quota=r.pln_quota - self.incentive_capacity)
            lev += additional
        elif a.kind == "offgrid":
            b -= self.offgrid_cost
            regions[i] = replace(r, re_installed=r.re_installed + self.offgrid_capacity)
            # off-grid is fully additional (no private market there)
            lev += self.offgrid_cost * 0.5
        elif a.kind == "quota":
            b -= self.quota_cost
            regions[i] = replace(r, pln_quota=r.pln_quota + self.quota_release)

        return INState(b=b, regions=tuple(regions), private_leverage=lev)

    # -- reward -------------------------------------------------------------
    def reward(self, s: INState, a: INAction) -> float:
        # Access-equity and RE-coverage terms are evaluated on the resulting
        # state s' (R(s, a, s') convention), so the effect of an incentive /
        # off-grid project is credited in the step it is issued.
        sp = self.transition(s, a)
        rr = 0.0
        # (1) budget spent this step
        if a.kind == "incentive":
            rr += self.w_cost * self.incentive_cost
        elif a.kind == "offgrid":
            rr += self.w_cost * self.offgrid_cost
        elif a.kind == "quota":
            rr += self.w_cost * self.quota_cost

        # (2) energy access equity: coverage *served* among low-access regions
        #     (on s'), positive and in [0,1]; mirrors the U.S. model's served term.
        pop_low = sum(r.population for r in sp.regions if r.low_access == 1)
        low_served = 0.0
        for r in sp.regions:
            if r.low_access == 1:
                coverage = min(1.0, (r.re_installed + r.nre_installed) / r.demand)
                low_served += r.population * coverage
        rr += (low_served / (pop_low + 1e-6)) * self.w_access_coverage

        # (3) RE coverage (population weighted, on s')
        pop_total = sum(r.population for r in sp.regions)
        re_cov = sum(r.population * min(1.0, r.re_installed / r.demand) for r in sp.regions)
        rr += (re_cov / (pop_total + 1e-6)) * self.w_re_coverage

        # (4) leverage / crowding-in bonus and deadweight penalty. An incentive
        #     is rewarded for the ADDITIONAL private capital it crowds in
        #     (highest where the market would not otherwise invest — poor/remote
        #     regions) and penalized for DEADWEIGHT (subsidizing attractive urban
        #     clusters that would have invested anyway).
        if a.kind == "incentive":
            _realized, additional, deadweight = self._crowd_in(s.regions[a.region_idx])
            rr += self.w_leverage * (additional / (self.base_leverage * self.incentive_cost))
            rr += self.w_deadweight * (deadweight / (self.base_leverage * self.incentive_cost))
        return rr

    def is_terminal(self, s: INState) -> bool:
        if s.b < 0:
            return True
        return all((r.re_installed + r.nre_installed) >= r.demand for r in s.regions)

    def initial_state(self, rng: np.random.Generator) -> INState:
        budget = self.initial_budget * (1.0 + (rng.random() - 0.5) * 0.1)
        regions = []
        for r in self.regions:
            varied = r.demand * (1.0 + (rng.random() - 0.5) * 0.2)
            regions.append(replace(r, demand=varied))
        return INState(b=budget, regions=tuple(regions), private_leverage=0.0)
