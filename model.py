from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Individual:
    species_id: int
    trait: float
    patch_id: int
    alive: bool = True


@dataclass
class Patch:
    patch_id: int
    x: float
    y: float
    env_optimum: float
    carrying_capacity: int
    disturbance_prob: float


@dataclass
class Parameters:
    n_patches: int = 25
    landscape_size: float = 10.0
    initial_species: int = 20
    initial_individuals_per_patch: int = 30
    timesteps: int = 200

    env_noise_sd: float = 0.03
    env_directional_shift: float = 0.002

    survival_baseline: float = 0.92
    env_filter_sigma: float = 0.8

    fecundity_max: int = 3
    recruitment_base_prob: float = 0.65

    competition_sigma: float = 0.45
    competition_strength: float = 0.025

    mutation_sd: float = 0.05
    dispersal_scale: float = 2.0
    dispersal_prob: float = 0.25

    disturbance_mortality: float = 0.75
    species_pool_trait_sd: float = 1.0

    random_seed: int = 42


@dataclass
class SimulationState:
    patches: List[Patch]
    individuals: List[Individual]
    distance_matrix: np.ndarray
    trait_history: List[Dict[str, float]] = field(default_factory=list)
    community_history: List[Dict[str, float]] = field(default_factory=list)
    disturbance_history: List[Dict[str, float]] = field(default_factory=list)

    def alive_individuals(self) -> List[Individual]:
        return [ind for ind in self.individuals if ind.alive]

    def alive_by_patch(self) -> Dict[int, List[Individual]]:
        grouped: Dict[int, List[Individual]] = {p.patch_id: [] for p in self.patches}
        for ind in self.individuals:
            if ind.alive:
                grouped[ind.patch_id].append(ind)
        return grouped

    def alive_species_by_patch(self) -> Dict[int, set[int]]:
        grouped = self.alive_by_patch()
        return {
            patch_id: {ind.species_id for ind in inds}
            for patch_id, inds in grouped.items()
        }

    def regional_species(self) -> set[int]:
        return {ind.species_id for ind in self.individuals if ind.alive}


def species_trait_map(individuals: List[Individual]) -> Dict[int, float]:
    values: Dict[int, List[float]] = {}
    for ind in individuals:
        if ind.alive:
            values.setdefault(ind.species_id, []).append(ind.trait)
    return {sp: float(np.mean(ts)) for sp, ts in values.items()}
