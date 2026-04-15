from __future__ import annotations

from typing import List
import numpy as np

from model import Individual, Parameters, Patch, SimulationState


def pairwise_distances(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def create_landscape(params: Parameters, rng: np.random.Generator) -> List[Patch]:
    coords = rng.uniform(0.0, params.landscape_size, size=(params.n_patches, 2))

    # Smooth spatial environmental gradient + local noise
    raw_env = (
        0.8 * (coords[:, 0] / params.landscape_size - 0.5)
        - 0.6 * (coords[:, 1] / params.landscape_size - 0.5)
        + rng.normal(0.0, 0.15, size=params.n_patches)
    )

    capacities = rng.integers(low=40, high=90, size=params.n_patches)
    disturbance_probs = rng.uniform(0.02, 0.08, size=params.n_patches)

    patches = [
        Patch(
            patch_id=i,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            env_optimum=float(raw_env[i]),
            carrying_capacity=int(capacities[i]),
            disturbance_prob=float(disturbance_probs[i]),
        )
        for i in range(params.n_patches)
    ]
    return patches


def initialize_individuals(
    params: Parameters, patches: List[Patch], rng: np.random.Generator
) -> list[Individual]:
    individuals: list[Individual] = []

    species_trait_means = rng.normal(
        loc=0.0,
        scale=params.species_pool_trait_sd,
        size=params.initial_species,
    )

    for patch in patches:
        species_choices = rng.choice(
            np.arange(params.initial_species),
            size=params.initial_individuals_per_patch,
            replace=True,
        )
        for sp in species_choices:
            trait = rng.normal(species_trait_means[sp], 0.15)
            individuals.append(
                Individual(
                    species_id=int(sp),
                    trait=float(trait),
                    patch_id=patch.patch_id,
                    alive=True,
                )
            )

    return individuals


def initialize_state(params: Parameters) -> SimulationState:
    rng = np.random.default_rng(params.random_seed)
    patches = create_landscape(params, rng)
    coords = np.array([[p.x, p.y] for p in patches], dtype=float)
    dist_matrix = pairwise_distances(coords)
    individuals = initialize_individuals(params, patches, rng)

    return SimulationState(
        patches=patches,
        individuals=individuals,
        distance_matrix=dist_matrix,
    )
