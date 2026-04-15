from __future__ import annotations

from typing import Dict, List
import numpy as np

from model import Individual, Parameters, SimulationState


def update_patch_environments(
    state: SimulationState, params: Parameters, rng: np.random.Generator
) -> None:
    for patch in state.patches:
        patch.env_optimum += params.env_directional_shift + rng.normal(
            0.0, params.env_noise_sd
        )


def environmental_survival_probability(
    trait: float, env_optimum: float, params: Parameters
) -> float:
    mismatch = trait - env_optimum
    env_component = np.exp(-(mismatch ** 2) / (2.0 * params.env_filter_sigma ** 2))
    p = params.survival_baseline * env_component
    return float(np.clip(p, 0.0, 1.0))


def trait_competition_load(
    focal_trait: float, neighbors: List[Individual], params: Parameters
) -> float:
    if not neighbors:
        return 0.0
    diffs = np.array([focal_trait - ind.trait for ind in neighbors], dtype=float)
    overlap = np.exp(-(diffs ** 2) / (2.0 * params.competition_sigma ** 2))
    return float(params.competition_strength * np.sum(overlap))


def survival_step(
    state: SimulationState, params: Parameters, rng: np.random.Generator
) -> None:
    by_patch = state.alive_by_patch()

    for patch in state.patches:
        local_inds = by_patch[patch.patch_id]
        for focal in local_inds:
            neighbors = [ind for ind in local_inds if ind is not focal]
            p_survive = environmental_survival_probability(
                focal.trait, patch.env_optimum, params
            )
            comp_load = trait_competition_load(focal.trait, neighbors, params)
            p_survive *= np.exp(-comp_load)
            p_survive = float(np.clip(p_survive, 0.0, 1.0))
            focal.alive = bool(rng.random() < p_survive)


def choose_target_patch(
    source_patch_id: int,
    distance_matrix: np.ndarray,
    params: Parameters,
    rng: np.random.Generator,
) -> int:
    distances = distance_matrix[source_patch_id]
    weights = np.exp(-distances / params.dispersal_scale)
    weights[source_patch_id] = 0.0
    total = np.sum(weights)

    if total <= 0.0:
        return source_patch_id

    probs = weights / total
    return int(rng.choice(np.arange(len(distances)), p=probs))


def reproduction_step(
    state: SimulationState, params: Parameters, rng: np.random.Generator
) -> list[Individual]:
    offspring: list[Individual] = []
    by_patch = state.alive_by_patch()

    for patch in state.patches:
        local_inds = by_patch[patch.patch_id]
        if not local_inds:
            continue

        local_density = len(local_inds) / patch.carrying_capacity
        density_penalty = np.exp(-1.5 * max(local_density - 1.0, 0.0))

        for parent in local_inds:
            match = np.exp(
                -((parent.trait - patch.env_optimum) ** 2)
                / (2.0 * params.env_filter_sigma ** 2)
            )
            expected_offspring = params.fecundity_max * match * density_penalty

            n_offspring = int(rng.poisson(lam=max(expected_offspring, 0.0)))
            for _ in range(n_offspring):
                child_trait = parent.trait + rng.normal(0.0, params.mutation_sd)

                if rng.random() < params.dispersal_prob:
                    target_patch = choose_target_patch(
                        source_patch_id=patch.patch_id,
                        distance_matrix=state.distance_matrix,
                        params=params,
                        rng=rng,
                    )
                else:
                    target_patch = patch.patch_id

                offspring.append(
                    Individual(
                        species_id=parent.species_id,
                        trait=float(child_trait),
                        patch_id=int(target_patch),
                        alive=True,
                    )
                )

    return offspring


def recruitment_step(
    state: SimulationState,
    offspring: list[Individual],
    params: Parameters,
    rng: np.random.Generator,
) -> None:
    current_alive = state.alive_by_patch()
    recruits_by_patch: Dict[int, list[Individual]] = {p.patch_id: [] for p in state.patches}
    for child in offspring:
        recruits_by_patch[child.patch_id].append(child)

    accepted_recruits: list[Individual] = []

    for patch in state.patches:
        residents = current_alive[patch.patch_id]
        n_residents = len(residents)
        available_slots = max(patch.carrying_capacity - n_residents, 0)

        candidates = recruits_by_patch[patch.patch_id]
        if not candidates or available_slots == 0:
            continue

        scored_candidates: list[tuple[float, Individual]] = []
        for child in candidates:
            env_match = np.exp(
                -((child.trait - patch.env_optimum) ** 2)
                / (2.0 * params.env_filter_sigma ** 2)
            )
            comp_load = trait_competition_load(child.trait, residents, params)
            p_establish = params.recruitment_base_prob * env_match * np.exp(-comp_load)
            p_establish = float(np.clip(p_establish, 0.0, 1.0))

            if rng.random() < p_establish:
                score = p_establish + 1e-6 * rng.random()
                scored_candidates.append((score, child))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        accepted_recruits.extend([child for _, child in scored_candidates[:available_slots]])

    state.individuals.extend(accepted_recruits)


def disturbance_step(
    state: SimulationState, params: Parameters, rng: np.random.Generator, t: int
) -> None:
    by_patch = state.alive_by_patch()

    for patch in state.patches:
        disturbed = rng.random() < patch.disturbance_prob
        if disturbed:
            local_inds = by_patch[patch.patch_id]
            n_before = len(local_inds)
            for ind in local_inds:
                if rng.random() < params.disturbance_mortality:
                    ind.alive = False
            n_after = sum(ind.alive for ind in local_inds)
            state.disturbance_history.append(
                {
                    "time": t,
                    "patch_id": patch.patch_id,
                    "disturbed": 1,
                    "n_before": n_before,
                    "n_after": n_after,
                }
            )
        else:
            state.disturbance_history.append(
                {
                    "time": t,
                    "patch_id": patch.patch_id,
                    "disturbed": 0,
                    "n_before": len(by_patch[patch.patch_id]),
                    "n_after": len(by_patch[patch.patch_id]),
                }
            )


def prune_dead(state: SimulationState) -> None:
    state.individuals = [ind for ind in state.individuals if ind.alive]


def jaccard_similarity(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def record_state(state: SimulationState, t: int) -> None:
    alive = state.alive_individuals()
    by_patch = state.alive_by_patch()
    species_by_patch = state.alive_species_by_patch()

    local_richness = np.array([len(species_by_patch[p.patch_id]) for p in state.patches], dtype=float)
    local_abundance = np.array([len(by_patch[p.patch_id]) for p in state.patches], dtype=float)

    regional_species = {ind.species_id for ind in alive}
    all_traits = np.array([ind.trait for ind in alive], dtype=float) if alive else np.array([])

    similarities = []
    patch_ids = [p.patch_id for p in state.patches]
    for i in range(len(patch_ids)):
        for j in range(i + 1, len(patch_ids)):
            sim = jaccard_similarity(species_by_patch[patch_ids[i]], species_by_patch[patch_ids[j]])
            similarities.append(sim)

    beta_diversity = float(1.0 - np.mean(similarities)) if similarities else 0.0

    state.community_history.append(
        {
            "time": t,
            "regional_richness": len(regional_species),
            "mean_local_richness": float(np.mean(local_richness)),
            "mean_local_abundance": float(np.mean(local_abundance)),
            "beta_diversity_jaccard": beta_diversity,
            "total_abundance": len(alive),
        }
    )

    for patch in state.patches:
        patch_inds = by_patch[patch.patch_id]
        patch_traits = np.array([ind.trait for ind in patch_inds], dtype=float)
        state.trait_history.append(
            {
                "time": t,
                "patch_id": patch.patch_id,
                "env_optimum": patch.env_optimum,
                "mean_trait": float(np.mean(patch_traits)) if len(patch_traits) > 0 else np.nan,
                "trait_variance": float(np.var(patch_traits)) if len(patch_traits) > 1 else np.nan,
                "abundance": len(patch_inds),
                "richness": len(species_by_patch[patch.patch_id]),
            }
        )


def step_simulation(
    state: SimulationState, params: Parameters, rng: np.random.Generator, t: int
) -> None:
    update_patch_environments(state, params, rng)
    survival_step(state, params, rng)
    prune_dead(state)
    offspring = reproduction_step(state, params, rng)
    recruitment_step(state, offspring, params, rng)
    disturbance_step(state, params, rng, t)
    prune_dead(state)
    record_state(state, t)
