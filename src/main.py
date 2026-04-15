from __future__ import annotations

import numpy as np

from model import Parameters
from landscape import initialize_state
from processes import record_state, step_simulation
from analysis import save_outputs


def run_simulation() -> None:
    params = Parameters()
    rng = np.random.default_rng(params.random_seed)

    state = initialize_state(params)

    # Record initial condition at time 0
    record_state(state, 0)

    for t in range(1, params.timesteps + 1):
        step_simulation(state, params, rng, t)

    save_outputs(state)

    final = state.community_history[-1]
    print("Simulation complete")
    print(f"Final regional richness: {final['regional_richness']}")
    print(f"Final mean local richness: {final['mean_local_richness']:.2f}")
    print(f"Final beta diversity (1 - mean Jaccard): {final['beta_diversity_jaccard']:.3f}")
    print(f"Final total abundance: {final['total_abundance']}")


if __name__ == "__main__":
    run_simulation()
