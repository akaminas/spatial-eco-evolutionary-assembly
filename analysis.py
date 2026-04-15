from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import SimulationState


def histories_to_dataframes(state: SimulationState) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    community_df = pd.DataFrame(state.community_history)
    trait_df = pd.DataFrame(state.trait_history)
    disturbance_df = pd.DataFrame(state.disturbance_history)
    return community_df, trait_df, disturbance_df


def save_outputs(
    state: SimulationState,
    output_dir: str = "outputs",
    figure_dir: str = "figures",
) -> None:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    Path(figure_dir).mkdir(exist_ok=True, parents=True)

    community_df, trait_df, disturbance_df = histories_to_dataframes(state)

    community_df.to_csv(Path(output_dir) / "community_history.csv", index=False)
    trait_df.to_csv(Path(output_dir) / "trait_history.csv", index=False)
    disturbance_df.to_csv(Path(output_dir) / "disturbance_history.csv", index=False)

    plot_regional_dynamics(community_df, Path(figure_dir) / "regional_dynamics.png")
    plot_trait_environment_tracking(trait_df, Path(figure_dir) / "trait_environment_tracking.png")
    plot_patch_trait_map(state, trait_df, Path(figure_dir) / "patch_trait_map_final.png")
    plot_disturbance_summary(disturbance_df, Path(figure_dir) / "disturbance_frequency.png")


def plot_regional_dynamics(community_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(community_df["time"], community_df["regional_richness"], label="Regional richness")
    ax.plot(community_df["time"], community_df["mean_local_richness"], label="Mean local richness")
    ax.set_xlabel("Time")
    ax.set_ylabel("Richness")
    ax.set_title("Community diversity through time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_trait_environment_tracking(trait_df: pd.DataFrame, output_path: Path) -> None:
    agg = (
        trait_df.groupby("time")[["mean_trait", "env_optimum"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agg["time"], agg["mean_trait"], label="Mean patch trait")
    ax.plot(agg["time"], agg["env_optimum"], label="Mean patch environment")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean value")
    ax.set_title("Tracking of environment by trait means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_patch_trait_map(
    state: SimulationState, trait_df: pd.DataFrame, output_path: Path
) -> None:
    final_time = trait_df["time"].max()
    final_df = trait_df[trait_df["time"] == final_time].copy()

    x = np.array([p.x for p in state.patches], dtype=float)
    y = np.array([p.y for p in state.patches], dtype=float)

    merged = final_df.sort_values("patch_id")
    color_values = merged["mean_trait"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(x, y, c=color_values, s=120, edgecolor="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Final patch-level mean trait")
    plt.colorbar(sc, ax=ax, label="Mean trait")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_disturbance_summary(disturbance_df: pd.DataFrame, output_path: Path) -> None:
    patch_counts = (
        disturbance_df.groupby("patch_id")["disturbed"]
        .sum()
        .reset_index()
        .rename(columns={"disturbed": "n_disturbances"})
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(patch_counts["patch_id"], patch_counts["n_disturbances"])
    ax.set_xlabel("Patch")
    ax.set_ylabel("Number of disturbances")
    ax.set_title("Disturbance frequency by patch")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
