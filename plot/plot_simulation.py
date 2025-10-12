import os

import tyro
from dataclasses import dataclass

from plot.plot_utils import *
from utils import ArrayIO
from valuation.experiment import ValuationExperiment


@dataclass
class Args:
    exp_name: Optional[str] = None
    """Name of the valuation experiment."""
    agent_path: str = "models/kangaroo_demo"
    sim_name: str = "train"
    """Name of the simulation."""
    num_episodes: int = 10
    """Number of episodes to plot."""

def main():
    # Parse arguments
    args = tyro.cli(Args)

    # Load experiment
    if args.exp_name is None:
        experiment = ValuationExperiment.from_path(Path(args.agent_path))
    else:
        experiment = ValuationExperiment.from_name(args.exp_name)

    # Load data
    data_dir = experiment.logs_dir / args.sim_name
    data = ArrayIO(data_dir)
    ep_indices = data["ep_indices"].reshape(-1)
    blending_weights = data["blending_weights"]

    # Create plot dir
    plots_dir = experiment.plots_dir / "sim"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot neural blending weights
    for ep_nr in range(args.num_episodes):
        beta_fig, beta_ax = plt.subplots(figsize=(5, 2))
        _ep_indices = (ep_indices == ep_nr)
        neural_weights = blending_weights[_ep_indices, 0]
        beta_ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        beta_ax.plot(neural_weights, linewidth=0.75)
        beta_ax.set_ylabel(r"$\beta$")
        beta_ax.set_xlabel("Step")
        beta_ax.set_ylim(0, 1)
        beta_ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        beta_ax.set_title(experiment.env_name.capitalize())
        beta_fig.tight_layout()
        save_fig(beta_fig, plots_dir / f"beta_ep{ep_nr}.png", dpi=200)


if __name__ == "__main__":
    main()
