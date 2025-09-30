from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import tyro
from dataclasses import dataclass, field
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from plot.plot_utils import get_cmap
from utils import ArrayIO, optional
from valuation.experiment import ValuationExperiment

from prettytable import PrettyTable


@dataclass
class Args:
    exp_names: List[str] = field(default_factory=lambda: [])
    """Name of the valuation experiment. If 'None', the static model will be evaluated."""
    include_baseline: bool = False
    """Whether to include static model"""
    agent_path: Optional[str] = None
    """Path to pretrained BlendRL model"""
    baseline_actor_modes: List[str] = field(default_factory=lambda: ["hybrid"])
    """Baseline actor modes"""
    exp_actor_modes: List[str] = field(default_factory=lambda: ["hybrid"])
    """Experiment actor modes"""
    include_p: bool = False
    """Whether to include p-values"""
    bin_size: int = 500
    """Bin size."""
    show_plot: bool = False

STATIC_EXPERIMENT_NAME = "BlendRL"
STATIC_COLOR = "black"
CMAP = get_cmap("Dark2")
COLORS = [CMAP(v) for v in np.linspace(0, 1, 9)[:-1]]
STATIC_CMAP = get_cmap("Set1")
STATIC_COLORS = {
    "hybrid": "black",
    "logic": "blue",
    "neural": "red"
}
BAR_WIDTH = 0.7
P_TEXT_X_OFFSET = 0.01
P_TEXT_Y_OFFSET = -0.04
P_TEXT_Y_INTER_OFFSET = 0.04

def get_p_text(p: float, thresholds: List[float], precision: int = 3) -> str:
    sorted_thresholds = sorted(thresholds, reverse=True)
    for i, threshold in enumerate(sorted_thresholds):
        if p >= threshold:
            if i == 0:
                return "={:.{prec}g}".format(p, prec=precision)
            else:
                return f"<{sorted_thresholds[i-1]}"
    return f"<{sorted_thresholds[-1]}"

def print_stats(stats: dict, indent = 0):
    for key, value in stats.items():
        if not isinstance(value, dict):
            print('\t'*indent + f"{key}: {value}")
        else:
            print('\t'*indent + f"{key}:")
            print_stats(value, indent + 1)

@dataclass
class ExpData:
    base_exp_name: str
    seed: Optional[int] = None

    @staticmethod
    def from_exp_name(exp_name: str) -> ExpData:
        ri = exp_name.rindex("_")
        seed = int(exp_name[ri+2:])
        base_exp_name = exp_name[:ri]
        return ExpData(base_exp_name, seed)

def main():
    # Parse arguments
    args = tyro.cli(Args)
    bin_size = args.bin_size

    # Gather data
    data = []

    # Get experiments
    data_keys = []
    if args.include_baseline:
        data_keys.extend([(None, am) for am in args.baseline_actor_modes])
    for exp_name in args.exp_names:
        data_keys.extend([(exp_name, am) for am in args.exp_actor_modes])

    for i, (exp_name, actor_mode) in enumerate(data_keys):
        is_baseline = exp_name is None
        if not is_baseline:
            experiment = ValuationExperiment.from_name(exp_name)
            exp_data = ExpData.from_exp_name(exp_name)
        else:
            experiment = ValuationExperiment.from_path(Path(args.agent_path))
            exp_data = ExpData(exp_name, None)

        exp_label = optional(exp_name, STATIC_EXPERIMENT_NAME) + f" ({actor_mode})"

        data_dir = experiment.logs_dir / f"test_{actor_mode}"

        # Collect data
        try:
            reader = ArrayIO(data_dir)
            ep_lengths = reader["ep_lengths"].flatten().astype(np.int32)
            ep_returns = reader["ep_returns"].flatten()
            ep_indices = reader["ep_indices"].flatten().astype(np.int32)
            num_episodes = len(ep_lengths)

            num_goals = 0
            reached_goal = None
            if experiment.env_name == "kangaroo" and "custom_reached_child" in reader:
                reached_goal = reader["custom_reached_child"]
            elif experiment.env_name == "seaquest" and "custom_rescued_divers" in reader:
                reached_goal = reader["custom_rescued_divers"]

            if reached_goal is not None:
                indices = np.argwhere(ep_indices < num_episodes)
                indices = indices[np.argwhere(indices < len(reached_goal))]
                reached_goal = reached_goal[indices]

                num_goals = reached_goal.sum() / num_episodes

            # Collect stats
            stats: dict = {
                "num_episodes": num_episodes,
            }
            for stat_name, values in (("episodic_lengths", ep_lengths), ("episodic_returns", ep_returns)):
                stats[stat_name] = {
                    #"values": values.tolist(),
                    "mean": np.mean(values),
                    "min": np.min(values),
                    "q25": np.percentile(values, 25),
                    "q50": np.median(values),
                    "q75": np.percentile(values, 75),
                    "max": np.max(values),
                    "std": np.std(values),
                }

            #print(exp_label)
            #print_stats(stats)
            #print("num_goals:", num_goals)

            color = STATIC_COLORS[actor_mode] if is_baseline else COLORS[(i - len(args.baseline_actor_modes)) % 8]

            data.append({
                "exp_name": exp_name,
                "ep_returns": ep_returns,
                "num_goals": num_goals,
                "num_episodes": num_episodes,
                "actor_mode": actor_mode,
                "is_baseline": is_baseline,
                "label": exp_label,
                "color": color,
                "base_exp_name": exp_data.base_exp_name,
                "seed": exp_data.seed,
                "stats": stats
            })
        except Exception as e:
            print(e)
            print(f"Skipping {exp_label} ...")

    # Print stats
    seed_table = PrettyTable(field_names=["exp", "actor mode", "seed", "num eps", "ep length (mean)", "ep length (std)", "ep return (mean)", "ep return (std)", "num goals"])
    exp_stats = dict()
    for data_row in data:
        key = (data_row["base_exp_name"], data_row["actor_mode"])
        if key not in exp_stats:
            exp_stats[key] = defaultdict(list)
        seed_table.add_row([data_row["base_exp_name"], data_row["actor_mode"], data_row["seed"], data_row["num_episodes"], data_row["stats"]["episodic_lengths"]["mean"], data_row["stats"]["episodic_lengths"]["std"], data_row["stats"]["episodic_returns"]["mean"], data_row["stats"]["episodic_returns"]["std"], data_row["num_goals"]])
        for metric in ("episodic_lengths", "episodic_returns"):
            exp_stats[key][metric].append(data_row["stats"][metric]["mean"])
        exp_stats[key]["num_goals"].append(data_row["num_goals"])
    print(seed_table)

    exp_table = PrettyTable(field_names=["exp", "actor mode", "ep return (mean)", "ep return (std)", "num goals (mean)", "num goals (std)"])
    for (base_exp_name, actor_mode), exp_stat in exp_stats.items():
        exp_row = [base_exp_name, actor_mode]
        for metric in ("episodic_returns", "num_goals"):
            exp_row.append(np.mean(exp_stat[metric]))
            exp_row.append(np.std(exp_stat[metric]))
        exp_table.add_row(exp_row)
    print(exp_table)

    # Create subplot
    if args.show_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        num_exps = len(data)
        min_ep_return = min([min(d["ep_returns"]) for d in data])
        max_ep_return = max([max(d["ep_returns"]) for d in data])
        min_bin_index = int(math.floor(min_ep_return / bin_size))
        min_bin = min_bin_index * bin_size
        max_bin_index = int(math.ceil(max_ep_return / bin_size))
        max_bin = max_bin_index * bin_size
        num_bin = max_bin_index - min_bin_index + 1
        bins = np.linspace(min_bin, max_bin, num_bin)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_width = bin_size / num_exps * BAR_WIDTH

        histograms = []
        p_values = []

        # Calculate data
        for i, d in enumerate(data):
            ep_returns = d["ep_returns"]

            hist, _ = np.histogram(ep_returns, bins=bins)
            histograms.append(hist)

            # Compute p-value
            if i == 0 or not args.include_p:
                p_values.append(None)
            else:
                _, p = ttest_rel(data[0]["ep_returns"], ep_returns)
                p_values.append(p)

        max_y_value = max([max(hist) for hist in histograms])

        # Plot data
        for index, d in enumerate(data):
            ep_returns = d["ep_returns"]
            is_baseline = d["is_baseline"]
            actor_mode = d["actor_mode"]
            label = d["label"]
            color = d["color"]

            hist = histograms[index]
            offset = (index - num_exps / 2 + 0.5) * bar_width
            ax.bar(bin_centers + offset, hist, width=bar_width, label=label, color=color, alpha=1)

            mean = np.mean(ep_returns)
            ax.axvline(mean, linestyle='--', color=color, linewidth=2)

            # Compute p-value
            p = p_values[index]
            if p is not None:
                y_pos = max_y_value * (1 - P_TEXT_Y_OFFSET - ((index-1) * P_TEXT_Y_INTER_OFFSET))
                p_text = get_p_text(p, [0.05, 0.01, 0.001])
                ax.text(mean + ((max_bin - min_bin) * P_TEXT_X_OFFSET), y_pos, f"p{p_text}", color=color, fontsize=10, va='top')

        # Customize plot
        ax.set_title("Episodic Returns")
        ax.set_xlabel("Episodic Return")
        ax.set_ylabel("Count")
        ax.set_xticks(bins)
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
