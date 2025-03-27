import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from ocatari.core import OCAtari

directory_path = "out_freeway/tensorboard/freeway_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0"

def run_random_baseline(env_name="ALE/Freeway-v5", num_episodes=100):
    env = OCAtari(
        env_name=env_name,
        mode="ram",
        obs_mode="ori",
        render_mode="rgb_array",
        render_oc_overlay=False,
    )

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()

    avg_return = np.mean(episode_returns)
    avg_length = np.mean(episode_lengths)

    print(f"\nRandom Baseline over {num_episodes} episodes:")
    print(f"  Average return: {avg_return:.2f}")
    print(f"  Average episode length: {avg_length:.2f}")

    return avg_return, avg_length


def extract_tag_values_from_events(file_path, target_tag):
    data = []
    for event in tf.compat.v1.train.summary_iterator(file_path):
        for value in event.summary.value:
            if value.HasField('simple_value') and value.tag == target_tag:
                data.append({"step": event.step, "value": value.simple_value})

    df = pd.DataFrame(data)
    if not df.empty:
        df["step"] = df["step"].astype(np.int32)
        df["value"] = df["value"].astype(np.float32)
    return df

def moving_average(series, window_size):
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

def process_and_plot_length_return(directory_path, target_tag, filename, compute_stats=False, baseline_value=None):
    tfevents_files = glob.glob(os.path.join(directory_path, "*.tfevents.*"))
    metric_data = pd.DataFrame()

    # Extract data from all matching tfevents files
    for file_path in tfevents_files:
        print(f"Processing file: {file_path}")
        data = extract_tag_values_from_events(file_path, target_tag)
        if not data.empty:
            metric_data = pd.concat([metric_data, data], ignore_index=True)

    if metric_data.empty:
        print(f"No data found for {target_tag}")
        return

    # Save the extracted data as CSV
    metric_data.to_csv(filename, index=False)
    print(f"Saved extracted {target_tag} data to {filename}")

    # Sort data by step to ensure proper plotting order
    metric_data.sort_values("step", inplace=True)

    # Compute stats
    legend_labels = []
    if compute_stats:
        mean_val = metric_data["value"].mean()
        std_val = metric_data["value"].std()
        print(f"{target_tag} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        legend_labels.append(f'Mean: {mean_val:.2f} ± {std_val:.2f}')

    # Downsample for plotting raw values
    metric_data["step_millions"] = metric_data["step"] / 1e6
    metric_data_plot = metric_data.iloc[::100, :].copy()

    # Calculate moving average
    metric_data["smoothed_value"] = moving_average(metric_data["value"], window_size=1000)

    plt.figure(figsize=(13, 7))

    # Plot raw data (faint)
    plt.plot(metric_data_plot["step_millions"], metric_data_plot["value"],
             alpha=0.22, linewidth=0.5, label="Raw data")

    # Plot moving average (bold black)
    plt.plot(metric_data["step_millions"], metric_data["smoothed_value"],
             linewidth=1.5, color='steelblue', label="Moving Avg (window=1000)")

    # Plot baseline
    if baseline_value is not None:
        plt.axhline(y=baseline_value, color='brown', linestyle='--', linewidth=2, label=f'Random: {baseline_value:.2f}')

    # Customize x-axis
    xticks = list(range(0, 61, 10))
    xtick_labels = [f"{i}M" for i in xticks]
    plt.xticks(xticks, xtick_labels)
    plt.locator_params(axis='y', nbins=15)

    plt.xlabel("Training Steps (in Millions)")
    plt.ylabel("Metric Value")
    plt.title(target_tag, loc="left", fontsize=12)

    # Combine all labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if compute_stats:
        labels.append(legend_labels[0])
        handles.append(plt.Line2D([], [], color='none'))  # Invisible entry

    plt.legend(handles, labels)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def process_and_plot_losses(directory_path, target_tag, filename):
    tfevents_files = glob.glob(os.path.join(directory_path, "*.tfevents.*"))
    metric_data = pd.DataFrame()

    # Extract data from all matching tfevents files
    for file_path in tfevents_files:
        print(f"Processing file: {file_path}")
        data = extract_tag_values_from_events(file_path, target_tag)
        if not data.empty:
            metric_data = pd.concat([metric_data, data], ignore_index=True)

    if metric_data.empty:
        print(f"No data found for {target_tag}")
        return

    # Save the extracted data as CSV
    metric_data.to_csv(filename, index=False)
    print(f"Saved extracted {target_tag} data to {filename}")

    # Sort data by step to ensure proper plotting order
    metric_data.sort_values("step", inplace=True)
    
    # Downsample for plotting and convert step values to millions (M)
    metric_data["step"] = metric_data["step"] / 1e6
    metric_data_plot = metric_data.iloc[::100, :].copy()

    plt.figure(figsize=(13, 7))
    # Plot the original metric data
    plt.plot(metric_data_plot["step"], metric_data_plot["value"],
             alpha=0.6, linewidth=0.8)

    # Customize x-axis ticks: 0M, 10M, 20M, ..., 60M
    xticks = list(range(0, 61, 10))
    xtick_labels = [f"{i}M" for i in xticks]
    plt.locator_params(axis='y', nbins=15)
    plt.xticks(xticks, xtick_labels)

    plt.xlabel("Training Steps (in Millions)")
    plt.ylabel("Metric Value")
    plt.title(target_tag, loc="left", fontsize=12)
    plt.show()



# === Example Baseline Values ===
baseline_return, baseline_length = run_random_baseline()

process_and_plot_length_return(directory_path, "charts/episodic_return", "episodic_return.csv", compute_stats=True, baseline_value=baseline_return)
process_and_plot_length_return(directory_path, "charts/episodic_length", "episodic_length.csv", compute_stats=True, baseline_value=baseline_length)
process_and_plot_losses(directory_path, "losses/value_loss", "value_loss.csv")
process_and_plot_losses(directory_path, "losses/policy_loss", "policy_loss.csv")
process_and_plot_losses(directory_path, "losses/entropy", "entropy.csv")
process_and_plot_losses(directory_path, "losses/approx_kl", "approx_kl.csv")
process_and_plot_losses(directory_path, "losses/clipfrac", "clipfrac.csv")
process_and_plot_losses(directory_path, "losses/explained_variance", "explained_variance.csv")
process_and_plot_losses(directory_path, "charts/learning_rate", "learning_rate.csv")
process_and_plot_losses(directory_path, "losses/blend_entropy", "blend_entropy.csv")
process_and_plot_losses(directory_path, "charts/SPS", "SPS.csv")