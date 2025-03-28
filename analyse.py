import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# === Main Execution ===
directory_path = "out_seaquest/tensorboard/seaquest_softmax_blender_logic_lr_0.00025_llr_0.00025_blr_0.00025_gamma_0.99_bentcoef_0.01_numenvs_50_steps_128__0"

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

def process_and_plot_metric(directory_path, target_tag, filename, compute_stats=False):
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

    # If enabled, calculate and print statistics (mean and std)
    if compute_stats:
        mean_val = metric_data["value"].mean()
        std_val = metric_data["value"].std()
        print(f"{target_tag} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")

    # Downsample for plotting and convert step values to millions (M)
    metric_data["step"] = metric_data["step"] / 1e6
    metric_data_plot = metric_data.iloc[::100, :].copy()

    plt.figure(figsize=(13, 7))
    # Plot the original metric data
    plt.plot(metric_data_plot["step"], metric_data_plot["value"],
             alpha=0.6, linewidth=0.8)

    if compute_stats:
        # Plot a horizontal line for the overall mean with std value in the label
        plt.axhline(y=mean_val, color='b', linestyle='--',
                    label=f'Mean: {mean_val:.4f}, Std: {std_val:.4f}')

    # Customize x-axis ticks: 0M, 10M, 20M, ..., 60M
    xticks = list(range(0, 61, 10))
    xtick_labels = [f"{i}M" for i in xticks]
    plt.locator_params(axis='y', nbins=15)
    plt.xticks(xticks, xtick_labels)

    plt.xlabel("Training Steps (in Millions)")
    plt.ylabel("Metric Value")
    plt.title(target_tag, loc="left", fontsize=12)
    if compute_stats:
        plt.legend()
    plt.show()


process_and_plot_metric(directory_path, "charts/episodic_return", "episodic_return.csv", compute_stats=True)
process_and_plot_metric(directory_path, "charts/episodic_game_return", "episodic_game_return.csv", compute_stats=True)
process_and_plot_metric(directory_path, "charts/episodic_length", "episodic_length.csv", compute_stats=True)
process_and_plot_metric(directory_path, "losses/value_loss", "value_loss.csv")
process_and_plot_metric(directory_path, "losses/policy_loss", "policy_loss.csv")
process_and_plot_metric(directory_path, "losses/entropy", "entropy.csv")
process_and_plot_metric(directory_path, "losses/approx_kl", "approx_kl.csv")
process_and_plot_metric(directory_path, "losses/clipfrac", "clipfrac.csv")
process_and_plot_metric(directory_path, "losses/explained_variance", "explained_variance.csv")
process_and_plot_metric(directory_path, "charts/learning_rate", "learning_rate.csv")
process_and_plot_metric(directory_path, "losses/blend_entropy", "blend_entropy.csv")
process_and_plot_metric(directory_path, "charts/SPS", "SPS.csv")
