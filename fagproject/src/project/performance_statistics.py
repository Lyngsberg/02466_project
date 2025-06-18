# This code snippet addresses the need to:
# - Differentiate runs by dataset
# - Compute mean + 95% CI for last validation MSE per optimizer/dataset
# - Plot MSE curves for each dataset

import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from collections import defaultdict

# Initialize W&B API and fetch sweep
api = wandb.Api()
sweep = api.sweep("lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/1kl7m2zs")

# Nested dicts by dataset → optimizer → list of val series
pn_val_by_dataset_optimizer = defaultdict(lambda: defaultdict(list))
pn_train_by_dataset_optimizer = defaultdict(lambda: defaultdict(list))

# Also keep track of runtime per dataset-model
runtime_by_dataset_model = defaultdict(lambda: defaultdict(list))

# Dataset for each run
for run in sweep.runs:
    config = run.config
    model_modul = config.get("model_modul_name", [])
    if not isinstance(model_modul, list) or len(model_modul) < 2:
        continue

    model_type = model_modul[1]
    if model_type not in ["NN_model1", "Polynomial_Network"]:
        continue

    dataset = config.get("data_type", "unknown")
    optimizer = config.get("optimizer_name", "unknown")

    try:
        history = run.history(samples=1500)
        val_series = (history["validation_MSE"].dropna().reset_index(drop=True))

        # Store data
        if model_type == "Polynomial_Network":
            pn_val_by_dataset_optimizer[dataset][optimizer].append(val_series)

        if "_runtime" in history.columns:
            last_runtime = history["_runtime"].dropna().values[-1]
            runtime_by_dataset_model[dataset][model_type].append(last_runtime)
    except Exception as e:
        print(f"Error fetching run {run.name}: {e}")
        continue


def compute_mean_and_ci(histories):
    max_epochs = max(len(h) for h in histories)
    padded = np.full((len(histories), max_epochs), np.nan)
    for i, h in enumerate(histories):
        padded[i, :len(h)] = h.values
    means = np.nanmean(padded, axis=0)
    cis = t.ppf(0.975, df=len(histories)-1) * sem(padded, axis=0, nan_policy='omit')
    return np.arange(1, max_epochs + 1), means, means - cis, means + cis


def plot_histories(histories_dict, dataset, title, ylabel):
    plt.figure(figsize=(10, 6))
    for key, histories in histories_dict.items():
        if not histories:
            continue
        epochs, mean, lower, upper = compute_mean_and_ci(histories)
        plt.plot(epochs, mean, label=key)
        plt.fill_between(epochs, lower, upper, alpha=0.2)
    full_title = f"{title} ({dataset})"
    plt.title(full_title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"fagproject/src/project/performance_plots/{full_title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# For each dataset, print final val MSE CI and plot curves
for dataset in pn_val_by_dataset_optimizer:
    print(f"\n=== Dataset: {dataset} ===")
    val_histories = pn_val_by_dataset_optimizer[dataset]

    for optimizer, histories in val_histories.items():
        if len(histories) < 2:
            continue
        last_vals = [h.values[-1] for h in histories if len(h) > 0]
        mean_last = np.mean(last_vals)
        ci = t.ppf(0.975, df=len(last_vals)-1) * sem(last_vals)
        print(f"{optimizer}: MSE = {mean_last:.4f} (95% CI: [{mean_last - ci:.4f}, {mean_last + ci:.4f}])")

    # Plot training and validation for the dataset
    plot_histories(val_histories, dataset, "PN: Validation MSE per Epoch by Optimizer", "Validation MSE")

