import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Init W&B API
api = wandb.Api()

# for the one with less data split 1/99 split so 99% test and 1% train
sweep_ids = {
    "Sweep 1": "lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/ngrp9rz2",
    "Sweep 2": "lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/keo88np6"
}
# for the one with normal split in data 80/20 split
# sweep_ids = {
#     "Sweep 1": "lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/9r7gcrrp",
#     "Sweep 2": "lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/4fz856to",
# }

# Containers
train_histories = {name: {} for name in sweep_ids}
val_histories = {name: {} for name in sweep_ids}
runtime_by_model = {name: {} for name in sweep_ids}

# Helper to collect data
def collect_data(sweep, sweep_name):
    for run in sweep.runs:
        config = run.config
        model_modul = config.get("model_modul_name", [])
        if not isinstance(model_modul, list) or len(model_modul) < 2:
            print(f"[{sweep_name}] Skipping run {run.name}: invalid model_modul_name {model_modul}")
            continue

        model_type = model_modul[1]
        if model_type not in ["General_NN", "Polynomial_Network"]:
            print(f"[{sweep_name}] Skipping run {run.name}: model type {model_type} not in target list")
            continue

        if model_type == "General_NN":
            layers = config.get("layers", [])
            model_key = f"{model_type}: {layers}"
        else:
            layers = config.get("layers", [])
            model_key = f"{model_type}: {layers}"

        try:
            history = run.history(samples=10000)
            val_series = history["validation_MSE"].dropna().reset_index(drop=True)
            train_series = history["train_MSE"].dropna().reset_index(drop=True)

            # Runtime
            if "_runtime" in history.columns:
                last_runtime = history["_runtime"].dropna().values[-1]
                runtime_by_model[sweep_name].setdefault(model_key, []).append(last_runtime)

            # Store histories
            train_histories[sweep_name].setdefault(model_key, []).append(train_series)
            val_histories[sweep_name].setdefault(model_key, []).append(val_series)

            print(f"[{sweep_name}] Processed run {run.name} as {model_key}")

        except Exception as e:
            print(f"[{sweep_name}] Error in run {run.name}: {e}")

# Load and process both sweeps
for name, sweep_id in sweep_ids.items():
    sweep = api.sweep(sweep_id)
    collect_data(sweep, name)

# Compute mean + CI
def compute_mean_and_ci(histories):
    max_epochs = max(len(h) for h in histories)
    padded = np.full((len(histories), max_epochs), np.nan)
    for i, h in enumerate(histories):
        padded[i, :len(h)] = h.values

    means = np.nanmean(padded, axis=0)
    ci = t.ppf(0.975, df=len(histories)-1) * sem(padded, axis=0, nan_policy='omit')
    epochs = np.arange(1, max_epochs + 1)
    return epochs, means, means - ci, means + ci

# Plot histories comparison
def plot_histories_comparison(histories_dict, title, ylabel):
    plt.figure(figsize=(10, 6))
    for sweep_name, model_dict in histories_dict.items():
        for model_type, histories in model_dict.items():
            if not histories:
                continue
            epochs, mean, lower, upper = compute_mean_and_ci(histories)
            label = f"{sweep_name} | {model_type}"
            plt.plot(epochs, mean, label=label)
            plt.fill_between(epochs, lower, upper, alpha=0.2)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Zoom in y-axis
    #for zoomed in for better visibility of the one with normal split
    #plt.ylim(16.5, 19)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').replace(':', '')}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Runtime bar plot
def plot_runtime_bar_comparison(runtime_dict, title="Mean Training Runtime per Model"):
    plt.figure(figsize=(10, 6))
    all_labels = []
    all_means = []
    all_errs = []

    for sweep_name, model_runtimes in runtime_dict.items():
        for model, times in model_runtimes.items():
            mean = np.mean(times)
            ci = t.ppf(0.975, df=len(times)-1) * sem(times)
            all_labels.append(f"{sweep_name} | {model}")
            all_means.append(mean)
            all_errs.append(ci)

    x = np.arange(len(all_labels))
    plt.bar(x, all_means, yerr=all_errs, capsize=5, alpha=0.7, color="skyblue")
    plt.xticks(x, all_labels, rotation=45, ha='right')
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

# === Call plotting ===
plot_histories_comparison(train_histories, "Train MSE Comparison", "Train MSE")
plot_histories_comparison(val_histories, "Validation MSE Comparison", "Validation MSE")
plot_runtime_bar_comparison(runtime_by_model)
