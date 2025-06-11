import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Initialize W&B API and fetch sweep
api = wandb.Api()
# The sweep ID has to be changed when a new sweep is done
sweep = api.sweep("lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/x8oyk7sp")

# Dictionaries to store per-model histories
train_histories = {}
val_histories = {}
runtime_by_model = {}

# PN-specific by optimizer
pn_train_by_optimizer = {}
pn_val_by_optimizer = {}

# Collect MSE histories
for run in sweep.runs:
    config = run.config
    model_modul = config.get("model_modul_name", [])
    if not isinstance(model_modul, list) or len(model_modul) < 2:
        continue

    model_type = model_modul[1]
    if model_type not in ["NN_model1", "Polynomial_Network"]:
        continue

    # 🧠 DIFFERENTIATE NN BY LAYERS
    if model_type == "NN_model1":
        layers = config.get("layers", [])
        model_key = f"{model_type}: {layers}"
    else:
        model_key = model_type

    try:
        history = run.history(samples=1500)
        val_series = np.log(history["validation_MSE"].dropna().reset_index(drop=True))
        train_series = np.log(history["train_MSE"].dropna().reset_index(drop=True))

        # Runtime tracking
        if "_runtime" in history.columns:
            last_runtime = history["_runtime"].dropna().values[-1]
            runtime_by_model.setdefault(model_key, []).append(last_runtime)

        # Store PN optimizer info separately
        if model_type == "Polynomial_Network":
            optimizer = config.get("optimizer_name", "unknown")
            pn_train_by_optimizer.setdefault(optimizer, []).append(train_series)
            pn_val_by_optimizer.setdefault(optimizer, []).append(val_series)

        # General storage
        train_histories.setdefault(model_key, []).append(train_series)
        val_histories.setdefault(model_key, []).append(val_series)

    except Exception as e:
        print(f"Error fetching history for run {run.name}: {e}")
        continue


def compute_mean_and_ci(histories, min_runs=1):
    max_epochs = max(len(h) for h in histories)
    padded = np.full((len(histories), max_epochs), np.nan)

    for i, h in enumerate(histories):
        padded[i, :len(h)] = h.values

    # Count how many runs have valid data per epoch
    valid_counts = np.sum(~np.isnan(padded), axis=0)

    # Compute mean and CI ignoring NaNs
    means = np.nanmean(padded, axis=0)
    ci = t.ppf(0.975, df=len(histories)-1) * sem(padded, axis=0, nan_policy='omit')

    # Mask epochs with fewer than min_runs runs contributing
    valid_mask = valid_counts >= min_runs

    epochs = np.arange(1, max_epochs + 1)

    return epochs[valid_mask], means[valid_mask], means[valid_mask] - ci[valid_mask], means[valid_mask] + ci[valid_mask]



# Plotting function
def plot_histories(histories_dict, title, ylabel):
    plt.figure(figsize=(10, 6))
    for model_type, histories in histories_dict.items():
        print(f"{title} | {model_type}: {len(histories)} runs")
        if not histories:
            continue
        epochs, mean, lower, upper = compute_mean_and_ci(histories)
        plt.plot(epochs, mean, label=model_type)
        plt.fill_between(epochs, lower, upper, alpha=0.2)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fagproject/src/project/performance_plots/{title.replace(' ', '_').replace(':', '')}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_runtime_bar(runtime_dict, title="Training Runtime per Model"):
    labels = []
    means = []
    lowers = []
    uppers = []

    for model, times in runtime_dict.items():
        if not times:
            continue
        times = np.array(times)
        mean = np.mean(times)
        ci = t.ppf(0.975, df=len(times)-1) * sem(times)
        labels.append(model)
        means.append(mean)
        lowers.append(mean - ci)
        uppers.append(mean + ci)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
            capsize=8, alpha=0.7, color="skyblue")
    plt.xticks(x, labels)
    plt.ylabel("Runtime (seconds)")
    plt.title(title)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"fagproject/src/project/performance_plots/{title.replace(' ', '_').replace(':', '')}.png", dpi=300, bbox_inches='tight')
    plt.close()


# Plot both train and validation MSE
#plot_histories(train_histories, "Mean Train MSE per Epoch with 95% CI", "Train MSE")
plot_histories(val_histories, "Mean Validation MSE per Epoch with 95% CI", "Validation MSE")

# Plot only PN: optimizer comparisons
plot_histories(pn_train_by_optimizer, "PN: Mean Train MSE per Epoch by Optimizer", "Train MSE")
#plot_histories(pn_val_by_optimizer, "PN: Mean Validation MSE per Epoch by Optimizer", "Validation MSE")

plot_runtime_bar(runtime_by_model, "Mean Training Runtime per Model (with 95% CI)")