import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# Initialize W&B API and fetch sweep
api = wandb.Api()
# The sweep ID has to be changed when a new sweep is done
sweep = api.sweep("lyngsberg-danmarks-tekniske-universitet-dtu/Fagprojekt/fov7ybpm")

# Dictionaries to store per-model histories
train_histories = {
    "NN_model1": [],
    "Polynomial_Network": []
}
val_histories = {
    "NN_model1": [],
    "Polynomial_Network": []
}

# Collect MSE histories
break_point = 0
for run in sweep.runs:
    # if break_point == 3:
    #     break
    # break_point += 1
    config = run.config
    model_modul = config.get("model_modul_name", [])

    if not isinstance(model_modul, list) or len(model_modul) < 2:
        continue

    model_type = model_modul[1]
    if model_type not in train_histories:
        continue

    try:
        history = run.history()
        print(history)
        val_series = history["validation_MSE"].dropna().reset_index(drop=True)
        train_series = history["train_MSE"].dropna().reset_index(drop=True)

        val_histories[model_type].append(val_series)
        train_histories[model_type].append(train_series)

    except Exception as e:
        print(f"Error fetching history for run {run.name}: {e}")
        continue


def compute_mean_and_ci(histories):
    """
    Compute mean and 95% confidence interval for MSE histories.
    """
    max_epochs = max(len(h) for h in histories)
    padded = np.full((len(histories), max_epochs), np.nan)

    for i, h in enumerate(histories):
        padded[i, :len(h)] = h.values

    means = np.nanmean(padded, axis=0)
    ci = t.ppf(0.975, df=len(histories)-1) * sem(padded, axis=0, nan_policy='omit')
    return np.arange(1, max_epochs+1), means, means - ci, means + ci


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
    plt.show()


# Plot both train and validation MSE
plot_histories(train_histories, "Mean Train MSE per Epoch with 95% CI", "Train MSE")
plot_histories(val_histories, "Mean Validation MSE per Epoch with 95% CI", "Validation MSE")
