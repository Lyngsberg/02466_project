import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from PN_models import Polynomial_Network
from sklearn.model_selection import train_test_split
from plot import plot_sampled_function_vs_polynomial_estimate
from scipy import stats
from sympy import Symbol, Mul, Pow
from collections import defaultdict
from Deep_PN.PN_model_triang_deep import Polynomial_Network, PN_Neuron
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import pickle
import os


def plot_coefficients_with_ci(summary, save_path):
    keys = list(summary.keys())
    means = [summary[k]["mean"] for k in keys]
    cis = [summary[k]["ci"] for k in keys]

    # Convert sympy keys to strings for labeling
    labels = [str(k) for k in keys]

    # Reverse the order
    keys = keys[::-1]
    means = means[::-1]
    cis = cis[::-1]
    labels = labels[::-1]

    x = np.arange(len(keys))

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means, yerr=cis, fmt='o', capsize=5, color='tab:blue')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Coefficient Value")
    plt.title("Mean and Confidence Interval of Coefficients")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

folder_path = "fagproject/src/project/optimizer_data"
output_dir = "fagproject/src/project/interpretability_plots"

loaded_objects = {}

for filename in os.listdir(folder_path):
    if filename.endswith(".pkl"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        var_name = os.path.splitext(filename)[0]
        loaded_objects[var_name] = obj

for obj_name, summary_dict in loaded_objects.items():
    save_path = os.path.join(output_dir, f"{obj_name}.png")
    plot_coefficients_with_ci(summary_dict, save_path)

print(loaded_objects)