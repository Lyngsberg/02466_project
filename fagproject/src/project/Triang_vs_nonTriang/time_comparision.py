import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import t, sem
from PN_models_for_triang_vs_nontriang import Polynomial_Network, PolynomialNet, PN_Neuron
from sklearn.model_selection import train_test_split
from plot import plot_sampled_function_vs_polynomial_estimate, plot_weights, plot_weights_mean, plot_weights_mean_compare


# W12_W21 = []
# W13_W31 = []
# W23_W32 = []


def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs=1000, learning_rate=0.01, path=None,
                tol=1e-5, patience=20):
    start_time = time.perf_counter()  # Start timer

    weight_history = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    no_improve_counter = 0

    for epoch in range(n_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        train_loss = criterion(predictions, Y_train)
        train_loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            train_loss = criterion(model(X_train), Y_train)
            val_loss = criterion(model(X_val), Y_val)


        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Validation Loss: {val_loss.item():.6f}")

        # Early stopping check
        if abs(best_val_loss - val_loss.item()) < tol:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"Converged at epoch {epoch}.")
                break
        else:
            best_val_loss = val_loss.item()
            no_improve_counter = 0

    elapsed_time = time.perf_counter() - start_time

    return model, train_losses, val_losses, elapsed_time

def plot_runtime_comparison(triang_times, notriang_times):
    triang_times = np.array(triang_times)
    notriang_times = np.array(notriang_times)

    mean_triang = np.mean(triang_times)
    mean_notriang = np.mean(notriang_times)

    ci_triang = t.ppf(0.975, df=len(triang_times)-1) * sem(triang_times)
    ci_notriang = t.ppf(0.975, df=len(notriang_times)-1) * sem(notriang_times)

    labels = ['Triangulized', 'Non-triangulized']
    means = [mean_triang, mean_notriang]
    errors = [[ci_triang], [ci_notriang]]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, means, yerr=np.array(errors).T, capsize=8, color=['skyblue', 'lightcoral'], alpha=0.8)
    plt.ylabel("Training Time (seconds)")
    plt.title("Average Training Time: Triangulized vs Non-triangulized")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

path_quadratic_1 = 'fagproject/data/train_q_1.pkl'
path_quadratic_2 = 'fagproject/data/train_q:2.pkl'
path_quadratic_with_noise_1 = 'fagproject/data/train_q_n_1.pkl'
path_quadratic_with_noise_2 = 'fagproject/data/train_q_n_2.pkl'

path_cubic_1 = 'fagproject/data/train_c_1.pkl'
path_cubic_2 = 'fagproject/data/train_c:2.pkl'
path_cubic_with_noise_1 = 'fagproject/data/train_c_n_1.pkl'
path_cubic_with_noise_2 = 'fagproject/data/train_c_n_2.pkl'

path_smooth_1 = 'fagproject/data/train_s_1.pkl'
path_smooth_2 = 'fagproject/data/train_s:2.pkl'
path_smooth_with_noise_1 = 'fagproject/data/train_s_n_1.pkl'
path_smooth_with_noise_2 = 'fagproject/data/train_s_n_2.pkl'


if __name__ == "__main__":
    all_weight_histories_triang = []
    all_weight_histories_notriang = []
    triang_times = []
    notriang_times = []

    for i in range(60):
        random_seed = i
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        path = 'fagproject/data/train_s_2.pkl'
        X, y = torch.load(path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        poly_network_triangulized = Polynomial_Network(n_neurons=1)
        poly_network = PolynomialNet()

        n_epochs = 2000
        lr = 0.01

        # For triangulized
        _, _, _, time_triang = train_model(poly_network_triangulized, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)
        triang_times.append(time_triang)

        # For non-triangulized

        _, _, _, time_notriang = train_model(poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)
        notriang_times.append(time_notriang)

    plot_runtime_comparison(triang_times, notriang_times)


