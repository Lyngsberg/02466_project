from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch

def plot_sampled_function_vs_polynomial_estimate(x_test, y_test, output, polynomial=None):   
    X = x_test
    Z = y_test
    
    # Reshape X and Y to match Z's shape
    Y = X[:, 1].reshape(Z.shape)
    X = X[:, 0].reshape(Z.shape)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot Real Data with Polynomial Surface
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(X, Y, Z, label='Real Data', c='b')
    
    if polynomial is not None:
        # Generate polynomial surface
        f_poly = sp.lambdify((sp.Symbol('x'), sp.Symbol('y')), polynomial, 'numpy')
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = np.linspace(Y.min(), Y.max(), 100)
        X_poly, Y_poly = np.meshgrid(x_vals, y_vals)
        Z_poly = f_poly(X_poly, Y_poly)  # Evaluate polynomial
        
        # Plot polynomial surface
        ax1.plot_surface(X_poly, Y_poly, Z_poly, color='g', alpha=0.4, label='Polynomial Surface')
    
    ax1.set_title('Real Data with Polynomial Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Plot Model Output
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.scatter(X, Y, output, label='Model Output', c='r')
    ax2.set_title('Model Output')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.show()

def plot_weights(weight_history):
    weight_history = np.array(weight_history)  # shape: (epochs, n_neurons, 3, 3)

    n_neurons = weight_history.shape[1]
    for neuron_idx in range(n_neurons):
        plt.figure()
        # Plot sum of symmetric off-diagonal weights for each epoch
        plt.plot(
            [weight_history[epoch, neuron_idx, 0, 1] + weight_history[epoch, neuron_idx, 1, 0] for epoch in range(weight_history.shape[0])],
            label='W[0,1] + W[1,0]'
        )
        plt.plot(
            [weight_history[epoch, neuron_idx, 0, 2] + weight_history[epoch, neuron_idx, 2, 0] for epoch in range(weight_history.shape[0])],
            label='W[0,2] + W[2,0]'
        )
        plt.plot(
            [weight_history[epoch, neuron_idx, 1, 2] + weight_history[epoch, neuron_idx, 2, 1] for epoch in range(weight_history.shape[0])],
            label='W[1,2] + W[2,1]'
        )
        # Plot diagonal weights
        plt.plot(
            [weight_history[epoch, neuron_idx, 0, 0] for epoch in range(weight_history.shape[0])],
            label='W[0,0]'
        )
        plt.plot(
            [weight_history[epoch, neuron_idx, 1, 1] for epoch in range(weight_history.shape[0])],
            label='W[1,1]'
        )
        plt.plot(
            [weight_history[epoch, neuron_idx, 2, 2] for epoch in range(weight_history.shape[0])],
            label='W[2,2]'
        )
        plt.xlabel("Epoch")
        plt.ylabel("Weight value")
        plt.title(f"Weight evolution for neuron {neuron_idx}")
        plt.legend()
        plt.show()

def plot_weights_mean(all_weight_histories):
    """
    all_weight_histories: list of weight_history arrays, each of shape (epochs, n_neurons, 3, 3)
    """
    all_weight_histories = np.array(all_weight_histories)  # shape: (n_runs, epochs, n_neurons, 3, 3)
    n_runs, n_epochs, n_neurons, _, _ = all_weight_histories.shape

    def plot_with_ci(data, label):
        # data shape: (n_runs, n_epochs)
        mean = np.mean(data, axis=0)  # mean over runs, shape: (n_epochs,)
        stderr = np.std(data, axis=0, ddof=1) / np.sqrt(n_runs)  # std error over runs, shape: (n_epochs,)
        ci = 1.96 * stderr  # 95% confidence interval
        plt.plot(mean, label=label)
        plt.fill_between(np.arange(n_epochs), mean - ci, mean + ci, alpha=0.2)

    for neuron_idx in range(n_neurons):
        plt.figure()
        # W[0,1] + W[1,0]
        data = all_weight_histories[:, :, neuron_idx, 0, 1] + all_weight_histories[:, :, neuron_idx, 1, 0]  # shape: (n_runs, n_epochs)
        plot_with_ci(data, 'W[0,1] + W[1,0]')
        # W[0,2] + W[2,0]
        data = all_weight_histories[:, :, neuron_idx, 0, 2] + all_weight_histories[:, :, neuron_idx, 2, 0]
        plot_with_ci(data, 'W[0,2] + W[2,0]')
        # W[1,2] + W[2,1]
        data = all_weight_histories[:, :, neuron_idx, 1, 2] + all_weight_histories[:, :, neuron_idx, 2, 1]
        plot_with_ci(data, 'W[1,2] + W[2,1]')
        # Diagonal weights
        for i in range(3):
            data = all_weight_histories[:, :, neuron_idx, i, i]
            plot_with_ci(data, f'W[{i},{i}]')
        plt.xlabel("Epoch")
        plt.ylabel("Weight value")
        plt.title(f"Weight evolution for neuron {neuron_idx} (mean ± 95% CI)")
        plt.legend()
        plt.show()

def plot_weights_mean_compare(all_weight_histories_triang, all_weight_histories_notriang):
    all_weight_histories_triang = np.array(all_weight_histories_triang)
    all_weight_histories_notriang = np.array(all_weight_histories_notriang)

    # Remove extra singleton dimensions
    all_weight_histories_triang = np.squeeze(all_weight_histories_triang)
    all_weight_histories_notriang = np.squeeze(all_weight_histories_notriang)

    # Add neuron axis if missing (for single-neuron case)
    if all_weight_histories_triang.ndim == 4:
        all_weight_histories_triang = all_weight_histories_triang[:, :, np.newaxis, :, :]
        all_weight_histories_notriang = all_weight_histories_notriang[:, :, np.newaxis, :, :]

    n_runs, n_epochs, n_neurons, _, _ = all_weight_histories_triang.shape

    def plot_with_ci(data, label, color, linestyle='-'):
        mean = np.mean(data, axis=0)
        stderr = np.std(data, axis=0, ddof=1) / np.sqrt(n_runs)
        ci = 1.96 * stderr
        plt.plot(mean, label=label, color=color, linestyle=linestyle)
        plt.fill_between(np.arange(n_epochs), mean - ci, mean + ci, alpha=0.2, color=color)

    for neuron_idx in range(n_neurons):
        plt.figure(figsize=(10, 7))
        pairs = [((0,1),(1,0)), ((0,2),(2,0)), ((1,2),(2,1))]
        colors = ['blue', 'green', 'purple']
        for idx, ((i,j),(j2,i2)) in enumerate(pairs):
            data_triang = all_weight_histories_triang[:, :, neuron_idx, i, j] + all_weight_histories_triang[:, :, neuron_idx, j2, i2]
            data_notriang = all_weight_histories_notriang[:, :, neuron_idx, i, j] + all_weight_histories_notriang[:, :, neuron_idx, j2, i2]
            plot_with_ci(data_triang, f'Triangulized W[{i},{j}]+W[{j},{i}]', colors[idx], '-')
            plot_with_ci(data_notriang, f'Not Triangulized W[{i},{j}]+W[{j},{i}]', colors[idx], '--')
        diag_colors = ['black', 'red', 'orange']
        for i in range(3):
            data_triang = all_weight_histories_triang[:, :, neuron_idx, i, i]
            data_notriang = all_weight_histories_notriang[:, :, neuron_idx, i, i]
            plot_with_ci(data_triang, f'Triangulized W[{i},{i}]', diag_colors[i], '-')
            plot_with_ci(data_notriang, f'Not Triangulized W[{i},{i}]', diag_colors[i], '--')
        plt.xlabel("Epoch")
        plt.ylabel("Weight value")
        plt.title(f"Weight evolution for neuron {neuron_idx} (mean ± 95% CI)")
        plt.legend()
        plt.tight_layout()
        plt.show()