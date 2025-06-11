import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from Deep_PN.PN_model_triang_deep import Polynomial_Network, PN_Neuron
from Deep_PN.NN_deep import General_NN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import sem, t

def train_model(model, n_epochs, data_type, optimizer_type, learning_rate=0.01, batch_size = 64, seed=42):
    # Data loading
    data_path = f"fagproject/data/{data_type}"
    x, y = torch.load(data_path)

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_features = X_test.shape[1]
    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)

    train_losses = []
    val_losses = []

    def closure():
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)
        return loss

    for epoch in range(n_epochs):
        model.train()

        if optimizer_type in ['Adam', 'SGD']:
            optimizer.zero_grad()
            predictions = model(X_train)
            train_loss = criterion(predictions, Y_train)
            train_loss.backward()
            optimizer.step()
        elif optimizer_type == 'LBFGS':
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test)
            val_loss = criterion(val_predictions, Y_test).item()


        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            epoch

    return model, train_losses, val_losses

def compute_final_ci(losses, confidence=0.95):
    """
    Compute mean and confidence interval for a list of scalar loss values.
    """
    losses = np.array(losses, dtype=np.float32)
    mean = np.mean(losses)
    n = len(losses)
    stderr = sem(losses)
    h = stderr * t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - h, mean + h

layers = [[1], [1,1], [1,1,1], [2], [2,1], [2,2], [2,1,1], [2,2,1], [2,2,2]]
n_epochs = 1000
lr = 0.0001
optimizer = 'Adam'

for layer in layers:
    PN_val_loss = []
    NN_val_loss = []
    for i in range(20):

        Polynomial_Net = Polynomial_Network([1], in_features=2)
        NeuralNet = General_NN(layer, in_features=2)

        data = "train_c_2.pkl"
        print("\nTraining Neural Network:")
        NeuralNet, train_losses_NN, val_losses_NN = train_model(
            NeuralNet, n_epochs, data, optimizer)
        
        print(val_losses_NN[-1])
        print(f"\nTraining Polynomial_Network (layers={layer}):")
        poly_network, train_losses_poly_network, val_losses_poly_network = train_model(
            Polynomial_Net, n_epochs, data, optimizer)
        PN_val_loss.append(val_losses_poly_network)
        NN_val_loss.append(val_losses_NN)

        print(val_losses_poly_network[-1])
    PN_mean, PN_lower, PN_upper = compute_final_ci(PN_val_loss[-1])
    NN_mean, NN_lower, NN_upper = compute_final_ci(NN_val_loss[-1])

    # Check CI overlap and whether NN is better
    overlap = not (PN_upper < NN_lower or NN_upper < PN_lower)
    nn_better = NN_upper < PN_lower  # NN upper bound < PN lower bound

    if nn_better or not overlap:
        print(f"\nLayer configuration: {layer}")
        print(f"PN CI: ({PN_lower:.4f}, {PN_upper:.4f})")
        print(f"NN CI: ({NN_lower:.4f}, {NN_upper:.4f})")
        if nn_better:
            print("NN is statistically better than PN.")
        elif not overlap:
            print("Confidence intervals do not overlap.")
