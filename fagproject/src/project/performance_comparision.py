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

        if epoch % 200 == 0:
            print(epoch)

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

layers = [[1], [1,1], [1,1,1], [2], [2,1], [2,2], [2,1,1], [2,2,1], [2,2,2], [2,2,2,2], [3,2,1,3]]
data_types = ["train_c_2.pkl", "train_c_4.pkl", "train_q_2.pkl", "train_q_4.pkl", "train_s_2.pkl", "train_s_4.pkl"]
n_epochs = 3000
lr = 0.0001
optimizer = 'Adam'

results_summary = {}
matching_performance = []

for data_type in data_types:
    print(f"\n=== Data Type: {data_type} ===")
    found = False
    for layer in layers:
        PN_val_loss = []
        NN_val_loss = []

        for i in range(20):
            print(f"Data type: {data_type}, layer: {layer}, training iteration: {i}")
            Polynomial_Net = Polynomial_Network([1], in_features=2)
            NeuralNet = General_NN(layer, in_features=2)

            # Train NN
            NeuralNet, _, val_losses_NN = train_model(
                NeuralNet, n_epochs, data_type, optimizer, learning_rate=lr)
            NN_val_loss.append(val_losses_NN[-1])

            # Train PN
            Polynomial_Net, _, val_losses_PN = train_model(
                Polynomial_Net, n_epochs, data_type, optimizer, learning_rate=lr)
            PN_val_loss.append(val_losses_PN[-1])

        # Compute mean and CI for both
        PN_mean, PN_lower, PN_upper = compute_final_ci(PN_val_loss)
        NN_mean, NN_lower, NN_upper = compute_final_ci(NN_val_loss)

        # Check conditions
        overlap = not (PN_upper < NN_lower or NN_upper < PN_lower)
        nn_better = NN_upper < PN_lower

        if nn_better or overlap:
            print(f"\n✅ Match Found for {data_type}")
            print(f"Layer configuration: {layer}")
            print(f"PN Mean ± CI: {PN_mean:.4f} ({PN_lower:.4f}, {PN_upper:.4f})")
            print(f"NN Mean ± CI: {NN_mean:.4f} ({NN_lower:.4f}, {NN_upper:.4f})")

            results_summary[data_type] = {
                'layer': layer,
                'reason': 'NN better' if nn_better else 'Overlap',
                'PN_mean': PN_mean,
                'PN_CI': (PN_lower, PN_upper),
                'NN_mean': NN_mean,
                'NN_CI': (NN_lower, NN_upper)
            }

            matching_performance.append((data_type, layer, PN_lower, PN_upper, NN_lower, NN_upper))
            found = True
            break  # Exit loop for this data_type

    if not found:
        print(f"❌ No overlap or NN improvement found for {data_type}")
print(matching_performance)