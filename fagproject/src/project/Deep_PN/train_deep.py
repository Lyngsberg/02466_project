import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from PN_model_triang_deep import Polynomial_Network, PN_Neuron
from NN_deep import General_NN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Now training...")


def find_optimal_lambda(n_epochs, layers, lr=0.01, path=None, optimizer_type='Adam'):
    best_val_loss = float('inf')
    index = 0
    l2_lambda_list = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    for i, l2 in enumerate(l2_lambda_list):
        data = pd.read_csv(path).dropna()
        data.replace({"Yes": 1, "No": 0}, inplace=True)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values.reshape(-1, 1)

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3)

        scaler_X = StandardScaler()
        X_train_np = scaler_X.fit_transform(X_train_np)
        X_test_np = scaler_X.transform(X_test_np)

        scaler_y = StandardScaler()
        scaler_y.fit(y_train_np)
        y_train_np = scaler_y.transform(y_train_np)
        y_test_np = scaler_y.transform(y_test_np)

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        y_test = torch.tensor(y_test_np, dtype=torch.float32)

        print(f"Training with l2_lambda: {l2}")
        model = Polynomial_Network(layers, in_features=X_train.shape[1])
        poly_network, train_losses_poly_network, val_losses_poly_network = train_model(
            model=model, X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test,
            n_epochs=n_epochs, learning_rate=lr, path=None, optimizer_type=optimizer_type, l2_lambda=l2, scaler_y=scaler_y)

        criterion = nn.MSELoss()
        with torch.no_grad():
            test_loss = criterion(poly_network(X_train), y_train)

        test_loss_unscaled = test_loss * (scaler_y.scale_[0] ** 2)
        print(f"Test Loss (Polynomial_Network, layers={layers}): {test_loss_unscaled.item():.4f}")

        if val_losses_poly_network[-1] < best_val_loss:
            best_val_loss = val_losses_poly_network[-1]
            index = i
        print(f"l2_lambda: {l2}, Validation Loss: {val_losses_poly_network[-1]:.4f}")

    print(f"Optimal l2_lambda index: {index}, Value: {l2_lambda_list[index]}, Best Validation Loss: {best_val_loss:.4f}")
    return index


def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, path=None, optimizer_type='Adam', l2_lambda=1e-4, scaler_y=None):
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
        base_loss = criterion(predictions, Y_train)
        l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
        train_loss = base_loss + l2_lambda * l2_norm
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        return train_loss

    for epoch in range(n_epochs):
        model.train()

        if optimizer_type in ['Adam', 'SGD']:
            optimizer.zero_grad()
            predictions = model(X_train)
            base_loss = criterion(predictions, Y_train)
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            train_loss = base_loss + l2_lambda * l2_norm
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        elif optimizer_type == 'LBFGS':
            optimizer.step(closure)

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            base_val_loss = criterion(val_predictions, Y_val).item()
            val_loss = base_val_loss + l2_lambda * sum(param.pow(2.0).sum().item() for param in model.parameters())

        train_loss_unscaled = float(train_loss) * (scaler_y.scale_[0] ** 2)
        val_loss_unscaled = float(val_loss) * (scaler_y.scale_[0] ** 2)

        train_losses.append(train_loss_unscaled)
        val_losses.append(val_loss_unscaled)

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {train_loss_unscaled:.4f}, Validation Loss: {val_loss_unscaled:.4f}")

        # if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and epoch == n_epochs - 1:
        #     symbols = sp.symbols(f'x0:{X_train.shape[1]}')
        #     polynomial = model.symbolic_forward(*symbols)
        #     print(f"Polynomial.simplify: {polynomial.simplify().evalf(3)}")

    return model, train_losses, val_losses

# Data loading
# path = 'fagproject/data/Student_Performance.csv'
data = pd.read_excel('fagproject/data/Folds5x2_pp.xlsx')
PN_layers = [3, 3, 3] # Example layers for Polynomial_Network
NN_layers = [9, 6, 2] # Example layers for General_NN is actually [4, 9, 6, 2, 1] because of the input layer and output layer
n_epochs = 10000 # Number of epochs for training
lr = 0.0001 # Learning rate for the optimizer
optimizer = 'Adam' # Optimizer type, can be 'Adam', 'SGD', or 'LBFGS'
#l2_lambda = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
l2_lambda = [1e-4]  # Using a single value for simplicity
index = 0

# data = pd.read_csv(path).dropna()
# data.replace({"Yes": 1, "No": 0}, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3)

scaler_X = StandardScaler()
X_train_np = scaler_X.fit_transform(X_train_np)
X_test_np = scaler_X.transform(X_test_np)

scaler_y = StandardScaler()
scaler_y.fit(y_train_np)
y_train_np = scaler_y.transform(y_train_np)
y_test_np = scaler_y.transform(y_test_np)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

num_features = X_train.shape[1]



#index = find_optimal_lambda(n_epochs=n_epochs, layers=layers, lr=lr, path=path, optimizer_type=optimizer) # Uncomment this line to find the optimal lambda
# print(f'l2_val: {l2_lambda[index]}')

Polynomial_Net = Polynomial_Network(PN_layers, in_features=num_features)
NeuralNet = General_NN(NN_layers, in_features=num_features)
print(f"In-features: {X_train.shape[1]}")

print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(
    NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs,
    learning_rate=lr, optimizer_type=optimizer, l2_lambda=l2_lambda[index], scaler_y=scaler_y)

print(f"\nTraining Polynomial_Network (layers={PN_layers}):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(
    Polynomial_Net, X_train, y_train, X_test, y_test, n_epochs=n_epochs,
    learning_rate=lr, optimizer_type=optimizer, l2_lambda=l2_lambda[index], scaler_y=scaler_y)

with torch.no_grad():
    loss_fn = nn.MSELoss()
    test_loss_poly_network = loss_fn(poly_network(X_test), y_test)
    test_loss_NN = loss_fn(NeuralNet(X_test), y_test)

test_loss_poly_network = test_loss_poly_network * (scaler_y.scale_[0] ** 2)
test_loss_NN = test_loss_NN * (scaler_y.scale_[0] ** 2)

print(f"Test Loss (Neural Network, layers={NN_layers}): {test_loss_NN.item():.4f}")
print(f"Test Loss (Polynomial_Network, layers={PN_layers}): {test_loss_poly_network.item():.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses_NN, label=f"Neural-Network (layers={NN_layers}) - Train Loss", linestyle="solid")
plt.plot(val_losses_NN, label=f"Neural-Network (layers={NN_layers}) - Validation Loss", linestyle="dashed")
plt.plot(train_losses_poly_network, label=f"Polynomial_Network (layers={PN_layers}) - Train Loss - l2_lambda: {l2_lambda[index]}", linestyle="solid")
plt.plot(val_losses_poly_network, label=f"Polynomial_Network (layers={PN_layers}) - Validation Loss - l2_lambda: {l2_lambda[index]}", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss (original scale)")
plt.legend()
plt.title(f"Training vs Validation Losss\nNeural Network vs Polynomial Network (l2_lambda={l2_lambda[index]})")
plt.show()