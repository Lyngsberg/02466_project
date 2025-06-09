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
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

scaler_y = StandardScaler()

def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, optimizer_type='Adam'):
    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)

    train_losses = []
    val_losses = []

    def closure():
        optimizer.zero_grad()
        predictions = model(X_train)
        base_loss = criterion(predictions, Y_train)
        l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
        loss = base_loss + 1e-4 * l2_norm
        loss.backward()
        return loss

    for epoch in range(n_epochs):
        model.train()

        if optimizer_type in ['Adam', 'SGD']:
            optimizer.zero_grad()
            predictions = model(X_train)
            base_loss = criterion(predictions, Y_train)
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            loss = base_loss + 1e-4 * l2_norm
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
        elif optimizer_type == 'LBFGS':
            optimizer.step(closure)
            train_loss = closure().item()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            base_val_loss = criterion(val_predictions, Y_val).item()
            val_loss = base_val_loss + 1e-4 * sum(param.pow(2.0).sum().item() for param in model.parameters())

        # Inverse-transform losses
        train_loss_unscaled = train_loss * (scaler_y.scale_[0] ** 2)
        val_loss_unscaled = val_loss * (scaler_y.scale_[0] ** 2)

        train_losses.append(train_loss_unscaled)
        val_losses.append(val_loss_unscaled)

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {train_loss_unscaled:.4f}, Validation Loss: {val_loss_unscaled:.4f}")

        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and epoch == n_epochs - 1:
            symbols = sp.symbols(f'x0:{X_train.shape[1]}')
            polynomial = model.symbolic_forward(*symbols)
            print(f"Polynomial.simplify: {polynomial.simplify().evalf(3)}")
            for i, neuron in enumerate(model.layers[0]):
                print(f"Neuron {i} weights matrix W:\n{neuron.W.data}")

    return model, train_losses, val_losses

path = 'fagproject/data/Student_Performance.csv'

# Load and preprocess data
data = pd.read_csv(path).dropna()
data.replace({"Yes": 1, "No": 0}, inplace=True)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.3, random_state=random_seed)

scaler_X = StandardScaler()
X_train_np = scaler_X.fit_transform(X_train_np)
X_test_np = scaler_X.transform(X_test_np)

scaler_y.fit(y_train_np)
y_train_np = scaler_y.transform(y_train_np)
y_test_np = scaler_y.transform(y_test_np)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

num_features = X_train.shape[1]
layers = [1,1,1]
n_epochs = 5000
lr = 0.001
optimizer = 'Adam'

# Initialize models
torch.manual_seed(random_seed)
Polynomial_Net = Polynomial_Network(layers, in_features=num_features)
torch.manual_seed(random_seed)
NeuralNet = General_NN(layers, in_features=num_features)
print(f"In-features: {X_train.shape[1]}")

# Train and evaluate models
print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, optimizer_type=optimizer)

print(f"\nTraining Polynomial_Network (layers={layers}):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(Polynomial_Net, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, optimizer_type=optimizer)

print("\nTraining complete. Evaluating models...")

with torch.no_grad():
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test), y_test).item()
    test_loss_NN = nn.MSELoss()(NeuralNet(X_test), y_test).item()

# Inverse-transform test losses
test_loss_poly_network_unscaled = test_loss_poly_network * (scaler_y.scale_[0] ** 2)
test_loss_NN_unscaled = test_loss_NN * (scaler_y.scale_[0] ** 2)

print(f"Test Loss (Neural Network): {test_loss_NN_unscaled:.4f}")
print(f"Test Loss (Polynomial_Network, layers={layers}): {test_loss_poly_network_unscaled:.4f}")

# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses_NN, label="Neural-Network - Train Loss", linestyle="solid")
plt.plot(val_losses_NN, label="Neural-Network - Validation Loss", linestyle="dashed")
plt.plot(train_losses_poly_network, label=f"Polynomial_Network - Train Loss", linestyle="solid")
plt.plot(val_losses_poly_network, label=f"Polynomial_Network - Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss (original scale)")
plt.legend()
plt.title(f"Training vs Validation Loss - {optimizer} - {layers}")
plt.show()