import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from PN_models import Polynomial_Network
from NN_models import NN_model1
from sklearn.model_selection import train_test_split

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)



def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs=1000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        train_loss = criterion(predictions, Y_train)
        train_loss.backward()
        optimizer.step()

        if model.__class__.__name__ == ('Polynomial_Network' or 'PolynomialNet') and (epoch % 300 == 0 or epoch == n_epochs-1):
            x, y = sp.symbols('x y')
            polynomial = model.symbolic_forward(x, y)
            print(f"Polynomial: {polynomial}")
            print(f"Polynomial.simplify: {polynomial.simplify()}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, Y_val)

        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    return model, train_losses, val_losses

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

path = path_quadratic_1
X, y = torch.load(path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
poly_network = Polynomial_Network(n_neurons=1)
# poly_network = PolynomialNet()
# NeuralNet = NN_model1()
NeuralNet = NN_model1()

n_epochs = 1000
lr = 0.01

# Ensure same initial weights for fair comparison
# poly_network.pn_neuron[0].W.data = poly_net.W.data.detach().clone()

# Train models with validation tracking
print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr)

print("\nTraining Polynomial_Network (n_neurons=1):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr)

# Evaluate on test data
with torch.no_grad():
    test_loss_NN = nn.MSELoss()(NeuralNet(X_test), y_test)
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test), y_test)

print(f"Test Loss (Neural Network): {test_loss_NN.item():.4f}")
print(f"Test Loss (Polynomial_Network, n_neurons=1): {test_loss_poly_network.item():.4f}")

# plot the loss
# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses_NN, label="Neural Network - Train Loss", linestyle="solid")
plt.plot(val_losses_NN, label="Neural Network - Validation Loss", linestyle="dashed")
plt.plot(train_losses_poly_network, label="Polynomial_Network (n=1) - Train Loss", linestyle="solid")
plt.plot(val_losses_poly_network, label="Polynomial_Network (n=1) - Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
output_poly_net = NeuralNet(X_test)
output_poly_network = poly_network(X_test)
