import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
<<<<<<< Updated upstream
from PN_models import Polynomial_Network, PolynomialNet, PN_Neuron, Deep_Polynomial_Network, PolynomialWidth2
=======
from PN_models import Polynomial_Network, PN_Neuron, Deep_Polynomial_Network, PN_Neuron2
>>>>>>> Stashed changes
from NN_models import NN_model1
from sklearn.model_selection import train_test_split
from plot import plot_sampled_function_vs_polynomial_estimate

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)



def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs=1000, learning_rate=0.01, path = None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #other BFGS optim try its not from pytorch!!!
    # optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)


    train_losses = []
    val_losses = []
    #adam opt
    # for epoch in range(n_epochs):
    #     # Training step
    #     model.train()
    #     optimizer.zero_grad()
    #     predictions = model(X_train)
    #     train_loss = criterion(predictions, Y_train)
    #     train_loss.backward()
    #     optimizer.step()

    for epoch in range(n_epochs):
        def closure():
            optimizer.zero_grad()
            predictions = model(X_train)
            train_loss = criterion(predictions, Y_train)
            train_loss.backward()
            return train_loss
        
        # Perform LBFGS optimization step
        optimizer.step(closure)
        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet', 'Deep_Polynomial_Network'] and (epoch % 300 == 0 or epoch == n_epochs-1):
            x, y = sp.symbols('x y')
            polynomial = model.symbolic_forward(x, y)
            print(f"Polynomial: {polynomial}")
            print(f"Polynomial.simplify: {polynomial.simplify()}")

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
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
        
        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and epoch == n_epochs - 1:
            x, y = sp.symbols('x y')
            polynomial = model.symbolic_forward(x, y)
            print(f"Polynomial: {polynomial}")
            print(f"Polynomial.simplify: {polynomial.simplify()}")
            plot_sampled_function_vs_polynomial_estimate(X_val, Y_val, val_predictions, polynomial=polynomial)


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

path = path_smooth_with_noise_1
X, y = torch.load(path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
<<<<<<< Updated upstream
# poly_network = Polynomial_Network(n_neurons=1)
poly_network = Polynomial_Network(n_neurons=1)
deep_poly_network = Deep_Polynomial_Network(n_layers=2)
# wide_poly_network = PolynomialWidth2()
=======
poly_network = Deep_Polynomial_Network(in_features=3, n_layers=1)  # 2 input features, 1 layer
>>>>>>> Stashed changes
# poly_network = PolynomialNet()
# NeuralNet = NN_model1()
# NeuralNet = Polynomial_Network(n_neurons=1)

n_epochs = 10000
lr = 0.01

# Ensure same initial weights for fair comparison
# poly_network.pn_neuron[0].W.data = poly_net.W.data.detach().clone()

# Train models with validation tracking
# print("\nTraining Neural Network:")
# NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)
print("\nTraining Deep_Polynomial_Network:")
deep_poly_network, train_losses_deep_poly_network, val_losses_deep_poly_network = train_model(deep_poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)

print("\nTraining Polynomial_Network (n_neurons=1):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)

# print("\nTraining PolynomialWidth2 (n_neurons=1):")
# wide_poly_network, train_losses_wide_poly_network, val_losses_wide_poly_network = train_model(wide_poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)

# Evaluate on test data
with torch.no_grad():
    test_loss_NN = nn.MSELoss()(deep_poly_network(X_test), y_test)
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test), y_test)
    # test_loss_wide_poly_network = nn.MSELoss()(wide_poly_network(X_test), y_test)

print(f"Test Loss (deep_poly_network (n=2)): {test_loss_NN.item():.4f}")
print(f"Test Loss (Polynomial_Network, n_neurons=1): {test_loss_poly_network.item():.4f}")
# print(f"Test Loss (PolynomialWidth2, n_neurons=1): {test_loss_wide_poly_network.item():.4f}")

# plot the loss
# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses_deep_poly_network, label="deep_poly_network (n=2) - Train Loss", linestyle="solid")
plt.plot(val_losses_deep_poly_network, label="deep_poly_network (n=2) - Validation Loss", linestyle="dashed")
plt.plot(train_losses_poly_network, label="Polynomial_Network (n=1) - Train Loss", linestyle="solid")
plt.plot(val_losses_poly_network, label="Polynomial_Network (n=1) - Validation Loss", linestyle="dashed")
# plt.plot(train_losses_wide_poly_network, label="PolynomialWidth2 (n=1) - Train Loss", linestyle="solid")
# plt.plot(val_losses_wide_poly_network, label="PolynomialWidth2 (n=1) - Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()
output_poly_net = deep_poly_network(X_test)
output_poly_network = poly_network(X_test)
# output_wide_poly_network = wide_poly_network(X_test)

