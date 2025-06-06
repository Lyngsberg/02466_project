import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from PN_model_triang_deep import Polynomial_Network, PN_Neuron
from NN_deep import General_NN
from sklearn.model_selection import train_test_split
# from plot import plot_sampled_function_vs_polynomial_estimate
print("Now training...")
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)



def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, path = None, optimizer_type='Adam'):
    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        # LBFGS optimizer requires a closure function to compute the loss
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)


    train_losses = []
    val_losses = []
    def closure():
        optimizer.zero_grad()
        predictions = model(X_train)
        train_loss = criterion(predictions, Y_train)
        train_loss.backward()
        return train_loss
    #adam opt
    for epoch in range(n_epochs):

        if optimizer.__class__.__name__ == 'Adam' or optimizer.__class__.__name__ == 'SGD':
            # Training step
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train)
            train_loss = criterion(predictions, Y_train)
            train_loss.backward()
            optimizer.step()

        elif optimizer.__class__.__name__ == 'LBFGS':
            # Perform LBFGS optimization step
            optimizer.step(closure)

            

        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            train_loss = criterion(model(X_train), Y_train)
            val_loss = criterion(model(X_val), Y_val)


        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and epoch == n_epochs - 1:
            symbols = sp.symbols(f'x0:{X_train.shape[1]}')  # e.g., x0, x1, x2 for 3D
            polynomial = model.symbolic_forward(*symbols)
            # print(f"Polynomial: {polynomial}")
            print(f"Polynomial: {polynomial}")
            print(f"Polynomial.simplify: {polynomial.simplify()}")
            # Print the weights of the neurons
            for i, neuron in enumerate(model.layers[0]):
                print(f"Neuron {i} weights matrix W:\n{neuron.W.data}")
            # plot_sampled_function_vs_polynomial_estimate(X_val, Y_val, val_predictions, polynomial=polynomial)


    return model, train_losses, val_losses

path_quadratic_1 = 'fagproject/data/train_q.pkl'
path_quadratic_with_noise_1 = 'fagproject/data/train_q_n.pkl'

path_cubic_1 = 'fagproject/data/train_c.pkl'
path_cubic_with_noise_1 = 'fagproject/data/train_c_n.pkl'

path_smooth_1 = 'fagproject/data/train_s.pkl'
path_smooth_with_noise_1 = 'fagproject/data/train_s_n.pkl'

path_student_performance = 'fagproject/data/Student_Performance.csv'

path = path_student_performance

# Load data
data = pd.read_csv(path)
data = data.dropna()
# Convert "Yes"/"No" to 1/0
data.replace({"Yes": 1, "No": 0}, inplace=True)
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target variable
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Ensure y is a column vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)


num_features = X_train.shape[1]
layers = [10] 
n_epochs = 500
lr = 0.001
optimizer = 'LBFGS'  # Choose from 'Adam', 'SGD', 'LBFGS'

# Initialize models
torch.manual_seed(random_seed)
Polynomial_Net = Polynomial_Network(layers, in_features=num_features)
torch.manual_seed(random_seed)
# NeuralNet = NN_model1(in_features=num_features)
NeuralNet = General_NN(layers, in_features=num_features)
print(f"In-features: {X_train.shape[1]}")


# Train models with validation tracking
print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path, optimizer_type=optimizer)

print(f"\nTraining Polynomial_Network (layers={layers}):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(Polynomial_Net, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path, optimizer_type=optimizer)

# Evaluate on test data
with torch.no_grad():
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test), y_test)
    test_loss_NN = nn.MSELoss()(NeuralNet(X_test), y_test)

print(f"Test Loss (Neural Network): {test_loss_NN.item():.4f}")
print(f"Test Loss (Polynomial_Network, layers={layers}): {test_loss_poly_network.item():.4f}")

# log-transform the losses
train_losses_NN = np.log(train_losses_NN)
val_losses_NN = np.log(val_losses_NN)
train_losses_poly_network = np.log(train_losses_poly_network)
val_losses_poly_network = np.log(val_losses_poly_network)

# plot the loss
# Plot training vs validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses_NN, label="Neural-Network - Train Loss", linestyle="solid")
plt.plot(val_losses_NN, label="Neural-Network - Validation Loss", linestyle="dashed")
plt.plot(train_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Train Loss", linestyle="solid")
plt.plot(val_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Validation Loss", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()

# # Save the figure
# plt.savefig(f"fagproject/figs/{optimizer}_{layers}_{lr}_loss_plot.png")