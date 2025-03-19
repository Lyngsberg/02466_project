import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the Feedforward Neural Network
class NN_model1(nn.Module):
    def __init__(self):
        super(NN_model1, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 input neurons -> 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden neurons -> 1 output neuron

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Activation function for hidden layer
        x = torch.sigmoid(self.output(x))  # Sigmoid activation for output
        return x

class PN_Neuron(nn.Module):
    def __init__(self):
        super(PN_Neuron, self).__init__()
        self.W = nn.Parameter(torch.randn(3, 3)) # Learnable 3x3 matrix
    
    def forward(self, x):
        ones = torch.ones(x.shape[0], 1)
        z = torch.cat((x, ones), dim=1)  
        output = torch.sum(z @ self.W * z, dim=1, keepdim=True)
        return output
    
class Polynomial_Network(nn.Module):
    def __init__(self, n_neurons):
        super(Polynomial_Network, self).__init__()
        self.pn_neuron = nn.ModuleList([PN_Neuron() for _ in range(n_neurons)])
        # self.final_linear = nn.Linear(n_neurons, 1)
    
    def forward(self, x):
        pn_outputs = torch.cat([neuron(x) for neuron in self.pn_neuron], dim=1)
        # output = self.final_linear(pn_outputs)
        output = torch.sum(pn_outputs, dim=1, keepdim=True)
        return output
    
class PolynomialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(3, 3)) 

    def forward(self, x):
        x_exp = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1) 
        # print(f'x_exp shape: {x_exp.shape}, x_exp: {x_exp}, W shape: {self.W.shape}, x.shape: {x.shape}, x: {x}')
        # print(f'x_exp.unsqueeze(1) shape: {x_exp.unsqueeze(1).shape}, x_exp.unsqueeze(1): {x_exp.unsqueeze(1)}')
        # print(f'x_exp.unsqueeze(2) shape: {x_exp.unsqueeze(2).shape}, x_exp.unsqueeze(2): {x_exp.unsqueeze(2)}, x_exp.unsqueeze(1) @ self.W @ x_exp.unsqueeze(2): {(x_exp.unsqueeze(1) @ self.W @ x_exp.unsqueeze(2)).shape}')
        return torch.sum(x_exp.unsqueeze(1) @ self.W @ x_exp.unsqueeze(2), dim=(1,2), keepdim=True).squeeze(-1)

def quadratic_polynomial(x,y):
    return 3*x**2 + 2*y**2 + 1

def cubic_polynomial(x, y):
    return 3*x**3 + 2*y**3 + 1

def smooth_function(x,y):
    return np.sin(x) + np.cos(y)

# Generate data with noise
def generate_data_with_noise(n_samples, function, noise):
    x = np.random.uniform(-3, 3, (n_samples, 2))  # Generates (x, y) pairs
    z = function(x[:, 0], x[:, 1]) + noise * np.random.randn(n_samples)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1)

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

        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = criterion(val_predictions, Y_val)

        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

    return model, train_losses, val_losses

# Sample size
n_samples = 100

# Generate dataset
X_q, Y_q = generate_data_with_noise(n_samples, quadratic_polynomial, 1)
X_c, Y_c = generate_data_with_noise(n_samples, cubic_polynomial, 1)
X_s, Y_s = generate_data_with_noise(n_samples, smooth_function, 1)

# Split into training (70%), validation (15%), and test (15%)
train_size = int(0.7 * n_samples)
val_size = int(0.15 * n_samples)
test_size = n_samples - train_size - val_size

# Train, validation, test split
X_train_q, Y_train_q = X_q[:train_size], Y_q[:train_size]
X_val_q, Y_val_q = X_q[train_size:train_size + val_size], Y_q[train_size:train_size + val_size]
X_test_q, Y_test_q = X_q[train_size + val_size:], Y_q[train_size + val_size:]

X_train_c, Y_train_c = X_c[:train_size], Y_c[:train_size]
X_val_c, Y_val_c = X_c[train_size:train_size + val_size], Y_c[train_size:train_size + val_size]
X_test_c, Y_test_c = X_c[train_size + val_size:], Y_c[train_size + val_size:]

X_train_s, Y_train_s = X_s[:train_size], Y_s[:train_size]
X_val_s, Y_val_s = X_s[train_size:train_size + val_size], Y_s[train_size:train_size + val_size]
X_test_s, Y_test_s = X_s[train_size + val_size:], Y_s[train_size + val_size:]

Datasets = [(X_train_q, Y_train_q, X_val_q, Y_val_q, X_test_q, Y_test_q),
            (X_train_c, Y_train_c, X_val_c, Y_val_c, X_test_c, Y_test_c),
            (X_train_s, Y_train_s, X_val_s, Y_val_s, X_test_s, Y_test_s)]



# Initialize models
poly_network = Polynomial_Network(n_neurons=1)
NeuralNet = NN_model1()

# Ensure same initial weights for fair comparison
# poly_network.pn_neuron[0].W.data = poly_net.W.data.detach().clone()

# Train models with validation tracking
print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train_q, Y_train_q, X_val_q, Y_val_q, n_epochs=1000, learning_rate=0.01)

print("\nTraining Polynomial_Network (n_neurons=1):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(poly_network, X_train_q, Y_train_q, X_val_q, Y_val_q, n_epochs=1000, learning_rate=0.01)

# Evaluate on test data
with torch.no_grad():
    test_loss_NN = nn.MSELoss()(NeuralNet(X_test_q), Y_test_q)
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test_q), Y_test_q)

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
output_poly_net = NeuralNet(X_test_q)
output_poly_network = poly_network(X_test_q)
