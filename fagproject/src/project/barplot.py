import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from PN_models import Polynomial_Network, PolynomialNet, PN_Neuron
from NN_models import NN_model1
from sklearn.model_selection import train_test_split
from plot import plot_sampled_function_vs_polynomial_estimate

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def plot_polynomial_coefficients(expr: sp.Expr):
    """Plot magnitudes of coefficients of all variables and monomials in a sympy expression, sorted by magnitude."""
    print(expr)
    expr = sp.simplify(expr)
    expr = sp.expand(expr)

    # Convert to polynomial and extract terms
    poly = sp.Poly(expr)
    coeffs_dict = poly.as_dict()

    labels = []
    magnitudes = []

    for monomial_powers, coeff in coeffs_dict.items():
        # Create readable term label
        term = ' * '.join([
            f"{str(var)}" + (f"**{exp}" if exp > 1 else "") 
            for var, exp in zip(poly.gens, monomial_powers) if exp > 0
        ]) or "1"  # For constant term
        labels.append(term)
        magnitudes.append(abs(float(coeff)))

    # Sort by magnitude
    sorted_items = sorted(zip(labels, magnitudes), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_magnitudes = zip(*sorted_items)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.bar(sorted_labels, sorted_magnitudes, color='orange')
    plt.xlabel("Polynomial Term")
    plt.ylabel("Coefficient Magnitude")
    plt.title("Magnitude of Polynomial Coefficients (Sorted)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()




def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs=1000, learning_rate=0.01, path = None):
    criterion = nn.MSELoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate) #other BFGS optim try its not from pytorch!!!
    optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)

    is_polynomial_model = isinstance(model, (Polynomial_Network, PolynomialNet))

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
        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and (epoch % 300 == 0 or epoch == n_epochs-1):
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
    if is_polynomial_model:
        x, y = sp.symbols('x y')
        final_polynomial = poly_network.symbolic_forward(x, y)
        plot_polynomial_coefficients(final_polynomial)

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

path = path_quadratic_with_noise_1
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
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)

print("\nTraining Polynomial_Network (n_neurons=1):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)

# Evaluate on test data
with torch.no_grad():
    test_loss_NN = nn.MSELoss()(NeuralNet(X_test), y_test)
    test_loss_poly_network = nn.MSELoss()(poly_network(X_test), y_test)


output_poly_net = NeuralNet(X_test)
output_poly_network = poly_network(X_test)

