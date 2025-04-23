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
from plot import plot_sampled_function_vs_polynomial_estimate, plot_weights, plot_weights_mean, plot_weights_mean_compare


W12_W21 = []
W13_W31 = []
W23_W32 = []


def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs=1000, learning_rate=0.01, path = None):
    weight_history = []
    weight_history_all = []
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) #other BFGS optim try its not from pytorch!!!
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)


    train_losses = []
    val_losses = []
    #adam opt
    for epoch in range(n_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        train_loss = criterion(predictions, Y_train)
        train_loss.backward()
        optimizer.step()

    # for epoch in range(n_epochs):
    #     def closure():
    #         optimizer.zero_grad()
    #         predictions = model(X_train)
    #         train_loss = criterion(predictions, Y_train)
    #         train_loss.backward()
    #         return train_loss
        
    #     # Perform LBFGS optimization step
    #     optimizer.step(closure)
        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and (epoch == n_epochs-1):
            x, y = sp.symbols('x y')
            polynomial = model.symbolic_forward(x, y)
            print(f"Polynomial: {polynomial}")
            print(f"Polynomial.simplify: {polynomial.simplify()}")
            # Print the weights of the neurons
            if hasattr(model, "pn_neuron"):
                for i, neuron in enumerate(model.pn_neuron):
                    W = neuron.W.data
                    # ... your code ...
            elif hasattr(model, "W"):
                W = model.W.data
                # print('W:', W)
                # W12 = W[0, 1].item()
                # W21 = W[1, 0].item()
                # W13, W31 = W[0, 2].item(), W[2, 0].item()
                # W23, W32 = W[1, 2].item(), W[2, 1].item()
                # print(f"Neuron {i}: W12 = {W12}, W21 = {W21}", 'sum:', W12 + W21)
                # W12_W21.append(W12 + W21)
                # W13_W31.append(W13 + W31)
                # W23_W32.append(W23 + W32)







        # Validation step
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            train_loss = criterion(model(X_train), Y_train)
            val_loss = criterion(model(X_val), Y_val)
            if hasattr(model, "pn_neuron"):
                epoch_weights = [neuron.W.cpu().numpy().copy() for neuron in model.pn_neuron]
                weight_history.append(epoch_weights)
            elif hasattr(model, "W"):
                epoch_weights = model.W.cpu().numpy().copy()
                weight_history.append(epoch_weights)
                #print(f"Epoch {epoch}, Weights: {epoch_weights}, weight_history: {weight_history}")


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
            #plot_sampled_function_vs_polynomial_estimate(X_val, Y_val, val_predictions, polynomial=polynomial)
    if hasattr(model, "pn_neuron"):
        #plot_weights(weight_history)
        weight_history_triang.append(weight_history)
    elif hasattr(model, "W"):
        #plot_weights(weight_history)
        weight_history_notriang.append(weight_history)
    
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


if __name__ == "__main__":
    all_weight_histories_triang = []
    all_weight_histories_notriang = []
    for i in range(30):
        random_seed = i
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        path = 'fagproject/data/train_s_2.pkl'
        X, y = torch.load(path)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        poly_network_triangulized = Polynomial_Network(n_neurons=1)
        poly_network = PolynomialNet()

        n_epochs = 2000
        lr = 0.01

        # For triangulized
        weight_history_triang = []
        train_model(poly_network_triangulized, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)
        if weight_history_triang:
            all_weight_histories_triang.append(list(weight_history_triang)) 

        # For not triangulized
        weight_history_notriang = []
        train_model(poly_network, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path)
        if weight_history_notriang:
            all_weight_histories_notriang.append(list(weight_history_notriang))

    # Plot both
    if all_weight_histories_triang and all_weight_histories_notriang:
        print(all_weight_histories_triang)
        print(np.array(all_weight_histories_triang).shape)
        print(np.array(all_weight_histories_notriang).shape)
        plot_weights_mean_compare(all_weight_histories_triang, all_weight_histories_notriang)

