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
# random_seed = 42
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)

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
        model = Polynomial_Network(layers, in_features=num_features)
        poly_network, train_losses_poly_network, val_losses_poly_network = train_model(model=model, X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test,
                                      n_epochs=n_epochs, learning_rate=lr, path=None, optimizer_type=optimizer_type, l2_lambda=l2)

        criterion = nn.MSELoss()
        # Test the model
        with torch.no_grad():
            test_loss = criterion(poly_network(X_train), y_train)


        test_loss_unscaled = test_loss * (scaler_y.scale_[0] ** 2)
            

        # print(f"Test Loss (Neural Network): {test_loss_NN.item():.4f}")
        print(f"Test Loss (Polynomial_Network, layers={layers}): {test_loss_unscaled.item():.4f}")




        # plot the loss
        # Plot training vs validation loss
        plt.figure(figsize=(10, 5))
        # plt.plot(train_losses_NN, label="Neural-Network - Train Loss", linestyle="solid")
        # plt.plot(val_losses_NN, label="Neural-Network - Validation Loss", linestyle="dashed")
        plt.plot(train_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Train Loss - l2_lambda: {l2_lambda[i]}", linestyle="solid")
        plt.plot(val_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Validation Loss - l2_lambda: {l2_lambda[i]}", linestyle="dashed")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.show()

        if val_losses_poly_network[-1] < best_val_loss:
            best_val_loss = val_losses_poly_network[-1]
            index = i
        print(f"l2_lambda: {l2}, Validation Loss: {val_losses_poly_network[-1]:.4f}")
    print(f"Optimal l2_lambda index: {index}, Value: {l2_lambda_list[index]}, Best Validation Loss: {best_val_loss:.4f}")


    return index


def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, path=None, optimizer_type='Adam', l2_lambda=1e-4):
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
        base_loss = criterion(predictions,Y_train)
        l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
        train_loss = base_loss + l2_lambda * l2_norm
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return train_loss
    #adam opt
    for epoch in range(n_epochs):
        model.train()

        if optimizer_type in ['Adam', 'SGD']:
            optimizer.zero_grad()
            predictions = model(X_train)
            base_loss = criterion(predictions,Y_train)

            
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            train_loss = base_loss + l2_lambda * l2_norm
            train_loss.backward()

            #print(f'optimizer step: {optimizer.step()}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch}, Train Loss: {train_losses[-1].item():.4f}, Validation Loss: {val_losses[-1].item():.4f}")

        if model.__class__.__name__ in ['Polynomial_Network', 'PolynomialNet'] and epoch == n_epochs - 1:
            symbols = sp.symbols(f'x0:{X_train.shape[1]}')
            polynomial = model.symbolic_forward(*symbols)
            # print(f"Polynomial: {polynomial}")
            #print(f"Polynomial: {polynomial}")
            #print(f"Polynomial.simplify: {polynomial.simplify()}")
            # Print the weights of the neurons
            #for i, neuron in enumerate(model.layers[0]):
                #print(f"Neuron {i} weights matrix W:\n{neuron.W.data}")
            # plot_sampled_function_vs_polynomial_estimate(X_val, Y_vafl, val_predictions, polynomial=polynomial)


    return model, train_losses, val_losses

path_quadratic_1 = 'fagproject/data/train_q.pkl'
path_quadratic_with_noise_1 = 'fagproject/data/train_q_n.pkl'

path_cubic_1 = 'fagproject/data/train_c.pkl'
path_cubic_with_noise_1 = 'fagproject/data/train_c_n.pkl'

path_smooth_1 = 'fagproject/data/train_s.pkl'
path_smooth_with_noise_1 = 'fagproject/data/train_s_n.pkl'

path_student_performance = 'fagproject/data/Student_Performance.csv'

path = path_student_performance

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



num_features = X_train.shape[1]
layers = [3,3,3] 
n_epochs = 10000
lr = 0.0001
optimizer = 'Adam'  # Choose from 'Adam', 'SGD', 'LBFGS'
l2_lambda = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

index = find_optimal_lambda(n_epochs= n_epochs, layers=layers, lr=lr, path = path, optimizer_type=optimizer)
print(f'l2_val: {l2_lambda[index]}')
# Initialize models
#torch.manual_seed(random_seed)
Polynomial_Net = Polynomial_Network(layers, in_features=num_features)
#torch.manual_seed(random_seed)
# NeuralNet = NN_model1(in_features=num_features)
NeuralNet = General_NN(layers, in_features=num_features)
print(f"In-features: {X_train.shape[1]}")

# Train and evaluate models
print("\nTraining Neural Network:")
NeuralNet, train_losses_NN, val_losses_NN = train_model(NeuralNet, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path, optimizer_type=optimizer, l2_lambda=l2_lambda[index])

print(f"\nTraining Polynomial_Network (layers={layers}):")
poly_network, train_losses_poly_network, val_losses_poly_network = train_model(Polynomial_Net, X_train, y_train, X_test, y_test, n_epochs=n_epochs, learning_rate=lr, path=path, optimizer_type=optimizer, l2_lambda=l2_lambda[index])

with torch.no_grad():
    test_loss_poly_network = nn.MSELoss(poly_network(X_test), y_test)
    test_loss_NN = nn.MSELoss(NeuralNet(X_test), y_test)

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
plt.plot(train_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Train Loss - l2_lambda: {l2_lambda[index]}", linestyle="solid")
plt.plot(val_losses_poly_network, label=f"Polynomial_Network (layers={layers}) - Validation Loss - l2_lambda: {l2_lambda[index]}", linestyle="dashed")
plt.xlabel("Epochs")
plt.ylabel("Loss (original scale)")
plt.legend()
plt.title(f"Training vs Validation Loss - {optimizer} - {layers}")
plt.show()