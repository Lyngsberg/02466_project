import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class PN_Neuron(nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.in_features = in_features
        self.W = nn.Parameter(torch.randn(in_features + 1, in_features + 1))  # Learnable (in_features+1)x(in_features+1) matrix

    def forward(self, x):
        # x shape: (batch_size, in_features)
        ones = torch.ones(x.shape[0], 1, device=x.device)
        z = torch.cat([x, ones], dim=1)  # z shape: (batch_size, in_features + 1)
        output = torch.sum(z.unsqueeze(1) @ self.W @ z.unsqueeze(2), dim=(1, 2))
        return output.unsqueeze(1)  # Shape: (batch_size, 1)

    def symbolic_forward(self, *symbols):
        z = sp.Matrix(list(symbols) + [1])
        W = sp.Matrix(self.W.tolist())
        return (z.T * W * z)[0]

class Polynomial_Network(nn.Module):
    def __init__(self, layers, in_features):
        super().__init__()
        self.layers = nn.ModuleList()
        self.symbolic_layers = []

        current_in_features = in_features
        for n_neurons in layers:
            layer = nn.ModuleList([PN_Neuron(current_in_features) for _ in range(n_neurons)])
            self.layers.append(layer)
            current_in_features = n_neurons  # Next layer's input is this layer's output

        self.output_layer = nn.Linear(current_in_features, 1)  # Final scalar output

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([neuron(x) for neuron in layer], dim=1)
        return self.output_layer(x)

    def symbolic_forward(self, *symbols):
        input_syms = list(symbols)
        for layer in self.layers:
            input_syms = [neuron.symbolic_forward(*input_syms) for neuron in layer]
        linear_weights = self.output_layer.weight.data.numpy().flatten()
        bias = self.output_layer.bias.item()
        return sum(w * h for w, h in zip(linear_weights, input_syms)) + bias

class PolynomialWidth2(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(3, 3))
        self.w2 = nn.Parameter(torch.randn(3, 3))
    
    def forward(self, x):
        x_exp = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # Shape (batch, 3)
        
        # Quadratic transformation for each neuron
        n1 = torch.sum(x_exp.unsqueeze(1) @ self.w1 @ x_exp.unsqueeze(2), dim=(1, 2))
        n2 = torch.sum(x_exp.unsqueeze(1) @ self.w2 @ x_exp.unsqueeze(2), dim=(1, 2))
        
        return torch.stack([n1, n2], dim=1)  # Shape (batch, 2)
    
    def symbolic_forward(self, x, y):
        x_exp = sp.Matrix([x, y, 1])
        W1 = sp.Matrix(self.w1.tolist())
        W2 = sp.Matrix(self.w2.tolist())
        return [(x_exp.T * W1 * x_exp)[0], (x_exp.T * W2 * x_exp)[0]]    



