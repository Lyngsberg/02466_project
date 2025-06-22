import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp

# random_seed = 42
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)

class PN_Neuron(nn.Module):
    def __init__(self, in_features=2):
        super().__init__()
        self.in_features = in_features
        size = in_features + 1  # account for bias term

        # Glorot initialization adapted for symmetric quadratic forms
        fan_in = size
        fan_out = size
        std = np.sqrt(2.0 / (fan_in + fan_out))*0.1
        W = torch.randn(size, size) * std

        # Zero lower-triangular part to ensure symmetry via upper triangle
        for i in range(size):
            for j in range(i):
                W[i, j] = 0.0

        self.W = nn.Parameter(W)

        # Create and register mask for upper triangle (including diagonal)
        self.register_buffer('mask', torch.triu(torch.ones(size, size)))
        self.mask.data.fill_diagonal_(1)

        # Register hook to keep gradients masked
        self.W.register_hook(lambda grad: grad * self.mask)

    def forward(self, x):
        W_upper = self.W * self.mask
        ones = torch.ones(x.shape[0], 1, device=x.device)
        z = torch.cat([x, ones], dim=1)  # shape: (batch_size, in_features + 1)
        output = torch.sum(z.unsqueeze(1) @ W_upper @ z.unsqueeze(2), dim=(1, 2))
        return output.unsqueeze(1)

    def symbolic_forward(self, *symbols):
        z = sp.Matrix(list(symbols) + [1])
        W = sp.Matrix((self.W @ self.mask).detach().cpu().numpy().tolist())
        return (z.T @ W @ z)[0]


class Polynomial_Network(nn.Module):
    def __init__(self, layers, in_features):
        super().__init__()
        self.layers = nn.ModuleList()
        self.symbolic_layers = []

        current_in_features = in_features
        for n_neurons in layers:
            layer = nn.ModuleList([PN_Neuron(current_in_features) for _ in range(n_neurons)])
            self.layers.append(layer)
            current_in_features = n_neurons

        self.output_layer = nn.Linear(current_in_features, 1)

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
