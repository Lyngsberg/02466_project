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
        self.W = nn.Parameter(torch.randn(in_features + 1, in_features + 1)*0.00001)  # Learnable (in_features+1)x(in_features+1) matrix
        for i in range(0,   in_features + 1):
            for j in range(0, i):
                self.W.data[i, j] = 0
        print(f"PN_Neuron initialized with W shape: {self.W.shape}, W: {self.W}")
        # Create a mask for the upper triangular part
        self.register_buffer('mask', torch.triu(torch.ones(in_features + 1, in_features + 1)))
        # Ensure the diagonal is not masked
        self.mask.data.fill_diagonal_(1)
        print(f"Mask shape: {self.mask.shape}, Mask: {self.mask}")
        # Register a hook to apply the mask to the gradients
        self.W.register_hook(lambda grad: grad * self.mask)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        W_upper = self.W * self.mask
        ones = torch.ones(x.shape[0], 1, device=x.device)
        z = torch.cat([x, ones], dim=1)  # z shape: (batch_size, in_features + 1)
        output = torch.sum(z.unsqueeze(1) @ W_upper @ z.unsqueeze(2), dim=(1, 2))
        return output.unsqueeze(1)  # Shape: (batch_size, 1)

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