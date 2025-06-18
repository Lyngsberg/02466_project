import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the Feedforward Neural Network
class NN_model1(nn.Module):
    def __init__(self, in_features=5):
        super().__init__()
        self.hidden = nn.Linear(in_features, 5)
        self.output = nn.Linear(5, 1)

        # Xavier initialization
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Activation function for hidden layer
        x = self.output(x)  # Output layer
        return x

class General_NN(nn.Module):
    def __init__(self, layers, in_features):
        super().__init__()
        self.layers = nn.ModuleList()

        current_in = in_features
        for out_features in layers[:-1]:
            layer = nn.Linear(current_in, out_features)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            current_in = out_features

        self.output_layer = nn.Linear(current_in, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

