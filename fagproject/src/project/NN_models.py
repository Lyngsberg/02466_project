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


