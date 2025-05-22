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

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Activation function for hidden layer
        x = self.output(x)  # Output layer
        return x
    

print("NN_model1 loaded")
