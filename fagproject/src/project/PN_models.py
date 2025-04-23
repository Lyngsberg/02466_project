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
    def __init__(self):
        super(PN_Neuron, self).__init__()
        self.W = nn.Parameter(torch.randn(3, 3))  # Learnable 3x3 matrix

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1)
        z = torch.cat((x, ones), dim=1)  
        output = torch.sum(z.unsqueeze(1) @ self.W @ z.unsqueeze(2), dim=(1,2), keepdim=True).squeeze(-1)
        return output
    
    def symbolic_forward(self, x, y):
        x_exp = sp.Matrix([x, y, 1])  # Expand input (x, y, 1)
        return (x_exp.T * sp.Matrix(self.W.tolist()) * x_exp)[0]  # Compute bilinear form

class PN_Neuron2(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(3, 3))  # For 2D input
        self.W_1d = nn.Parameter(torch.randn(2, 2))  # For 1D input

    def forward(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device)
        z = torch.cat((x, ones), dim=1)

        # Infer correct W based on input size
        if z.shape[1] == 3:
            if not hasattr(self, 'W'):
                self.W = nn.Parameter(torch.randn(3, 3))
            W = self.W
        elif z.shape[1] == 2:
            if not hasattr(self, 'W_1d'):
                self.W_1d = nn.Parameter(torch.randn(2, 2))
            W = self.W_1d
        else:
            raise ValueError("Unsupported input shape for PN_Neuron")

        output = torch.sum(z.unsqueeze(1) @ W @ z.unsqueeze(2), dim=(1,2))
        return output.unsqueeze(1)

    def symbolic_forward(self, x, y=None):
        if y is not None:
            z = sp.Matrix([x, y, 1])
            W = sp.Matrix(self.W.tolist())
        else:
            z = sp.Matrix([x, 1])
            W = sp.Matrix(self.W_1d.tolist())
        return (z.T * W * z)[0]


class Polynomial_Network(nn.Module):
    def __init__(self, n_neurons):
        super(Polynomial_Network, self).__init__()
        self.pn_neuron = nn.ModuleList([PN_Neuron2() for _ in range(n_neurons)])

    def forward(self, x):
        pn_outputs = torch.cat([neuron(x) for neuron in self.pn_neuron], dim=1)
        output = torch.sum(pn_outputs, dim=1, keepdim=True)
        return output

    def symbolic_forward(self, x, y):
        return sum(neuron.symbolic_forward(x, y) for neuron in self.pn_neuron)
    
    
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
    
    def symbolic_forward(self, x, y):
        x_exp = sp.Matrix([x, y, 1])
        return (x_exp.T * sp.Matrix(self.W.tolist()) * x_exp)[0]
    

class Deep_Polynomial_Network(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(PN_Neuron2())  # First layer takes (x, y)
        for _ in range(n_layers - 1):
            self.layers.append(PN_Neuron2())  # Subsequent layers take scalar input

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def symbolic_forward(self, x, y):
        # Start with first layer using (x, y)
        out = self.layers[0].symbolic_forward(x, y)
        
        # Pass scalar output to each next layer
        for layer in self.layers[1:]:
            out = layer.symbolic_forward(out)
        return out

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
    