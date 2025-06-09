import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
class Polynomial_Network(nn.Module):
    def __init__(self, n_neurons):
        super(Polynomial_Network, self).__init__()
        self.pn_neuron = nn.ModuleList([PN_Neuron() for _ in range(n_neurons)])

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
    

class PN_Neuron(nn.Module):
    def __init__(self):
        super(PN_Neuron, self).__init__()
        # Store full matrix as parameter
        self.W = nn.Parameter(torch.randn(3, 3))
        self.W.data[1, 0] = 0
        self.W.data[2, 0] = 0
        self.W.data[2, 1] = 0
        # Create a mask for the upper triangular part
        self.register_buffer('mask', torch.triu(torch.ones(3, 3)))
        # Register a hook to apply the mask to the gradients
        self.W.register_hook(lambda grad: grad * self.mask)
    
    def forward(self, x):
        # Extract the upper-triangular part during forward
        W_upper = self.W * self.mask
        ones = torch.ones(x.shape[0], 1, device=x.device)
        z = torch.cat((x, ones), dim=1)
        output = torch.sum(z.unsqueeze(1) @ W_upper @ z.unsqueeze(2), dim=(1,2), keepdim=True).squeeze(-1)
        return output
    
    def symbolic_forward(self, x, y):
        # Use the upper triangular part for symbolic evaluation as well
        W_list = (self.W * self.mask).detach().cpu().numpy().tolist()
        x_exp = sp.Matrix([x, y, 1])
        return (x_exp.T * sp.Matrix(W_list) * x_exp)[0]

# class PN_Neuron(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Store only the 6 parameters of the upper-triangular matrix
#         self.params = nn.Parameter(torch.randn(6))

#     def forward(self, x):
#         # Reconstruct the upper-triangular matrix from params
#         W = torch.zeros(3, 3, device=x.device)
#         W[0, 0], W[0, 1], W[0, 2], W[1, 1], W[1, 2], W[2, 2] = self.params
#         ones = torch.ones(x.shape[0], 1, device=x.device)
#         z = torch.cat((x, ones), dim=1)
#         output = torch.sum(z.unsqueeze(1) @ W @ z.unsqueeze(2), dim=(1, 2), keepdim=True).squeeze(-1)
#         return output

#     def symbolic_forward(self, x, y):
#         W = sp.zeros(3)
#         p = self.params.detach().cpu().numpy()
#         W[0, 0], W[0, 1], W[0, 2], W[1, 1], W[1, 2], W[2, 2] = p
#         z = sp.Matrix([x, y, 1])
#         return (z.T * W * z)[0]