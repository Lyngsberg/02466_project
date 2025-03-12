import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PN_model import PolynomialNet
from sklearn.model_selection import train_test_split
from NN_models import NN_model1

def train_model(model, x_train, y_train, epochs=1000, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients
        output = model(x_train)  # Forward pass
        loss = criterion(output, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        # Print weights before the update
        if model.__class__.__name__ == 'PolynomialNet':
            print(f"Epoch {epoch+1} - Weights before update:\n{model.W.data}")
            
            optimizer.step()
            
            # Print weights after the update
            print(f"Epoch {epoch+1} - Weights after update:\n{model.W.data}")
        else:
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} - Loss: {loss.item()}")

model_PN = PolynomialNet()
model_NN = NN_model1()
epochs = 10
lr = 0.1
path = 'fagproject/data/train_c_n.pkl'
X, y = torch.load(path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_model(model_PN, X_train, y_train, epochs, lr)
train_model(model_NN, X_train, y_train, epochs, lr)

