import torch
import torch.nn as nn
import torch.optim as optim

def train_NN_model(model, x_train, y_train, epochs=1000, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()  # Reset gradients
        output = model(x_train)  # Forward pass
        loss = criterion(output, y_train)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')