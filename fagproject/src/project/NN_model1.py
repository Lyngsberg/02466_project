import torch
import torch.nn as nn
import torch.optim as optim

# Define the Feedforward Neural Network
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 2 input neurons -> 2 hidden neurons
        self.output = nn.Linear(2, 1)  # 2 hidden neurons -> 1 output neuron

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Activation function for hidden layer
        x = torch.sigmoid(self.output(x))  # Sigmoid activation for output
        return x

# Training function
def train_model(model, x_train, y_train, epochs=1000, lr=0.1):
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

if __name__=="__main__":
    # Create the model
    model = FFNN()

    # Training data (X: inputs, y: labels)
    x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # Example XOR problem

    # Train the model
    train_model(model, x_train, y_train)

    # Example input (batch size of 1, 2 features)
    example_input = torch.tensor([[0.5, -0.3]])
    output = model(example_input)
    print("Output:", output.item())
