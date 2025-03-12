import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import wandb
import random
import NN_models
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
import importlib    

def train_model(data_type: str, model_name: str, batch_size: int, learning_rate: float, epochs: int, seed: int):
    """
    Train our Feed forward neural networks
    """
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    run = wandb.init(
        entity= "lyngsberg-danmarks-tekniske-universitet-dtu",
        project="Fagprojekt",
        config={
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": learning_rate,
            "EPOCHS": epochs,
            "SEED": seed
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = f"fagproject/data/{data_type}"
    x,y = torch.load(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed)

    x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)   


    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    NN_models = importlib.import_module('NN_models')
    model_class = getattr(NN_models, model_name)
    model = model_class().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_mse += loss.item()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_mse,
        })

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Epoch MSE: {epoch_loss:.4f}, "
            f"Val MSE: {val_mse:.4f}, "
  
        )

    model_path = os.path.join("fagproject/models/NN_models/", f"{model_name}{datetime.now()}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally to {model_path}")


    artifact = wandb.Artifact(
        name="Neural_Network",
        type="model",
        description="A trained model",
        metadata={"Val loss": val_mse,
                  "Date": datetime.now()},
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", type=str, default="train_q.pkl")
    parser.add_argument("--model-name", type=str, default="NN_model1")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(args.data_type, args.model_name, args.batch_size, args.learning_rate, args.epochs, args.seed)