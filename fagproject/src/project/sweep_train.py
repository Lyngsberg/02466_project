import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import random
import importlib    
import shutil
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_model():
    # Get hyperparameters from wandb.config
    data_type = wandb.config.data_type
    model_modul_name = wandb.config.model_modul_name
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    seed = wandb.config.seed

    modul_name = model_modul_name[0]
    model_name = model_modul_name[1]
    
    # Set the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    run = wandb.init(
        entity= "lyngsberg-danmarks-tekniske-universitet-dtu",
        project="Fagprojekt",
        config={  # Config used for the current run
            "BATCH_SIZE": batch_size,
            "LEARNING_RATE": learning_rate,
            "EPOCHS": epochs,
            "SEED": seed
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = f"fagproject/data/{data_type}"
    x, y = torch.load(data_path)
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
    models_modul = importlib.import_module(modul_name)
    model_class = getattr(models_modul, model_name)
    model = model_class().to(device) if models_modul != "PN_models" else model_class(n_neurons=1).to(device)

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

    model_path = os.path.join("fagproject/models", f"{model_name}{data_type}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved locally to {model_path}")


    artifact = wandb.Artifact(
        name="Neural_Network" if modul_name == "NN_models" else "Polynomial_Network",
        type="model",
        description="A trained model",
        metadata={"Val loss": val_mse, "Date": datetime.now()},
    )
    artifact.add_file(model_path)
    run.log_artifact(artifact)

# Initialize sweep
if __name__ == "__main__":
    # This will only be triggered by wandb.agent, not run directly
    pass
