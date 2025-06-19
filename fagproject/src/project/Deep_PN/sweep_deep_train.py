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
import pandas as pd
from sklearn.preprocessing import StandardScaler

def train_model():
    print("Now training...")
    run = wandb.init(
        entity="lyngsberg-danmarks-tekniske-universitet-dtu",
        project="Fagprojekt",
        reinit=True
    )
    
    config = wandb.config  # Now this is safe

    # Get hyperparameters from wandb.config
    data_type = config.data_type
    model_modul_name = config.model_modul_name
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    epochs = config.epochs
    seed = config.seed
    optimizer_name = config.optimizer_name
    layers = config.layers
    l2_lambda = config.l2_lambda



    """
    # Get hyperparameters from wandb.config
    data_type = wandb.config.data_type
    model_modul_name = wandb.config.model_modul_name
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    seed = wandb.config.seed
    """

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

    # Load data
    if data_type == "Student_Performance.csv":
        data = pd.read_csv(f'fagproject/data/{data_type}')
        data = data.dropna()
        # Convert "Yes"/"No" to 1/0
        data.replace({"Yes": 1, "No": 0}, inplace=True)
    elif data_type == "Folds5x2_pp.xlsx":
        data = pd.read_excel('fagproject/data/Folds5x2_pp.xlsx')
    else:
        raise ValueError("Unsupported data type. Please use 'Student_Performance.csv' or 'Folds5x2_pp.xlsx'.")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # First, split into temp train and test
    X_train_val, X_val, y_train_val, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Then split train_val into train and val
    X_train, X_test, y_train, y_test = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 0.25 of 0.8 = 0.2


    scaler_X = StandardScaler()
    X_train_np = scaler_X.fit_transform(X_train_np)
    X_val_np = scaler_X.transform(X_val_np)
    X_test_np = scaler_X.transform(X_test_np)

    scaler_y = StandardScaler()
    scaler_y.fit(y_train_np)
    y_train_np = scaler_y.transform(y_train_np)
    y_val_np = scaler_y.transform(y_val_np)
    y_test_np = scaler_y.transform(y_test_np)

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = X_train.shape[1]
    # Initialize the model
    models_modul = importlib.import_module(modul_name)
    model_class = getattr(models_modul, model_name)
    model = model_class(layers=layers, in_features=num_features).to(device) if modul_name != "PN_model_triang_deep" else model_class(layers=layers, in_features=num_features).to(device)

    criterion = nn.MSELoss().to(device)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)

    if optimizer_name == "LBFGS":
        def closure():
            optimizer.zero_grad()
            predictions = model(X_train)
            base_loss = criterion(predictions, y_train)
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            loss = base_loss + l2_lambda * l2_norm
            loss.backward()
            return loss

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            optimizer.step(closure)  # Pass the closure function

            # Unscale training loss
            train_loss = closure().item()  # Call once here
            train_mse = train_loss * (scaler_y.scale_[0] ** 2)

            model.eval()
            val_mse = 0.0
            num_batches = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = model(x_batch)
                    base_loss = criterion(outputs, y_batch)
                    l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
                    loss = base_loss + l2_lambda * l2_norm

                    # Unscale base loss
                    unscaled_val_loss = base_loss * (scaler_y.scale_[0] ** 2)
                    val_mse += unscaled_val_loss.item()
                    num_batches += 1

            val_mse /= num_batches

            wandb.log({
                "epoch": epoch + 1,
                "train_MSE": train_mse,
                "validation_MSE": val_mse,
            })
            print(f"Train MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")


    else:
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss_total = 0.0
            train_samples = 0

            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(x_batch)
                base_loss = criterion(outputs, y_batch)

                l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
                loss = base_loss + l2_lambda * l2_norm

                loss.backward()
                optimizer.step()

                # Track unscaled loss
                unscaled_loss = base_loss * (scaler_y.scale_[0] ** 2)
                train_loss_total += unscaled_loss.item() * x_batch.size(0)
                train_samples += x_batch.size(0)

            train_mse = train_loss_total / train_samples

            model.eval()
            val_loss_total = 0.0
            val_samples = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                    outputs = model(x_batch)
                    base_loss = criterion(outputs, y_batch)

                    l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
                    loss = base_loss + l2_lambda * l2_norm

                    # Track unscaled loss
                    unscaled_loss = base_loss * (scaler_y.scale_[0] ** 2)
                    val_loss_total += unscaled_loss.item() * x_batch.size(0)
                    val_samples += x_batch.size(0)

            val_mse = val_loss_total / val_samples

            wandb.log({
                "epoch": epoch + 1,
                "train_MSE": train_mse,
                "validation_MSE": val_mse,
            })

    print(val_mse)


    artifact = wandb.Artifact(
        name="Neural_Network" if modul_name == "NN_models" else "Polynomial_Network",
        type="model",
        description="A trained model",
        metadata={"val_loss": val_mse, "Date": datetime.now()},
    )
    #artifact.add_file(model_path)
    run.log_artifact(artifact)

# Initialize sweep
if __name__ == "__main__":
    train_model()