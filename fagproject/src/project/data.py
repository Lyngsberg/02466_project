import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import argparse
import random
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def quadratic_polynomial(x,y):
    return 3*x**2 + 2*y**2 + 1

def cubic_polynomial(x,y):
    return 3*x**3 + 2*y**3 + 1

def smooth_function(x,y):
    return np.sin(x) + np.cos(y)

def generate_data(n_samples, function, x_low, x_high, y_low, y_high):
    x = np.random.uniform(x_low, x_high, n_samples)
    y = np.random.uniform(y_low, y_high, n_samples)
    z = function(x, y)
    return x, y, z

def generate_data_with_noise(n_samples, function, noise, x_low, x_high, y_low, y_high):
    x = np.random.uniform(x_low, x_high, n_samples)
    y = np.random.uniform(y_low, y_high, n_samples)
    z = function(x, y) + noise * np.random.randn(n_samples)
    return x, y, z

def plot_data(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

# Convert to tensors and save data
def save_data(filename, *data):
    torch.save(data, filename)

# Define a function to process and save data
def process_and_save_data(X, Y, Z, filename):
    X_data = torch.tensor(np.vstack((X, Y)).T, dtype=torch.float32)
    Z_data = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)
    save_data(filename, X_data, Z_data)

def main(num_samples: int, seed: int, x_low: int, x_high: int, y_low:int, y_high: int, ran: int):

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for i in range(1,ran+1):
        X_q, Y_q, Z_q = generate_data(num_samples, quadratic_polynomial, i*x_low, i*x_high, i*y_low, i*y_high)
        X_c, Y_c, Z_c = generate_data(num_samples, cubic_polynomial, i*x_low, i*x_high, i*y_low, i*y_high)
        X_s, Y_s, Z_s = generate_data(num_samples, smooth_function, i*x_low, i*x_high, i*y_low, i*y_high)

        X_q_n, Y_q_n, Z_q_n = generate_data_with_noise(num_samples, quadratic_polynomial, 0.1, i*x_low, i*x_high, i*y_low, i*y_high)
        X_c_n, Y_c_n, Z_c_n = generate_data_with_noise(num_samples, cubic_polynomial, 0.1, i*x_low, i*x_high, i*y_low, i*y_high)
        X_s_n, Y_s_n, Z_s_n = generate_data_with_noise(num_samples, smooth_function, 0.1, i*x_low, i*x_high, i*y_low, i*y_high)



        # Process and save all datasets
        process_and_save_data(X_q, Y_q, Z_q, f'fagproject/data/train_q_{i}.pkl')
        process_and_save_data(X_q_n, Y_q_n, Z_q_n, f'fagproject/data/train_q_n_{i}.pkl')
        process_and_save_data(X_c, Y_c, Z_c, f'fagproject/data/train_c_{i}.pkl')
        process_and_save_data(X_c_n, Y_c_n, Z_c_n, f'fagproject/data/train_c_n_{i}.pkl')
        process_and_save_data(X_s, Y_s, Z_s, f'fagproject/data/train_s_{i}.pkl')
        process_and_save_data(X_s_n, Y_s_n,Z_s_n, f'fagproject/data/train_s_n_{i}.pkl')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--x_low", type=int, default=-1)
    parser.add_argument("--x_high", type=int, default=1)
    parser.add_argument("--y_low", type=int, default=-1)
    parser.add_argument("--y_high", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ran", type=int, default=1)
    args = parser.parse_args()

    main(args.num_samples, args.seed, args.x_low, args.x_high, args.y_low, args.y_high, args.ran)