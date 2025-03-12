import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle

def quadratic_polynomial(x,y):
    return 3*x**2 + 2*y**2 + 1

def cubic_polynomial(x,y):
    return 3*x**3 + 2*y**3 + 1

def smooth_function(x,y):
    return np.sin(x) + np.cos(y)

def generate_data(n_samples, function):
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    z = function(x, y)
    return x, y, z

def generate_data_with_noise(n_samples, function, noise):
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    z = function(x, y) + noise * np.random.randn(n_samples)
    return x, y, z

def plot_data(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()

num_samples = 1000

X_q, Y_q, Z_q = generate_data(num_samples, quadratic_polynomial)
X_c, Y_c, Z_c = generate_data(num_samples, cubic_polynomial)
X_s, Y_s, Z_s = generate_data(num_samples, smooth_function)

X_q_n, Y_q_n, Z_q_n = generate_data_with_noise(num_samples, quadratic_polynomial, 0.1)
X_c_n, Y_c_n, Z_c_n = generate_data_with_noise(num_samples, cubic_polynomial, 0.1)
X_s_n, Y_s_n, Z_s_n = generate_data_with_noise(num_samples, smooth_function, 0.1)

# Convert to tensors and save data
def save_data(filename, *data):
    torch.save(data, filename)

# Define a function to process and save data
def process_and_save_data(X, Y, Z, filename):
    X_data = torch.tensor(np.vstack((X, Y)).T, dtype=torch.float32)
    Z_data = torch.tensor(Z, dtype=torch.float32).unsqueeze(1)
    save_data(filename, X_data, Z_data)

# Process and save all datasets
process_and_save_data(X_q, Y_q, Z_q, 'fagproject/data/train_q.pkl')
process_and_save_data(X_q_n, Y_q_n, Z_q_n, 'fagproject/data/train_q_n.pkl')
process_and_save_data(X_c, Y_c, Z_c, 'fagproject/data/train_c.pkl')
process_and_save_data(X_c_n, Y_c_n, Z_c_n, 'fagproject/data/train_c_n.pkl')
process_and_save_data(X_s, Y_s, Z_s, 'fagproject/data/train_s.pkl')
process_and_save_data(X_s_n, Y_s_n,Z_s_n, 'fagproject/data/train_s_n.pkl')


# Make 6 sub-3d plots for each of the functions
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(231, projection='3d')
ax1.scatter(X_q, Y_q, Z_q)
ax1.set_title('Quadratic Polynomial')

ax2 = fig.add_subplot(232, projection='3d')
ax2.scatter(X_c, Y_c, Z_c)
ax2.set_title('Cubic Polynomial')

ax3 = fig.add_subplot(233, projection='3d')
ax3.scatter(X_s, Y_s, Z_s)
ax3.set_title('Smooth Function')

ax4 = fig.add_subplot(234, projection='3d')
ax4.scatter(X_q_n, Y_q_n, Z_q_n)
ax4.set_title('Quadratic Polynomial with Noise')

ax5 = fig.add_subplot(235, projection='3d')
ax5.scatter(X_c_n, Y_c_n, Z_c_n)
ax5.set_title('Cubic Polynomial with Noise')

ax6 = fig.add_subplot(236, projection='3d')
ax6.scatter(X_s_n, Y_s_n, Z_s_n)
ax6.set_title('Smooth Function with Noise')

plt.show()