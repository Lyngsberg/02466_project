from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import torch

def plot_sampled_function_vs_polynomial_estimate(x_test, y_test, output, polynomial=None):   
    X = x_test
    Z = y_test
    
    # Reshape X and Y to match Z's shape
    Y = X[:, 1].reshape(Z.shape)
    X = X[:, 0].reshape(Z.shape)
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot Real Data with Polynomial Surface
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.scatter(X, Y, Z, label='Real Data', c='b')
    
    if polynomial is not None:
        # Generate polynomial surface
        f_poly = sp.lambdify((sp.Symbol('x'), sp.Symbol('y')), polynomial, 'numpy')
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = np.linspace(Y.min(), Y.max(), 100)
        X_poly, Y_poly = np.meshgrid(x_vals, y_vals)
        Z_poly = f_poly(X_poly, Y_poly)  # Evaluate polynomial
        
        # Plot polynomial surface
        ax1.plot_surface(X_poly, Y_poly, Z_poly, color='g', alpha=0.4, label='Polynomial Surface')
    
    ax1.set_title('Real Data with Polynomial Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Plot Model Output
    ax2 = fig.add_subplot(232, projection='3d')
    ax2.scatter(X, Y, output, label='Model Output', c='r')
    ax2.set_title('Model Output')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.show()