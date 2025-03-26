from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
def plot_sampled_function(x_test, y_test, output, function_name, polynomial=None):   
    path_quadratic_1 = 'fagproject/data/train_q_1.pkl'
    path_quadratic_2 = 'fagproject/data/train_q:2.pkl'
    path_quadratic_with_noise_1 = 'fagproject/data/train_q_n_1.pkl'
    path_quadratic_with_noise_2 = 'fagproject/data/train_q_n_2.pkl'

    path_cubic_1 = 'fagproject/data/train_c_1.pkl'
    path_cubic_2 = 'fagproject/data/train_c:2.pkl'
    path_cubic_with_noise_1 = 'fagproject/data/train_c_n_1.pkl'
    path_cubic_with_noise_2 = 'fagproject/data/train_c_n_2.pkl'

    path_smooth_1 = 'fagproject/data/train_s_1.pkl'
    path_smooth_2 = 'fagproject/data/train_s:2.pkl'
    path_smooth_with_noise_1 = 'fagproject/data/train_s_n_1.pkl'
    path_smooth_with_noise_2 = 'fagproject/data/train_s_n_2.pkl' 
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

    
    #cubic_poly = lambda x, y: x**3 + y**3 - 3*x*y
    qubic_poly = lambda x, y: 3*x**2 + 2*y**2 + 1
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_vals = np.linspace(min(x_test[:, 0].numpy()), max(x_test[:, 0].numpy()), 50)
    y_vals = np.linspace(min(x_test[:, 1].numpy()), max(x_test[:, 1].numpy()), 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    #Z_cubic = cubic_poly(X, Y)
    Z_cubic = qubic_poly(X, Y)
    
    ax.plot_surface(X, Y, Z_cubic, color='y', alpha=0.4, label='Cubic Polynomial')
    
    if polynomial is not None:
        Z_poly = np.zeros_like(X)
        f_poly = sp.lambdify((sp.Symbol('x'), sp.Symbol('y')), polynomial, 'numpy')  # Convert to numpy function
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z_poly[i, j] = f_poly(X[i, j], Y[i, j])
        
        ax.plot_surface(X, Y, Z_poly, color='g', alpha=0.4, label='Learned Polynomial')
    plt.show()
    
