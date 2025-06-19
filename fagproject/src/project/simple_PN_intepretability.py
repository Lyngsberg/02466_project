import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.model_selection import train_test_split
from scipy import stats
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd

# Import your model here
from Deep_PN.PN_model_triang_deep import Polynomial_Network



def extract_all_terms(expr):
    """
    Extracts all polynomial terms from a sympy expression along with their coefficients.
    Handles constants, single variable terms, and mixed terms like x*y, x**2*y, etc.
    Returns a dictionary: {term_str: coefficient}
    """
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms()

    coeff_dict = {}

    for term in terms:
        coeff, rest = term.as_coeff_Mul()
        rest = sp.simplify(rest)
        term_str = str(rest)
        coeff_dict[term_str] = float(coeff)
    
    return coeff_dict

def plot_coefficients_with_confidence_intervals(coef_list, confidence=0.95, save_path=None):
    """
    Given a list of dicts (coef_list), where each dict maps polynomial terms -> coefficient value,
    compute mean and confidence interval for each term across runs, and plot with error bars.
    """
    # Convert to DataFrame
    df = pd.DataFrame(coef_list).fillna(0)

    terms = df.columns.tolist()
    means = df.mean(axis=0)
    sems = df.sem(axis=0)  # Standard Error of the Mean
    n = len(df)

    # Compute CI margin
    h = sems * stats.t.ppf((1 + confidence) / 2, n - 1)

    # Sort terms for nicer plotting (bias first, then variables)
    def term_sort_key(term):
        if term == '1':
            return (0, '')
        vars_part = ''.join(sorted(term.replace('**', '').replace('*', '').replace(' ', '')))
        if ('*' not in term) and ('+' not in term) and ('-' not in term) and (' ' not in term):
            return (1, vars_part)
        return (2, vars_part)

    sorted_terms = sorted(terms, key=term_sort_key)

    means = means[sorted_terms]
    h = h[sorted_terms]

    plt.figure(figsize=(max(8, len(sorted_terms) * 0.6), 6))
    plt.errorbar(x=range(len(sorted_terms)), y=means, yerr=h, fmt='o', ecolor='r', capsize=5, capthick=1, markersize=5)
    plt.xticks(range(len(sorted_terms)), sorted_terms, rotation=45, ha='right')
    plt.ylabel('Coefficient Value')
    plt.title(f'Polynomial Term Coefficients with {int(confidence*100)}% Confidence Intervals')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, optimizer_type='Adam', l2_lambda=1e-4):
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    Y_train = scaler_y.fit_transform(Y_train)
    Y_val = scaler_y.transform(Y_val)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        base_loss = criterion(predictions, Y_train)
        l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
        train_loss = base_loss + l2_lambda * l2_norm
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    x, y = sp.symbols('x y')
    polynomial = model.symbolic_forward(x, y)

    return model, polynomial

# ... all imports and code above remain the same ...

def main():
    # Load dataset
    path = 'fagproject/data/train_q_n_4.pkl'
    X, y = torch.load(path)

    n_epochs = 1500
    learning_rate = 0.0001
    k = 30  # Number of models to train
    layers = [1]

    coef_list = []
    for i in range(k):
        # Set random seed
        random_seed = i
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=i)
        num_features = X_train.shape[1]
        print(f"Training model {i+1}/{k}")
        model = Polynomial_Network(layers, in_features=num_features)
        model, polynomial_symbolic = train_model(model, X_train, y_train, X_val, y_val, n_epochs, learning_rate)

        # Extract all terms (including mixed) and their coefficients
        coef = extract_all_terms(polynomial_symbolic)
        coef_list.append(coef)

    # Plot average heatmap of weights for all terms across models
    plot_coefficients_with_confidence_intervals(coef_list, confidence=0.95, save_path=f"fagproject/src/project/interpretability_plots/coef_CI_plot_e_{n_epochs}_k_{k}_seeds.png")


if __name__ == "__main__":
    main()
