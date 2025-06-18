import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from PN_models import Polynomial_Network
from sklearn.model_selection import train_test_split
from plot import plot_sampled_function_vs_polynomial_estimate
from scipy import stats
from sympy import Symbol, Mul, Pow
from collections import defaultdict
from Deep_PN.PN_model_triang_deep import Polynomial_Network, PN_Neuron
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

# Set random seed
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)



def get_order(expr):
    if isinstance(expr, Symbol):
        return 1
    elif isinstance(expr, Pow):
        return expr.exp
    elif isinstance(expr, Mul):
        # Skip mixed terms like x*y
        if any(isinstance(arg, Mul) for arg in expr.args):
            return None
        vars_in_term = [a for a in expr.args if isinstance(a, Symbol)]
        exps_in_term = [1 for _ in vars_in_term]  # x is x**1
        exps_in_term += [a.exp for a in expr.args if isinstance(a, Pow)]
        if len(set(a if isinstance(a, Symbol) else a.base for a in expr.args)) > 1:
            return None  # mixed term
        return sum(exps_in_term)
    elif isinstance(expr, int) or isinstance(expr, float):
        return 0
    return None

def extract_max_by_order_single_dict(coef_dict):
    order_dict = defaultdict(list)

    for expr, coeff in coef_dict.items():
        order = get_order(expr)
        if order is not None:
            order_dict[int(order)].append(abs(coeff))

    # Get max coefficient per order
    max_by_order = {order: max(values) for order, values in order_dict.items()}
    return max_by_order

def summarize_coefficients(coef_list, confidence=0.95):
    keys = list(coef_list[0].keys())
    summary = {}

    for key in keys:
        values = []
        for d in coef_list:
            try:
                values.append(d[key])
            except KeyError:
                print(f"Missing key '{key}' in dictionary: {d}")
                continue  # Skip this dictionary
        values = np.array(values)
        mean = np.mean(values)
        sem = stats.sem(values)  # standard error of the mean
        ci_range = sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        summary[key] = {
            "mean": mean,
            "ci": ci_range
        }
    
    return summary

def extract_single_variable_terms(expr):
    expr = sp.expand(expr)
    terms = expr.as_ordered_terms()

    coeff_dict = {}

    for term in terms:
        free_symbols = term.free_symbols
        
        # Only keep terms with exactly one variable (x, x**2, y, y**3, etc.)
        if len(free_symbols) == 1:
            coeff, rest = term.as_coeff_Mul()
            rest = sp.simplify(rest)
            coeff_dict[rest] = float(coeff)

    return coeff_dict


def plot_coefficients_with_ci(summary):
    keys = list(summary.keys())
    means = [summary[k]["mean"] for k in keys]
    cis = [summary[k]["ci"] for k in keys]

    # Convert sympy keys to strings for labeling
    labels = [str(k) for k in keys]

    # Reverse the order
    keys = keys[::-1]
    means = means[::-1]
    cis = cis[::-1]
    labels = labels[::-1]

    x = np.arange(len(keys))

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means, yerr=cis, fmt='o', capsize=5, color='tab:blue')
    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Coefficient Value")
    plt.title("Mean and Confidence Interval of Coefficients")
    plt.ylim(-0.05, 0.1)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"fagproject/src/project/interpretability_plots/conf_e_{n_epochs}_k_{k}")
    plt.close()

def train_model(model, X_train, Y_train, X_val, Y_val, n_epochs, learning_rate=0.01, path = None, optimizer_type='Adam', l2_lambda=1e-4):
    # Scale first (while still NumPy arrays)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    Y_train = scaler_y.fit_transform(Y_train)
    Y_val = scaler_y.transform(Y_val)

    # Then convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    Y_val = torch.tensor(Y_val, dtype=torch.float32)

    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20)

    train_losses = []
    val_losses = []

    def closure():
        optimizer.zero_grad()
        predictions = model(X_train)
        base_loss = criterion(predictions,Y_train)
        l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
        train_loss = base_loss + l2_lambda * l2_norm
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        return train_loss

    for epoch in range(n_epochs):
        model.train()

        if optimizer_type in ['Adam', 'SGD']:
            optimizer.zero_grad()
            predictions = model(X_train)
            base_loss = criterion(predictions,Y_train)

            
            l2_norm = sum(param.pow(2.0).sum() for param in model.parameters())
            train_loss = base_loss + l2_lambda * l2_norm
            train_loss.backward()

            #print(f'optimizer step: {optimizer.step()}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        elif optimizer_type == 'LBFGS':
            optimizer.step(closure)


    x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
    #model = model.to('cpu')
    polynomial = model.symbolic_forward(x1, x2, x3, x4)
    print(polynomial)

    return model, train_losses, val_losses, polynomial

# Load dataset
path = "Folds5x2_pp.xlsx" 

if path == "Student_Performance.csv":
    data = pd.read_csv(f'fagproject/data/{path}')
    data = data.dropna()
    # Convert "Yes"/"No" to 1/0
    data.replace({"Yes": 1, "No": 0}, inplace=True)
elif path == "Folds5x2_pp.xlsx":
    data = pd.read_excel('fagproject/data/Folds5x2_pp.xlsx')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Train Polynomial Network
n_epochs = 10000
learning_rate = 0.00093
k = 30
layers = [3,3,3] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training Polynomial Network...")
criterion = nn.MSELoss()
coef_list = []
for i in range(k):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=i)
    X_val = torch.from_numpy(X_val).double().to(device)
    X_train = torch.from_numpy(X_train).double().to(device)
    num_features = X_train.shape[1]
    print(f"Traning model: {i+1}")
    Polynomial_Net = Polynomial_Network(layers, in_features=num_features)
    poly_network, train_losses, val_losses, polynomial_symbolic = train_model(Polynomial_Net, X_train, y_train, X_val, y_val, n_epochs=n_epochs, learning_rate=learning_rate, path=path)


    coef = extract_single_variable_terms(polynomial_symbolic)
    coef_list.append(coef)
    print(coef)

#print(coef_list)
results = [extract_max_by_order_single_dict(d) for d in coef_list]
summary = summarize_coefficients(results)
plot_coefficients_with_ci(summary)
