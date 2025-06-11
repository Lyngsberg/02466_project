import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.model_selection import train_test_split
from scipy import stats
from sympy import Symbol, Mul, Pow
from collections import defaultdict
from Deep_PN.PN_model_triang_deep import Polynomial_Network, PN_Neuron
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd



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


def plot_coefficients_heatmap(summary):
    import pandas as pd

    # Convert to DataFrame for seaborn heatmap
    data = []
    labels = []

    for key, stats in summary.items():
        labels.append(str(key))
        data.append([stats["mean"], stats["ci"]])

    df = pd.DataFrame(data, columns=["Mean", "Confidence Interval"], index=labels)

    # Sort rows by order (if possible)
    def order_key(label):
        try:
            if "**" in label:
                base, exp = label.split("**")
                return int(exp)
            elif label.isalpha():
                return 1
            else:
                return 0
        except:
            return 99

    df = df.loc[sorted(df.index, key=order_key)]

    plt.figure(figsize=(8, max(4, len(df)*0.5)))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5, cbar_kws={'label': 'Value'})
    plt.title("Heatmap of Coefficients (Mean and CI)")
    plt.xlabel("Statistic")
    plt.ylabel("Polynomial Term")
    plt.tight_layout()
    plt.savefig(f"fagproject/src/project/interpretability_plots/heatmap_e_{n_epochs}_k_{k}")
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
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
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


    x, y = sp.symbols('x y')
    #model = model.to('cpu')
    polynomial = model.symbolic_forward(x, y)
    print(polynomial)

    return model, train_losses, val_losses, polynomial

# Load dataset
path = 'fagproject/data/train_q_n_1.pkl'
X, y = torch.load(path)


# Train Polynomial Network
n_epochs = 10
learning_rate = 0.0001
k = 5
layers = [2] 

print("Training Polynomial Network...")

coef_list = []
for i in range(k):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=i)
    num_features = X_train.shape[1]
    print(f"Traning model: {i+1}")
    Polynomial_Net = Polynomial_Network(layers, in_features=num_features)
    poly_network, train_losses, val_losses, polynomial_symbolic = train_model(Polynomial_Net, X_train, y_train, X_val, y_val, n_epochs=n_epochs, learning_rate=learning_rate, path=path)

    # Evaluate on test data
    with torch.no_grad():
        test_loss = nn.MSELoss()(poly_network(X_val), y_val)
        #print(f"Test Loss (Polynomial Network): {test_loss.item():.4f}")

    coef = extract_single_variable_terms(polynomial_symbolic)
    coef_list.append(coef)

print(coef_list)
results = [extract_max_by_order_single_dict(d) for d in coef_list]
summary = summarize_coefficients(results)
plot_coefficients_heatmap(summary)

# Convert coefficient dictionaries to a DataFrame
df_coef = pd.DataFrame(coef_list)

# Drop any terms that are missing in all models (i.e., full NaN columns)
df_coef = df_coef.dropna(axis=1, how='all')

# Optional: fill remaining NaNs with 0s (if you assume 0 coefficient for missing terms)
df_coef = df_coef.fillna(0)

# Compute the correlation matrix
corr_matrix = df_coef.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix of Symbolic Coefficients")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"fagproject/src/project/interpretability_plots/correlation_heatmap_e_{n_epochs}_k_{k}")
plt.close()