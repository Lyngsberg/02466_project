from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from fagproject/src/project/train.py import train

# 1. Load data
X, y = load_iris(return_X_y=True)

# 2. Define parameter grid manually
param_grid = [10, 50, 100]

# 3. Set up outer and inner folds
outer_k = 5
inner_k = 3
outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=1)

# 4. Store results
outer_results = []

# 5. Outer loop
for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n--- Outer Fold {outer_fold+1}/{outer_k} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_score = -np.inf
    best_param = None

    # 6. Inner loop
    inner_cv = KFold(n_splits=inner_k, shuffle=True, random_state=1)
    for n_est in param_grid:
        inner_scores = []

        for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
            X_inner_train, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            y_inner_train, y_val = y_train[inner_train_idx], y_train[inner_val_idx]

            model = RandomForestClassifier(n_estimators=n_est, random_state=1)
            model.fit(X_inner_train, y_inner_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            inner_scores.append(acc)

        avg_inner_score = np.mean(inner_scores)
        print(f"  n_estimators={n_est}: Inner CV mean accuracy = {avg_inner_score:.4f}")

        if avg_inner_score > best_score:
            best_score = avg_inner_score
            best_param = n_est

    print(f"Best param from inner CV: n_estimators = {best_param}")

    # 7. Retrain on full training set and test on outer test set
    final_model = RandomForestClassifier(n_estimators=best_param, random_state=1)
    final_model.fit(X_train, y_train)
    y_final_pred = final_model.predict(X_test)
    outer_acc = accuracy_score(y_test, y_final_pred)
    outer_results.append(outer_acc)

    print(f"Outer test accuracy: {outer_acc:.4f}")

# 8. Final results
print("\nFinal Nested CV Results:")
print("Accuracies per outer fold:", outer_results)
print("Mean accuracy:", np.mean(outer_results))
