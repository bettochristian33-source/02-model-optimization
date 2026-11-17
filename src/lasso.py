import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Charger les données
df = pd.read_csv("data/dataset.csv")

X = df[["x"]]
y = df["y"]

# 2. Séparer les données
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=8, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=8, random_state=42
)

# 3. Définir les hyperparamètres
degree = 15
ridge_alpha = 0.1
lasso_alpha = 0.5

# 4. Construire le modèle Ridge
model_ridge = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=ridge_alpha)
)

# 5. Construire le modèle Lasso
model_lasso = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    StandardScaler(),
    Lasso(
        alpha=lasso_alpha,
        max_iter=100000,
        tol=1e-6
    )
)

# 6. Entraîner les modèles
model_ridge.fit(X_train, y_train)
model_lasso.fit(X_train, y_train)

# 7. Comparer les performances
models = {
    "Ridge": model_ridge,
    "Lasso": model_lasso
}

results = []

for name, model in models.items():

    train_mse = mean_squared_error(
        y_train,
        model.predict(X_train)
    )

    val_mse = mean_squared_error(
        y_val,
        model.predict(X_val)
    )

    coefficients = model.steps[-1][1].coef_

    results.append({
        "Model": name,
        "Training MSE": train_mse,
        "Validation MSE": val_mse,
        "Active Features": np.count_nonzero(coefficients)
    })

results_df = pd.DataFrame(results)

print("--- MODEL COMPARISON ---")
print(results_df.round(4))

# 8. Analyser les coefficients de Lasso
lasso_coefficients = model_lasso.named_steps["lasso"].coef_

print("\n--- LASSO COEFFICIENTS ---")

for i, coef in enumerate(lasso_coefficients, start=1):
    print(f"x^{i:2d} : {coef:.6f}")

print("\n--- FEATURE SELECTION ---")
print("Original features :", degree)
print("Active features   :", np.count_nonzero(lasso_coefficients))
print("Zero coefficients :", np.sum(lasso_coefficients == 0))

# 9. Générer les courbes
X_curve = pd.DataFrame({
    "x": np.linspace(-3, 3, 500)
})

plt.figure(figsize=(12, 6))

plt.scatter(
    X_train["x"], y_train,
    color="blue",
    label="Training Data"
)

plt.scatter(
    X_val["x"], y_val,
    color="orange",
    label="Validation Data"
)

# Courbe Ridge
plt.plot(
    X_curve["x"],
    model_ridge.predict(X_curve),
    color="purple",
    linewidth=2,
    label="Ridge L2"
)

# Courbe Lasso
plt.plot(
    X_curve["x"],
    model_lasso.predict(X_curve),
    color="red",
    linewidth=2,
    label="Lasso L1"
)

# Fonction originale
plt.plot(
    X_curve["x"],
    2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
    color="green",
    linestyle="--",
    label="True Function"
)

plt.title("Ridge vs Lasso Regularization")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("plots/lasso_regression.png", dpi=150)

plt.show()