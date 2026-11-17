import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
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
alpha = 0.1

# 4. Modèle sans régularisation
model_original = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    StandardScaler(),
    LinearRegression(tol=1e-12)
)

# 5. Modèle avec régularisation L2
model_ridge = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    StandardScaler(),
    Ridge(alpha=alpha)
)

# 6. Entraîner les deux modèles
model_original.fit(X_train, y_train)
model_ridge.fit(X_train, y_train)

# 7. Comparer les performances
models = {
    "Without Regularization": model_original,
    "Ridge Regression": model_ridge
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

    results.append({
        "Model": name,
        "Training MSE": train_mse,
        "Validation MSE": val_mse
    })

results_df = pd.DataFrame(results)

print("--- RIDGE REGRESSION ---")
print(f"Degree : {degree}")
print(f"Alpha  : {alpha}")

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4))

# 8. Générer les courbes
X_curve = pd.DataFrame({
    "x": np.linspace(-3, 3, 500)
})

# 9. Visualiser les données
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

# Modèle sans régularisation
plt.plot(
    X_curve["x"],
    model_original.predict(X_curve),
    color="red",
    label="Without Regularization"
)

# Modèle Ridge
plt.plot(
    X_curve["x"],
    model_ridge.predict(X_curve),
    color="purple",
    linewidth=2,
    label="Ridge Regression"
)

# Fonction originale
plt.plot(
    X_curve["x"],
    2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
    color="green",
    linestyle="--",
    label="True Function"
)

plt.title("Ridge Regression - L2 Regularization")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)

# 10. Enregistrer le graphique
plt.savefig("plots/ridge_regression.png", dpi=150)

plt.show()