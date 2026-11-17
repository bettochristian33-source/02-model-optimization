import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# 3. Définir un modèle complexe
degree = 15

model = make_pipeline(
    PolynomialFeatures(
        degree=degree,
        include_bias=False
    ),
    LinearRegression(tol=1e-12)
)

# 4. Entraîner le modèle
model.fit(X_train, y_train)

# 5. Faire les prédictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# 6. Calculer les erreurs
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# 7. Afficher les résultats
print("--- OVERFITTING ---")
print(f"Degree : {degree}")
print(f"Training MSE   : {train_mse:.4f}")
print(f"Validation MSE : {val_mse:.4f}")

# 8. Générer la courbe du modèle
X_curve = pd.DataFrame({
    "x": np.linspace(-3, 3, 300)
})

y_curve = model.predict(X_curve)

# 9. Visualiser les données
plt.figure(figsize=(10, 6))

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

# Modèle complexe
plt.plot(
    X_curve["x"], y_curve,
    color="red",
    linewidth=2,
    label="Overfitting Model"
)

# Fonction théorique
plt.plot(
    X_curve["x"],
    2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
    color="green",
    linestyle="--",
    label="True Function"
)

plt.title("Overfitting - Degree 15")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)

# 10. Enregistrer le graphique
plt.savefig("plots/overfitting.png", dpi=150)

plt.show()