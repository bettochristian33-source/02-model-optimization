import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

# 3. Créer un modèle trop simple
model = LinearRegression()

# 4. Entraîner le modèle
model.fit(X_train, y_train)

# 5. Faire les prédictions
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# 6. Calculer les erreurs
train_mse = mean_squared_error(y_train, y_train_pred)
val_mse = mean_squared_error(y_val, y_val_pred)

# 7. Afficher les résultats
print("--- UNDERFITTING ---")
print("Degree : 1")
print(f"Training MSE   : {train_mse:.4f}")
print(f"Validation MSE : {val_mse:.4f}")

# 8. Générer la droite du modèle
X_curve = pd.DataFrame({
    "x": np.linspace(-3, 3, 300)
})

y_curve = model.predict(X_curve)

# 9. Visualiser le modèle
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

plt.plot(
    X_curve["x"], y_curve,
    color="red",
    linewidth=2,
    label="Underfitting Model"
)

# Fonction théorique
plt.plot(
    X_curve["x"],
    2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
    color="green",
    linestyle="--",
    label="True Function"
)

plt.title("Underfitting - Degree 1")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("plots/underfitting.png", dpi=150)

plt.show()