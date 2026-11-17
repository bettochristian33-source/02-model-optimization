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

# 3. Définir les modèles
degrees = [1, 2, 15]

results = []

X_curve = pd.DataFrame({
    "x": np.linspace(-3, 3, 500)
})

# 4. Créer les graphiques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 5. Entraîner et comparer les modèles
for ax, degree in zip(axes, degrees):

    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression(tol=1e-12)
    )

    model.fit(X_train, y_train)

    train_mse = mean_squared_error(
        y_train, model.predict(X_train)
    )

    val_mse = mean_squared_error(
        y_val, model.predict(X_val)
    )

    results.append({
        "Degree": degree,
        "Training MSE": train_mse,
        "Validation MSE": val_mse
    })

    # Données d'entraînement
    ax.scatter(
        X_train["x"], y_train,
        color="blue",
        label="Training"
    )

    # Données de validation
    ax.scatter(
        X_val["x"], y_val,
        color="orange",
        label="Validation"
    )

    # Modèle appris
    ax.plot(
        X_curve["x"],
        model.predict(X_curve),
        color="red",
        label="Model"
    )

    # Fonction originale
    ax.plot(
        X_curve["x"],
        2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
        color="green",
        linestyle="--",
        label="True Function"
    )

    ax.set_title(f"Degree {degree}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)

# 6. Afficher les résultats
results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.round(4))

# 7. Enregistrer les résultats
results_df.to_csv(
    "data/model_comparison.csv",
    index=False
)

# 8. Enregistrer les graphiques
plt.tight_layout()

plt.savefig(
    "plots/model_comparison.png",
    dpi=150
)

plt.show()