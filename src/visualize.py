import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Charger les données
df = pd.read_csv("data/dataset.csv")

X = df["x"]
y = df["y"]

# 2. Générer la fonction théorique sans bruit
x_true = np.linspace(-3, 3, 200)
y_true = 2 * x_true**2 + 3 * x_true + 5

# 3. Créer le graphique
plt.figure(figsize=(10, 6))

# Afficher les données avec bruit
plt.scatter(
    X, y,
    color="blue",
    s=60,
    label="Observed Data"
)

# Afficher la fonction théorique
plt.plot(
    x_true, y_true,
    color="red",
    linewidth=2,
    label="True Function"
)

# 4. Personnaliser le graphique
plt.title("Nonlinear Dataset")
plt.xlabel("X")
plt.ylabel("Y")

plt.legend()
plt.grid(True, alpha=0.3)

# 5. Enregistrer le graphique
plt.savefig("plots/nonlinear_dataset.png", dpi=150)

# 6. Afficher le graphique
plt.show()