import numpy as np
import pandas as pd

# 1. Initialiser le générateur aléatoire
rng = np.random.default_rng(42)

# 2. Générer 40 observations
X = np.linspace(-3, 3, 40)

# 3. Générer du bruit aléatoire
noise = rng.normal(0, 2.5, size=40)

# 4. Créer une relation non linéaire
y = 2 * X**2 + 3 * X + 5 + noise

# 5. Créer le dataset
df = pd.DataFrame({
    "x": X,
    "y": y
})

# 6. Afficher les premières observations
print("--- DATASET ---")
print(df.head())

# 7. Afficher les dimensions
print("\n--- DIMENSIONS ---")
print(df.shape)

# 8. Enregistrer le dataset
df.to_csv("data/dataset.csv", index=False)

print("\nDataset créé avec succès !")