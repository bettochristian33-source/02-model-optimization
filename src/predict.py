import joblib
import pandas as pd

# 1. Charger le modèle sauvegardé
model = joblib.load("models/best_model.joblib")

# 2. Préparer une nouvelle observation
new_data = pd.DataFrame({
    "x": [2.0]
})

# 3. Faire une prédiction
prediction = model.predict(new_data)

# 4. Afficher le résultat
print("--- SAVED MODEL PREDICTION ---")
print("X : 2.0")
print(f"Predicted Y : {prediction[0]:.4f}")