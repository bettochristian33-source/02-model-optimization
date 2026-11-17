import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline

# 1. Charger les données
df = pd.read_csv("data/dataset.csv")

X = df[["x"]]
y = df["y"]

# 2. Réserver les données de test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=8, random_state=42
)

# 3. Construire notre pipeline
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# 4. Définir les hyperparamètres à tester
param_grid = [
    {
        "poly__degree": [1, 2, 3, 5],
        "regressor": [LinearRegression()]
    },
    {
        "poly__degree": [2, 5, 10, 15],
        "regressor": [Ridge()],
        "regressor__alpha": [0.01, 0.1, 1, 10]
    },
    {
        "poly__degree": [2, 5, 10, 15],
        "regressor": [
            Lasso(max_iter=100000, tol=1e-6)
        ],
        "regressor__alpha": [0.1, 0.5, 1, 2]
    }
]

# 5. Configurer la validation croisée
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 6. Configurer GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_mean_squared_error",
    verbose=1
)

# 7. Lancer la recherche
grid_search.fit(X_dev, y_dev)

# 8. Afficher le meilleur modèle
print("\n--- BEST MODEL ---")
print(grid_search.best_params_)

print("\n--- BEST CV SCORE ---")
print(f"MSE : {-grid_search.best_score_:.4f}")

# 9. Récupérer tous les résultats
results = pd.DataFrame(grid_search.cv_results_)

results["CV MSE"] = -results["mean_test_score"]

results = results.sort_values("CV MSE")

# 10. Afficher les cinq meilleurs modèles
columns = [
    "param_poly__degree",
    "param_regressor",
    "param_regressor__alpha",
    "CV MSE"
]

print("\n--- TOP 5 MODELS ---")
print(results[columns].head(5).to_string(index=False))

# 11. Enregistrer les résultats
results.to_csv(
    "data/grid_search_results.csv",
    index=False
)

print("\nOptimization completed successfully!")

# 12. Importer les bibliothèques nécessaires
import joblib
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 13. Récupérer le meilleur modèle entraîné
best_model = grid_search.best_estimator_

# 14. Prédire les données de test
y_test_pred = best_model.predict(X_test)

# 15. Calculer les métriques finales
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

print("\n--- FINAL TEST EVALUATION ---")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# 16. Sauvegarder le modèle
joblib.dump(
    best_model,
    "models/best_model.joblib"
)

print("\nModel saved successfully!")

# 17. Sauvegarder les métriques
metrics = {
    "MAE": float(mae),
    "MSE": float(mse),
    "RMSE": float(rmse),
    "R2": float(r2),
    "CV_MSE": float(-grid_search.best_score_)
}

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully!")

# 18. Enregistrer les prédictions
predictions = pd.DataFrame({
    "x": X_test["x"],
    "Actual": y_test,
    "Predicted": y_test_pred
})

predictions.to_csv(
    "data/test_predictions.csv",
    index=False
)

print("\n--- TEST PREDICTIONS ---")
print(predictions.round(4))