import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Configuration de la page
st.set_page_config(
    page_title="Model Optimization Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Machine Learning Optimization Lab")

st.write(
    "Explore Underfitting, Overfitting, "
    "Ridge, Lasso and Hyperparameter Optimization."
)

# 2. Charger les données
df = pd.read_csv("data/dataset.csv")

X = df[["x"]]
y = df["y"]

# 3. Séparer les données
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=8, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=8, random_state=42
)

# 4. Charger le meilleur modèle
best_model = joblib.load("models/best_model.joblib")

with open("models/metrics.json", "r") as file:
    metrics = json.load(file)

# 5. Créer les onglets
tab1, tab2, tab3 = st.tabs([
    "🧪 Model Playground",
    "🏆 Model Optimization",
    "🎯 Prediction"
])

# =====================================
# ONGLET 1 : MODEL PLAYGROUND
# =====================================

with tab1:

    st.header("Experiment with Machine Learning")

    # Paramètres interactifs
    col1, col2, col3 = st.columns(3)

    with col1:
        algorithm = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Ridge", "Lasso"]
        )

    with col2:
        degree = st.slider(
            "Polynomial Degree",
            min_value=1,
            max_value=15,
            value=2
        )

    with col3:
        alpha = st.select_slider(
            "Regularization Alpha",
            options=[0.01, 0.1, 0.5, 1.0, 10.0, 100.0],
            value=0.1,
            disabled=(algorithm == "Linear Regression")
        )

    # Sélectionner l'algorithme
    if algorithm == "Linear Regression":
        regressor = LinearRegression(tol=1e-12)

    elif algorithm == "Ridge":
        regressor = Ridge(alpha=alpha)

    else:
        regressor = Lasso(
            alpha=alpha,
            max_iter=100000,
            tol=1e-6
        )

    # Construire le pipeline
    model = make_pipeline(
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        StandardScaler(),
        regressor
    )

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Calculer les erreurs
    train_mse = mean_squared_error(
        y_train, model.predict(X_train)
    )

    val_mse = mean_squared_error(
        y_val, model.predict(X_val)
    )

    # Compter les caractéristiques actives
    coefficients = model.steps[-1][1].coef_
    active_features = np.count_nonzero(coefficients)

    # Afficher les indicateurs
    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Training MSE", f"{train_mse:.4f}")
    m2.metric("Validation MSE", f"{val_mse:.4f}")
    m3.metric("Polynomial Degree", degree)
    m4.metric("Active Features", int(active_features))

    # Générer les courbes
    X_curve = pd.DataFrame({
        "x": np.linspace(-3, 3, 500)
    })

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(
        X_train["x"], y_train,
        color="blue",
        label="Training Data"
    )

    ax.scatter(
        X_val["x"], y_val,
        color="orange",
        label="Validation Data"
    )

    ax.plot(
        X_curve["x"],
        model.predict(X_curve),
        color="red",
        linewidth=2,
        label="ML Model"
    )

    ax.plot(
        X_curve["x"],
        2 * X_curve["x"]**2 + 3 * X_curve["x"] + 5,
        color="green",
        linestyle="--",
        label="True Function"
    )

    ax.set_title(f"{algorithm} - Degree {degree}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    # Visualiser les coefficients
    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame({
        "Feature": [f"x^{i}" for i in range(1, degree + 1)],
        "Coefficient": coefficients
    })

    st.bar_chart(
        coef_df.set_index("Feature")
    )

    st.caption(
        "Coefficients correspond to standardized features."
    )

# =====================================
# ONGLET 2 : MODEL OPTIMIZATION
# =====================================

with tab2:

    st.header("GridSearchCV Results")

    results = pd.read_csv(
        "data/grid_search_results.csv"
    )

    columns = [
        "param_poly__degree",
        "param_regressor",
        "param_regressor__alpha",
        "CV MSE"
    ]

    ranking = results[columns].copy()

    ranking.columns = [
        "Degree",
        "Algorithm",
        "Alpha",
        "CV MSE"
    ]

    ranking = ranking.sort_values("CV MSE")

    st.success(
        "Best configuration selected by cross-validation"
    )

    st.dataframe(ranking.head(10))

    st.subheader("🏆 Final Model Performance")

    st.caption(
        "Fixed evaluation of the saved model "
        "on 8 held-out test observations."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("MAE", f"{metrics['MAE']:.4f}")
    c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
    c3.metric("R²", f"{metrics['R2']:.4f}")
    c4.metric("CV MSE", f"{metrics['CV_MSE']:.4f}")

    st.subheader("Actual vs Predicted")

    predictions = pd.read_csv(
        "data/test_predictions.csv"
    )

    st.dataframe(predictions)

# =====================================
# ONGLET 3 : PREDICTION
# =====================================

with tab3:

    st.header("Predict with the Saved Model")

    st.write(
        "Use the optimized model without retraining."
    )

    x_value = st.slider(
        "Select X",
        min_value=-3.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    new_data = pd.DataFrame({
        "x": [x_value]
    })

    prediction = best_model.predict(new_data)[0]

    st.metric(
        "Predicted Y",
        f"{prediction:.4f}"
    )

    # Visualiser la prédiction
    X_curve = pd.DataFrame({
        "x": np.linspace(-3, 3, 500)
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        X_curve["x"],
        best_model.predict(X_curve),
        color="red",
        label="Optimized Model"
    )

    ax.scatter(
        x_value,
        prediction,
        color="green",
        s=150,
        label="New Prediction"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "The model was loaded from best_model.joblib."
    )

# 6. Pied de page
st.divider()

st.caption(
    "Project 02 | Machine Learning Optimization | "
    "Python, Scikit-learn and Streamlit"
)

st.warning(
    "Educational project based on 40 synthetic observations."
)