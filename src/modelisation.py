import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def modelisation_lasso(df, features, target, test_size=0.2, random_state=42):
    """
    Entraîne un modèle Lasso avec validation croisée pour identifier les déterminants linéaires.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardisation cruciale pour le Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entraînement du LassoCV
    model = LassoCV(cv=5, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    # Prédictions et métriques
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Extraction des coefficients
    coeffs = pd.Series(model.coef_, index=features).sort_values(ascending=False)

    return model, metrics, coeffs


def modelisation_xgboost(df, features, target, test_size=0.2, random_state=42, params=None):
    """
    Entraîne un modèle XGBoost pour capturer les relations non-linéaires.
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if params is None:
        params = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'random_state': random_state
        }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Prédictions et métriques
    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    return model, metrics, importance