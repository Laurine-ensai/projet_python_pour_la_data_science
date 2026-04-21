import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def random_forest_depuis_notebook(
    df: pd.DataFrame,
    target,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth=None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    plot_importances: bool = True,
    compute_shap: bool = False,
    plot_shap: bool = False,
):
    """
    Fonction Random Forest réutilisable avec importance + SHAP.

    Paramètres
    ----------
    df : DataFrame
    target : str ou Series

    Returns
    -------
    dict avec résultats + SHAP
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame.")

    # ===== Target =====
    if isinstance(target, str):
        if target not in df.columns:
            raise ValueError(f"La target '{target}' est absente du DataFrame.")
        y = df[target].copy()
        X = df.drop(columns=[target]).copy()
        target_name = target
    elif isinstance(target, pd.Series):
        y = target.copy()
        X = df.copy()
        target_name = getattr(target, "name", "target")
    else:
        raise TypeError("target doit être soit un nom de colonne (str), soit une Series.")

    # ===== Variables numériques uniquement =====
    X = X.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError("Aucune variable numérique exploitable.")

    # ===== Suppression NA target =====
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if len(X) < 2:
        raise ValueError("Pas assez d'observations.")

    # ===== Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # ===== Modèle =====
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs
    )

    rf.fit(X_train, y_train)

    # ===== Scores =====
    score_train = rf.score(X_train, y_train)
    score_test = rf.score(X_test, y_test)

    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ===== Importance classique =====
    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values()

    if plot_importances:
        plt.figure(figsize=(8, 6))
        importances.plot(kind="barh")
        plt.title("Importance des variables - Random Forest")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()

    # ===== SHAP =====
    shap_values = None
    shap_importance = None

    if compute_shap or plot_shap:
        try:
            import shap

            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)

            # Tableau importance SHAP
            shap_importance = pd.DataFrame({
                "variable": X_test.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0)
            }).sort_values("mean_abs_shap", ascending=False)

            # Graphe type beeswarm (comme ton image)
            if plot_shap:
                shap.summary_plot(shap_values, X_test, plot_type="dot")

        except Exception as e:
            print(f"SHAP non calculé : {e}")

    # ===== Tableau prédictions =====
    df_result = pd.DataFrame({
        "réel": y_test,
        "prédit": y_pred
    })

    # ===== Return =====
    return {
        "target_name": target_name,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": rf,
        "y_pred": y_pred,
        "score_train": score_train,
        "score_test": score_test,
        "rmse": rmse,
        "r2": r2,
        "feature_importances": importances,
        "shap_values": shap_values,
        "shap_importance": shap_importance,
        "df_result": df_result,
    }