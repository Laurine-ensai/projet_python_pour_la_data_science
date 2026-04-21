import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def regression_lasso_depuis_notebook(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
    log_target: bool = True,
    save_selected_path: str | None = None
) -> dict:
    """
    Reprend le traitement et la régression Lasso du notebook dans une seule fonction.

    Paramètres
    ----------
    df : pd.DataFrame
        Base de données d'entrée.
    target : str
        Nom de la variable cible.
    test_size : float, default=0.2
        Proportion du jeu de test.
    random_state : int, default=42
        Graine aléatoire pour le split.
    log_target : bool, default=True
        Applique np.log1p à la target comme dans le notebook.
    save_selected_path : str | None, default=None
        Chemin optionnel pour sauvegarder les variables sélectionnées en CSV.

    Retour
    ------
    dict
        Dictionnaire contenant les données préparées, le pipeline, les coefficients,
        les variables sélectionnées et les métriques.
    """

    if target not in df.columns:
        raise ValueError(f"La target '{target}' est absente du DataFrame.")

    df = df.copy()
    df = df[df[target].notna()].copy()

    # Nettoyage / préparation géographique
    if "CODGEO" in df.columns:
        df["CODGEO"] = df["CODGEO"].astype(str).str.strip()

        geo_cols = [col for col in df.columns if col.startswith("departement_") or col.startswith("region_")]
        df = df.drop(columns=geo_cols, errors="ignore")
        df = df.drop(columns=["departement", "region"], errors="ignore")

        df["departement"] = df["CODGEO"].str[:2]
        df.loc[df["CODGEO"].str.startswith("2A"), "departement"] = "2A"
        df.loc[df["CODGEO"].str.startswith("2B"), "departement"] = "2B"

        dept_to_region = {'01': 'Auvergne-Rhone-Alpes',
 '02': 'Hauts-de-France',
 '03': 'Auvergne-Rhone-Alpes',
 '04': "Provence-Alpes-Cote d'Azur",
 '05': "Provence-Alpes-Cote d'Azur",
 '06': "Provence-Alpes-Cote d'Azur",
 '07': 'Auvergne-Rhone-Alpes',
 '08': 'Grand Est',
 '09': 'Occitanie',
 '10': 'Grand Est',
 '11': 'Occitanie',
 '12': 'Occitanie',
 '13': "Provence-Alpes-Cote d'Azur",
 '14': 'Normandie',
 '15': 'Auvergne-Rhone-Alpes',
 '16': 'Nouvelle-Aquitaine',
 '17': 'Nouvelle-Aquitaine',
 '18': 'Centre-Val de Loire',
 '19': 'Nouvelle-Aquitaine',
 '21': 'Bourgogne-Franche-Comte',
 '22': 'Bretagne',
 '23': 'Nouvelle-Aquitaine',
 '24': 'Nouvelle-Aquitaine',
 '25': 'Bourgogne-Franche-Comte',
 '26': 'Auvergne-Rhone-Alpes',
 '27': 'Normandie',
 '28': 'Centre-Val de Loire',
 '29': 'Bretagne',
 '2A': 'Corse',
 '2B': 'Corse',
 '30': 'Occitanie',
 '31': 'Occitanie',
 '32': 'Occitanie',
 '33': 'Nouvelle-Aquitaine',
 '34': 'Occitanie',
 '35': 'Bretagne',
 '36': 'Centre-Val de Loire',
 '37': 'Centre-Val de Loire',
 '38': 'Auvergne-Rhone-Alpes',
 '39': 'Bourgogne-Franche-Comte',
 '40': 'Nouvelle-Aquitaine',
 '41': 'Centre-Val de Loire',
 '42': 'Auvergne-Rhone-Alpes',
 '43': 'Auvergne-Rhone-Alpes',
 '44': 'Pays de la Loire',
 '45': 'Centre-Val de Loire',
 '46': 'Occitanie',
 '47': 'Nouvelle-Aquitaine',
 '48': 'Occitanie',
 '49': 'Pays de la Loire',
 '50': 'Normandie',
 '51': 'Grand Est',
 '52': 'Grand Est',
 '53': 'Pays de la Loire',
 '54': 'Grand Est',
 '55': 'Grand Est',
 '56': 'Bretagne',
 '57': 'Grand Est',
 '58': 'Bourgogne-Franche-Comte',
 '59': 'Hauts-de-France',
 '60': 'Hauts-de-France',
 '61': 'Normandie',
 '62': 'Hauts-de-France',
 '63': 'Auvergne-Rhone-Alpes',
 '64': 'Nouvelle-Aquitaine',
 '65': 'Occitanie',
 '66': 'Occitanie',
 '67': 'Grand Est',
 '68': 'Grand Est',
 '69': 'Auvergne-Rhone-Alpes',
 '70': 'Bourgogne-Franche-Comte',
 '71': 'Bourgogne-Franche-Comte',
 '72': 'Pays de la Loire',
 '73': 'Auvergne-Rhone-Alpes',
 '74': 'Auvergne-Rhone-Alpes',
 '75': 'Ile-de-France',
 '76': 'Normandie',
 '77': 'Ile-de-France',
 '78': 'Ile-de-France',
 '79': 'Nouvelle-Aquitaine',
 '80': 'Hauts-de-France',
 '81': 'Occitanie',
 '82': 'Occitanie',
 '83': "Provence-Alpes-Cote d'Azur",
 '84': "Provence-Alpes-Cote d'Azur",
 '85': 'Pays de la Loire',
 '86': 'Nouvelle-Aquitaine',
 '87': 'Nouvelle-Aquitaine',
 '88': 'Grand Est',
 '89': 'Bourgogne-Franche-Comte',
 '90': 'Bourgogne-Franche-Comte',
 '91': 'Ile-de-France',
 '92': 'Ile-de-France',
 '93': 'Ile-de-France',
 '94': 'Ile-de-France',
 '95': 'Ile-de-France',
 '971': 'Outre-Mer',
 '972': 'Outre-Mer',
 '973': 'Outre-Mer',
 '974': 'Outre-Mer',
 '976': 'Outre-Mer'}

        df["departement_long"] = df["CODGEO"].str[:3]
        drom_mask = df["departement_long"].isin(["971", "972", "973", "974", "976"])
        df.loc[drom_mask, "departement"] = df.loc[drom_mask, "departement_long"]

        df["region"] = df["departement"].map(dept_to_region).fillna("Autre")
        df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

    # Suppression de colonnes non exploitables / textuelles
    df_propre = df.drop(columns=[
        "CODGEO",
        "top_operateur",
        "Libellé géographique",
        "departement",
        "departement_long"
    ], errors="ignore").copy()

    # Imputation ciblée comme dans le notebook
    revenu_col = "[DISP] Médiane (€)"
    pop_col = "[DISP] Nbre de personnes dans les ménages fiscaux"

    if revenu_col in df_propre.columns:
        df_propre["revenu_manquant"] = df_propre[revenu_col].isna().astype(int)
        df_propre[revenu_col] = df_propre[revenu_col].fillna(df_propre[revenu_col].median())

    if pop_col in df_propre.columns:
        df_propre["population_manquante"] = df_propre[pop_col].isna().astype(int)
        df_propre[pop_col] = df_propre[pop_col].fillna(df_propre[pop_col].median())

    # Vérification finale
    object_cols = df_propre.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise ValueError(
            "Il reste des colonnes de type object après préparation : "
            f"{object_cols}. Il faut les encoder ou les supprimer."
        )

    X = df_propre.drop(columns=[target])
    y = np.log1p(df_propre[target]) if log_target else df_propre[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=5, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    lasso = pipeline.named_steps["lasso"]
    coef = pd.Series(lasso.coef_, index=X.columns)

    selected_features = coef[coef != 0].index.tolist()
    removed_features = coef[coef == 0].index.tolist()
    df_selected = X[selected_features].copy()

    if save_selected_path is not None:
        df_selected.to_csv(save_selected_path, index=False)

    results = {
        "df_prepare": df_propre,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "lasso": lasso,
        "coefficients": coef,
        "alpha_optimal": lasso.alpha_,
        "score_train": pipeline.score(X_train, y_train),
        "score_test": pipeline.score(X_test, y_test),
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "y_pred": y_pred,
        "selected_features": selected_features,
        "removed_features": removed_features,
        "df_selected": df_selected
    }

    if log_target:
        y_test_exp = np.expm1(y_test)
        y_pred_exp = np.expm1(y_pred)
        results["mse_original_scale"] = mean_squared_error(y_test_exp, y_pred_exp)
        results["r2_original_scale"] = r2_score(y_test_exp, y_pred_exp)

    return results
