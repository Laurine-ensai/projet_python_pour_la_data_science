import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

def diagnostic_cle_jointure(df, col, nom_df):   
    total = len(df)
    nb_nan = df[col].isna().sum()
    nb_non_nan = total - nb_nan
    nb_uniques = df[col].nunique()
    nb_uniques_sans_nan = df[col].dropna().nunique()

    print(f"--- Diagnostic clé de jointure : {col} ({nom_df}) ---")
    print(f"Type de la colonne           : {df[col].dtype}")
    print(f"Nb lignes total              : {total}")
    print(f"Nb valeurs manquantes (NaN)  : {nb_nan}")
    print(f"Nb valeurs non nulles        : {nb_non_nan}")
    print(f"Nb valeurs uniques (total)   : {nb_uniques}")
    print(f"Nb valeurs uniques (sans NaN): {nb_uniques_sans_nan}")

    # distribution des longueurs
    print("\nDistribution des longueurs :")
    longueurs = df[col].dropna().astype(str).str.len().value_counts().sort_index()
    print(longueurs)

    # check clé unique
    if nb_uniques_sans_nan == nb_non_nan:
        print("\nClé UNIQUE (hors NaN)")
    else:
        print("\n!!!! Clé NON unique !!!!")

    # check présence NaN
    if nb_nan > 0:
        print("!!!! Présence de valeurs manquantes !!!!")


def creer_gdf_irve(df, long_col, lat_col, crs="EPSG:4326"):
    df = df.copy()
    df["geometry"] = df.apply(
        lambda row: Point(row[long_col], row[lat_col]),
        axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)  
    return gdf


def joindre_communes(gdf_irve, communes):
    communes = communes.to_crs(gdf_irve.crs)  
    gdf_result = gpd.sjoin(
        gdf_irve,
        communes[['INSEE_COM', 'NOM', 'geometry']],
        how="left",
        predicate="within"
    )
    return gdf_result


def ajouter_codes_geo(df_irve, gdf_result, var="total"):
    """
    var : vaut both, manq ou total
    """
    if var not in ["both", "manq", "total"]:
        raise ValueError("L'argument 'var' doit être 'both', 'manq' ou 'total'.")
    df_irve = df_irve.copy()
    if var != "total":
        df_irve["code_geo_manquant"] = df_irve["code_insee_commune"].fillna(gdf_result["INSEE_COM"])
        df_irve["nom_commune"] = gdf_result["NOM"]
    if var != "manq":
        df_irve["code_geo_total"] = gdf_result["INSEE_COM"]
    return df_irve


def afficher_matrice_correlation(df, method='spearman', figsize=(14, 10)):
    """
    Calcule et affiche la heatmap des corrélations.
    """
    # On ne garde que les colonnes numériques et on drop les NaN pour le calcul
    df_num = df.select_dtypes(include=['number']).dropna()
    corr_matrix = df_num.corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title(f"Matrice de corrélation ({method})", fontsize=14)
    plt.xticks(rotation=75, ha='right')
    plt.tight_layout()
    plt.show()
   
    return corr_matrix


def identifier_fortes_correlations(corr_matrix, threshold=0.70):
    """
    Retourne une liste des paires de variables fortement corrélées.
    """
    fortes_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                fortes_corr.append({
                    'v1': corr_matrix.columns[i],
                    'v2': corr_matrix.columns[j],
                    'val': val
                })
    return fortes_corr


def analyser_recouvrement_cles(set_a, set_b, nom_a, nom_b):
    """Affiche le bilan du recouvrement entre deux ensembles de codes géo."""
    manquants_dans_b = len(set_a - set_b)
    manquants_dans_a = len(set_b - set_a)
    print(f"--- Comparaison {nom_a} vs {nom_b} ---")
    print(f"Codes de {nom_a} absents dans {nom_b} : {manquants_dans_b}")
    print(f"Codes de {nom_b} absents dans {nom_a} : {manquants_dans_a}\n")
    