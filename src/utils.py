import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

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

    print("--------------------------------------------------\n")

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
    df_irve = df_irve.copy()
    if var != "total":
        df_irve["code_geo_manquant"] = df_irve["code_insee_commune"].fillna(gdf_result["INSEE_COM"])
        df_irve["nom_commune"] = gdf_result["NOM"]
    if var != "manq" :
        df_irve["code_geo_total"] = gdf_result["INSEE_COM"]
    return df_irve

def compter_valeurs_manquantes(df, colonnes):
    return {col: df[col].isna().sum() for col in colonnes}

def compter_uniques(df, colonnes):
    return {col: df[col].nunique() for col in colonnes}


