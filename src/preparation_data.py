from shapely.geometry import Point
import geopandas as gpd
from cartiflette import carti_download
import pandas as pd


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


def nettoyer_code_insee(val):
    # Si valeur manquante ou "0.0" → NaN
    if pd.isna(val) or str(val) in ["0.0", "0"]:
        return pd.NA

    val = str(val)

    # Supprimer le ".0" à la fin
    if val.endswith(".0"):
        val = val[:-2]

    # Si longueur = 4 → ajouter un 0 devant
    if len(val) == 4:
        val = "0" + val

    # Si longueur = 5 → OK
    if len(val) == 5:
        return val

    # Sinon → valeur incohérente → NaN
    return pd.NA


def creer_gdf_irve(df, long_col, lat_col, crs="EPSG:4326"):
    df = df.copy()
    df["geometry"] = df.apply(
        lambda row: Point(row[long_col], row[lat_col]),
        axis=1
    )
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)  
    return gdf


def charger_communes(departements=None, crs=4326, year=2022):
    if departements is None:
        departements = [str(i).zfill(2) for i in range(1, 96)]
    communes = carti_download(
        crs=crs,
        borders="COMMUNE",
        filter_by="DEPARTEMENT",
        values=departements,
        vectorfile_format="geojson",
        source="EXPRESS-COG-CARTO-TERRITOIRE",
        year=year
    )
    return communes


def joindre_communes(gdf_irve, communes):
    communes = communes.to_crs(gdf_irve.crs)  
    gdf_result = gpd.sjoin(
        gdf_irve,
        communes[['INSEE_COM', 'NOM', 'geometry']],
        how="left",
        predicate="within"
    )
    return gdf_result


def ajouter_codes_geo(df_irve, gdf_result):
    df_irve = df_irve.copy()
    df_irve["code_geo_manquant"] = df_irve["code_insee_commune"].fillna(gdf_result["INSEE_COM"])
    df_irve["code_geo_total"] = gdf_result["INSEE_COM"]
    df_irve["nom_commune"] = gdf_result["NOM"]
    return df_irve




def compter_valeurs_manquantes(df, colonnes):
    return {col: df[col].isna().sum() for col in colonnes}


def compter_uniques(df, colonnes):
    return {col: df[col].nunique() for col in colonnes}


def corriger_codes_incoherents(df, codes_manquants_communs):
    """
    Remplace le code_geo_manquant par code_geo_total pour les lignes où :
    1. Le code actuel fait partie des codes identifiés comme problématiques.
    2. Les noms de communes (consolidated et nom issu du géocodage) concordent.
    """
    mask = (
        df["code_geo_manquant"].isin(codes_manquants_communs)
        & df["consolidated_commune"].notna()
        & df["nom_commune"].notna()
        & (df["consolidated_commune"] == df["nom_commune"])
    )
    df.loc[mask, "code_geo_manquant"] = df.loc[mask, "code_geo_total"]
    print(f"Correction appliquée sur {mask.sum()} lignes.")
    return df


def corriger_par_nom(df, codes_a_corriger):
    """
    Corrige les codes geo manquants en utilisant le code (issu du géocodage)
    associé au nom de la commune.
    """
    mapping_commune_code = (
        df.dropna(subset=["nom_commune", "code_geo_total"])
            .groupby("nom_commune")["code_geo_total"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
    )
    mask = (
        df["code_geo_manquant"].isin(codes_a_corriger)
        & df["consolidated_commune"].notna()
        & df["consolidated_commune"].isin(mapping_commune_code.index)
    )
    df.loc[mask, "code_geo_manquant"] = df.loc[mask, "consolidated_commune"].map(mapping_commune_code) 
    print(f"Correction par nom de commune appliquée sur {mask.sum()} lignes.")
    return df


def corriger_conflit_code_postal(df, codes_a_corriger):
    """
    Identifie et corrige les cas où le code_geo_manquant est en réalité 
    un code postal au lieu d'un code INSEE.
    """
    mask = (
        df["code_geo_manquant"].isin(codes_a_corriger) & 
        df["consolidated_code_postal"].notna()
    )
    # On vérifie si le code_geo_manquant est identique au code postal
    # Si oui, on privilégie le code_geo_total (code INSEE issu du géocodage)
    conflit_mask = mask & (df["code_geo_manquant"] == df["consolidated_code_postal"])
    nb_corrections = conflit_mask.sum()
    df.loc[conflit_mask, "code_geo_manquant"] = df.loc[conflit_mask, "code_geo_total"]
    print(f"Correction de conflit Code Postal appliquée sur {nb_corrections} lignes.")
    return df

