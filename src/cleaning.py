import pandas as pd

###### 1. NETTOYAGE BAS NIVEAU ######


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


def reparer_encodage_implantation(df):
    """
    Répare les caractères spéciaux mal encodés dans la colonne implantation_station.
    """
    replacements = {
        "Parking priv rserv  la clientle": "Parking privé réservé à la clientèle",
        "Parking priv  usage public": "Parking privé à usage public",
    }

    df = df.copy()
    if "implantation_station" in df.columns:
        df["implantation_station"] = (
            df["implantation_station"].replace(replacements, regex=True).str.strip()
        )
    return df


def regrouper_implantation_station(df):
    """
    Simplifie les catégories d'implantation pour la modélisation.
    """
    mapping = {
        "Parking privé réservé à la clientèle": "prive",
        "Parking privé à usage public": "prive",
        "Parking public": "public",
        "Voirie": "voirie",
        "Station dédiée à la recharge rapide": "rapide",
    }

    df = df.copy()
    if "implantation_station" in df.columns:
        df["implantation_station_clean"] = df["implantation_station"].replace(mapping)
        # On s'assure que les valeurs non mappées ne créent pas de problèmes
        df["implantation_station_clean"] = df["implantation_station_clean"].astype(
            "category"
        )

    return df


###### 2. STANDARDISATION DES VARIABLES ######
def standardize_all_codes(dfs_dict):
    """
    Applique le nettoyage du code INSEE sur tous les DataFrames fournis.

    Args:
        dfs_dict (dict): Dictionnaire { "nom_df": (dataframe, "nom_colonne_insee") }
    Returns:
        dict: Dictionnaire des DataFrames nettoyés
    """
    cleaned_dfs = {}
    for name, (df, col) in dfs_dict.items():
        print(f"Standardisation du code INSEE pour : {name} (colonne : {col})")
        df = df.copy()
        df["code_geo"] = df[col].apply(nettoyer_code_insee)
        cleaned_dfs[name] = df
    return cleaned_dfs


def clean_irve_variables_finales(df):
    """
    Nettoie uniquement les variables finales retenues
    et retourne un DataFrame prêt pour agrégation territoriale.
    """
    df = df.copy()

    # Variables finales conservées
    vars_finales = [
        "code_geo_total",
        "nom_operateur",
        "implantation_station_clean",
        "nbre_pdc",
        "puissance_nominale",
        "prise_type_ef",
        "prise_type_2",
        "prise_type_combo_ccs",
        "paiement_cb",
        "paiement_autre",
    ]
    df = df[vars_finales]

    # Booléens
    cols_bool = [
        "prise_type_ef",
        "prise_type_2",
        "prise_type_combo_ccs",
        "paiement_cb",
        "paiement_autre",
    ]
    mapping = {"true": True, "false": False, "1": True, "0": False}
    for col in cols_bool:
        df[col] = (
            df[col].astype(str).str.strip().str.lower().map(mapping).astype("boolean")
        )

    # Strings
    df["nom_operateur"] = (
        df["nom_operateur"].astype("string").fillna("inconnu").str.strip()
    )

    # Numériques
    df["nbre_pdc"] = pd.to_numeric(df["nbre_pdc"], errors="coerce")
    df["puissance_nominale"] = pd.to_numeric(df["puissance_nominale"], errors="coerce")

    # Variable recharge rapide
    df["borne_rapide"] = df["puissance_nominale"] >= 43

    return df


###### 3. CORRECTIONS GÉOGRAPHIQUES ######


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
    df.loc[mask, "code_geo_manquant"] = df.loc[mask, "consolidated_commune"].map(
        mapping_commune_code
    )
    print(f"Correction par nom de commune appliquée sur {mask.sum()} lignes.")
    return df


def corriger_conflit_code_postal(df, codes_a_corriger):
    """
    Identifie et corrige les cas où le code_geo_manquant est en réalité
    un code postal au lieu d'un code INSEE.
    """
    mask = (
        df["code_geo_manquant"].isin(codes_a_corriger)
        & df["consolidated_code_postal"].notna()
    )
    # On vérifie si le code_geo_manquant est identique au code postal
    # Si oui, on privilégie le code_geo_total (code INSEE issu du géocodage)
    conflit_mask = mask & (df["code_geo_manquant"] == df["consolidated_code_postal"])
    nb_corrections = conflit_mask.sum()
    df.loc[conflit_mask, "code_geo_manquant"] = df.loc[conflit_mask, "code_geo_total"]
    print(f"Correction de conflit Code Postal appliquée sur {nb_corrections} lignes.")
    return df


###### 4. AUTRES ######


def garder_derniere_observation_commune(df):
    """
    Conserve la ligne la plus récente pour chaque commune (CODGEO).
    Une seule ligne finale par commune.
    """
    df = df.copy()
    df["DATE_ARRETE"] = pd.to_datetime(df["DATE_ARRETE"], errors="coerce")
    df = df.sort_values(["CODGEO", "DATE_ARRETE"])
    df_latest = df.drop_duplicates(subset="CODGEO", keep="last")
    return df_latest


def imputer_valeurs_manquantes_fusion(df, cols_to_zero, cols_to_label):
    """
    Gère les NaN générés par la jointure.
    df : le dataframe fusionné
    cols_to_zero : la liste des colonnes passée depuis le notebook
    """
    df = df.copy()

    # Remplacement par 0 pour les colonnes numériques demandées
    cols_num_presentes = [c for c in cols_to_zero if c in df.columns]
    df[cols_num_presentes] = df[cols_num_presentes].fillna(0)

    # Remplacement par un label pour le top opérateur
    cols_presentes = [c for c in cols_to_label if c in df.columns]
    df[cols_presentes] = df[cols_presentes].fillna("Aucun")

    return df
