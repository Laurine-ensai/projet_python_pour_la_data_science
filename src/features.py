import pandas as pd

def creer_features_irve(df_irve, col_geo="code_geo_total"):
    """
    Crée un dataset agrégé par zone géographique à partir des bornes IRVE.
    """

    # -------------------------
    # FEATURES NUMÉRIQUES
    # -------------------------
    df_agg = df_irve.groupby(col_geo).agg(
        total_pdc=('nbre_pdc', 'sum'),
        puissance_moyenne=('puissance_nominale', 'mean'),
        puissance_max=('puissance_nominale', 'max'),
        nb_operateurs=('nom_operateur', 'nunique'),

        # INFRASTRUCTURE
        pct_type_2=('prise_type_2', 'mean'),
        pct_combo_ccs=('prise_type_combo_ccs', 'mean'),
        pct_chademo=('prise_type_chademo', 'mean'),
        pct_type_ef=('prise_type_ef', 'mean'),

        # ACCESSIBILITÉ
        pct_gratuit=('gratuit', 'mean'),
        pct_paiement_cb=('paiement_cb', 'mean'),
        pct_paiement_autre=('paiement_autre', 'mean')
    ).reset_index()

    # -------------------------
    # TOP OPERATEUR
    # -------------------------
    top_operateur = df_irve.groupby(col_geo)['nom_operateur'] \
        .agg(lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else "inconnu") \
        .reset_index(name='top_operateur')

    # -------------------------
    # ENVIRONNEMENT
    # -------------------------
    env = pd.get_dummies(df_irve['implantation_station'])
    env[col_geo] = df_irve[col_geo]
    env = env.groupby(col_geo).mean().reset_index()

    # -------------------------
    # FUSION
    # -------------------------
    df_final = df_agg \
        .merge(top_operateur, on=col_geo, how='left') \
        .merge(env, on=col_geo, how='left')

    return df_final

def creer_taux_equipement_ve(df):
    """
    Crée le taux d'équipement en véhicules électriques :
    NB_VP_RECHARGEABLES_EL / NB_VP
    """

    df = df.copy()

    df["taux_equipement_ve"] = (
        df["NB_VP_RECHARGEABLES_EL"] / df["NB_VP"]
    )

    return df

def preparer_base_ve(df):
    """
    Pipeline complet :
    - conserve la dernière observation par commune
    - crée le taux de véhicules électriques
    - garde variables utiles
    """

    df = garder_derniere_observation_commune(df)
    df = creer_taux_equipement_ve(df)

    vars_finales = [
        "CODGEO",
        "DATE_ARRETE",
        "NB_VP",
        "NB_VP_RECHARGEABLES_EL",
        "taux_equipement_ve"
    ]

    return df[vars_finales]



