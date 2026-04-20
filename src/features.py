import pandas as pd


def creer_features_irve(df_irve, col_geo="code_geo_total"):
    """
    Crée un dataset agrégé par zone géographique à partir des bornes IRVE.
    """
    df_irve = df_irve.copy()

    # FEATURES NUMÉRIQUES
    df_final = df_irve.groupby(col_geo).agg(
        total_pdc=("nbre_pdc", "sum"),
        puissance_moyenne=("puissance_nominale", "mean"),
        puissance_max=("puissance_nominale", "max"),
        nb_operateurs=("nom_operateur", "nunique"),
        pct_type_2=("prise_type_2", "mean"),
        pct_combo_ccs=("prise_type_combo_ccs", "mean"),
        pct_type_ef=("prise_type_ef", "mean"),
        pct_paiement_cb=("paiement_cb", "mean"),
        pct_paiement_autre=("paiement_autre", "mean"),
        pct_charge_rapide=("borne_rapide", "mean"),
    )

    top_operateur = (
        df_irve.groupby(col_geo)["nom_operateur"]
        .apply(lambda x: x.mode().iloc[0] if not x.empty else "inconnu")
        .to_frame(name="top_operateur")
    )
    df_final = df_final.join(top_operateur)

    env = pd.get_dummies(
        df_irve[[col_geo, "implantation_station_clean"]],
        columns=["implantation_station_clean"],
        prefix="",
        prefix_sep="",
    )
    env_agg = env.groupby(col_geo).mean()
    df_final = df_final.join(env_agg)

    return df_final.reset_index()


def creer_taux_equipement_ve(df):
    """
    Crée le taux d'équipement en véhicules électriques :
    NB_VP_RECHARGEABLES_EL / NB_VP
    """
    df = df.copy()
    df["taux_equipement_ve"] = df["NB_VP_RECHARGEABLES_EL"] / df["NB_VP"]
    return df
