#Pour df_irve

df_irve = pd.read_csv(
 'https://www.data.gouv.fr/api/1/datasets/r/eb76d20a-8501-400e-b336-d85724de5435'
)

print(f"Shape : {df_irve.shape}")
df_irve.sample(5)


# Exemple d'agrégation complexe pour df_irve
df_irve_agg = df_irve.groupby('code_geo_total').agg(
    total_pdc=('nbre_pdc', 'sum'),
    nb_operateurs=('nom_operateur', 'nunique'),
    puissance_moyenne=('puissance_nominale', 'mean'),
    # Calcul d'une part : on somme les 'True' (1) et on divise par le nombre de lignes
    part_cb=('paiement_cb', lambda x: (x == True).mean() * 100),
    part_rapide=('puissance_nominale', lambda x: (x > 22).mean() * 100)
).reset_index()