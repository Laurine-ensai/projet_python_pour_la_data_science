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