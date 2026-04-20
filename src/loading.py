import pandas as pd
from cartiflette import carti_download


def load_irve_data(path_or_url):
    """
    Charge les données IRVE et conserve uniquement les observations
    dont created_at est antérieur ou égal à la date maximale retenue.
    """
    df = pd.read_csv(path_or_url)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    date_max = pd.Timestamp("2026-04-17 12:54:56.166000+00:00")
    df = df[df["created_at"] <= date_max].copy()

    return df


def load_revenu_data(path_or_url):
    """
    Charge les données de revenus de l'INSEE.
    """
    df = pd.read_csv(path_or_url, sep=";")
    return df


def load_ve_immatriculations(path_or_url):
    """
    Charge les données d'immatriculations des véhicules électriques.
    """
    df = pd.read_csv(path_or_url, encoding="latin-1")
    return df


def charger_communes(departements=None, crs=4326, year=2022):
    """
    Télécharge les contours des communes via Cartiflette.
    """
    if departements is None:
        # Par défaut, on prend les départements métropolitains
        departements = [str(i).zfill(2) for i in range(1, 96)]

    communes = carti_download(
        crs=crs,
        borders="COMMUNE",
        filter_by="DEPARTEMENT",
        values=departements,
        vectorfile_format="geojson",
        source="EXPRESS-COG-CARTO-TERRITOIRE",
        year=year,
    )
    return communes


def load_all_datasets(paths_dict):
    """
    Fonction utilitaire pour charger les trois sources d'un coup
    à partir d'un dictionnaire de chemins.
    """
    df_irve = load_irve_data(paths_dict["irve"])
    df_revenu = load_revenu_data(paths_dict["revenu"])
    df_ve = load_ve_immatriculations(paths_dict["ve"])
    return df_irve, df_revenu, df_ve
