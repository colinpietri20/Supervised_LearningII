import pandas as pd
import numpy as np

colonnes_a_conserver = [
    'Valeur fonciere',
    'Nature mutation',
    'Type local',
    'Date mutation',
    'Code departement',
    'Code commune',
    'Commune',
    'Code postal',
    'Surface reelle bati',
    'Surface Carrez du 1er lot',
    'Nombre pieces principales',
    'Nombre de lots']

fichiers = [
    '/Users/cristianlarrain/Desktop/GIT-SUPERVISED-LEARNING/FICHIERS TXT/ValeursFoncieres-2021.txt',
    '/Users/cristianlarrain/Desktop/GIT-SUPERVISED-LEARNING/FICHIERS TXT/ValeursFoncieres-2021.txt',
    '/Users/cristianlarrain/Desktop/GIT-SUPERVISED-LEARNING/FICHIERS TXT/ValeursFoncieres-2021.txt',
    '/Users/cristianlarrain/Desktop/GIT-SUPERVISED-LEARNING/FICHIERS TXT/ValeursFoncieres-2021.txt']

df_complet = pd.concat(
    [
        pd.read_csv(
            f, 
            sep='|', 
            usecols=colonnes_a_conserver, 
            low_memory=False
        ) 
        for f in fichiers
    ], 
    ignore_index=True
)

df=df_complet
df = df.rename(columns={'Type local': 'type_local'})
df = df.rename(columns={'Nature mutation': 'nature_mutation'})
df = df.rename(columns={'Date mutation': 'date_mutation'})
df = df.rename(columns={'Valeur fonciere': 'valeur_foncière'})
df = df.rename(columns={'Code postal': 'code_postal'})
df = df.rename(columns={'Commune': 'commune'})
df = df.rename(columns={'Code departement': 'code_departement'})
df = df.rename(columns={'Code commune': 'code_commune'})
df = df.rename(columns={'Nombre de lots': 'nombre_de_lots'})

df = df[df['type_local'] == "Appartement"]
df = df[
    (df['nature_mutation'] == 'Vente') | 
    (df['nature_mutation'] == "Vente en l'état futur d'achèvement")
]

df_clean = df[df['nombre_de_lots'] == 1]

df_clean['valeur_foncière'] = pd.to_numeric(
    df_clean['valeur_foncière'].str.replace(',', '.'), 
    errors='coerce'
)
df_clean['Surface Carrez du 1er lot'] = pd.to_numeric(
    df_clean['Surface Carrez du 1er lot'].str.replace(',', '.'), 
    errors='coerce'
)
df_clean['date_mutation'] = pd.to_datetime(
    df_clean['date_mutation'], 
    format='%d/%m/%Y', 
    errors='coerce'
)
df_clean['code_postal'] = df_clean['code_postal'].astype(str).str.split('.').str[0]
df_clean = df_clean.drop_duplicates() 

df_step3 = df_clean.copy()
df_step3['code_postal'] = df_step3['code_postal'].astype(str).str.split('.').str[0]
df_step3['surface_m2'] = df_step3['Surface Carrez du 1er lot'].fillna(df_step3['Surface reelle bati'])

colonnes_critiques = [
    'valeur_foncière', 
    'surface_m2',                
    'Nombre pieces principales', 
    'code_commune'
]

df_step3.dropna(subset=colonnes_critiques, inplace=True)
df_step3 = df_step3[df_step3['valeur_foncière'] > 0]
df_step3 = df_step3[df_step3['surface_m2'] > 0]
df_step3 = df_step3[df_step3['Nombre pieces principales'] > 0]

colonnes_a_supprimer = [
    'Surface Carrez du 1er lot', 
    'Surface reelle bati',
]
df_step3 = df_step3.drop(columns=colonnes_a_supprimer)
df_step3['Nombre pieces principales'] = df_step3['Nombre pieces principales'].astype(int)

df_a_verifier = df_step3.copy()
nan_counts = df_a_verifier.isnull().sum()
total_nans = nan_counts.sum()
df_step3['prix_au_m2'] = df_step3['valeur_foncière'] / df_step3['surface_m2']

df_step3.to_parquet("clean_dataset")