import pandas 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import geopandas as gpd
import requests
import io
import numpy as np 

df_step3 = pd.read_parquet("clean_dataset", engine="pyarrow")

colonnes_a_decrire = ['valeur_foncière', 'surface_m2', 'Nombre pieces principales', 'prix_au_m2']
print(df_step3[colonnes_a_decrire].describe(percentiles=[.01, .05, .25, .75, .95, .99]).round(0))

warnings.filterwarnings('ignore')
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)

# Distribution des Surfaces
sns.histplot(df_step3['surface_m2'], bins=50, kde=True)
plt.title('Distribution des Surfaces (m²)')
plt.xlabel('Surface (m²)')

# Distribution des Prix au m²
plt.subplot(1, 2, 2)
sns.histplot(df_step3['prix_au_m2'], bins=50, kde=True)
plt.title('Distribution des Prix au m²')
plt.xlabel('Prix au m² (€)')
plt.show()


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(y=df_step3['surface_m2'])
plt.title('Boxplot des Surfaces')
plt.ylim(0, 300)

plt.subplot(1, 2, 2)
sns.boxplot(y=df_step3['prix_au_m2'])
plt.title('Boxplot des Prix au m²')
plt.ylim(0, 30000) 
plt.show()

df_sans_zeros = df_step3[
    (df_step3['surface_m2'] > 0) & 
    (df_step3['valeur_foncière'] > 0)
].copy()

P01_surface = df_sans_zeros['surface_m2'].quantile(0.01)
P99_surface = df_sans_zeros['surface_m2'].quantile(0.97)
P01_prix_m2 = df_sans_zeros['prix_au_m2'].quantile(0.01)
P99_prix_m2 = df_sans_zeros['prix_au_m2'].quantile(0.97)

df_final = df_sans_zeros[
    (df_sans_zeros['surface_m2'] >= P01_surface) &
    (df_sans_zeros['surface_m2'] <= P99_surface) &
    (df_sans_zeros['prix_au_m2'] >= P01_prix_m2) &
    (df_sans_zeros['prix_au_m2'] <= P99_prix_m2)
].copy()

warnings.filterwarnings('ignore')
plt.figure(figsize=(15, 6))


plt.subplot(1, 2, 1)
sns.histplot(df_final['surface_m2'], bins=50, kde=True)
plt.title('Distribution des Surfaces (Après Nettoyage)')
plt.xlabel('Surface (m²)')


plt.subplot(1, 2, 2)
sns.histplot(df_final['prix_au_m2'], bins=50, kde=True)
plt.title('Distribution des Prix au m² (Après Nettoyage)')
plt.xlabel('Prix au m² (€)')
plt.suptitle('Histogrammes après filtrage des outliers', fontsize=16, y=1.02)
plt.show()


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df_final['surface_m2'])
plt.title('Boxplot des Surfaces (Après Nettoyage)')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_final['prix_au_m2'])
plt.title('Boxplot des Prix au m² (Après Nettoyage)')
plt.suptitle('Boxplots après filtrage des outliers', fontsize=16, y=1.02)
plt.show()

colonnes_a_decrire = ['valeur_foncière', 'surface_m2', 'Nombre pieces principales', 'prix_au_m2']
print(df_final[colonnes_a_decrire].describe(percentiles=[.01, .25, .50, .75, .99]).round(0))

df_ventes_par_dept = df_final['code_departement'].value_counts(ascending=False).reset_index()
df_ventes_par_dept

url_geojson_github = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"

try:
    print("Tentative de téléchargement des données géographiques (Source GitHub)...")
    response = requests.get(url_geojson_github)
    response.raise_for_status() 
    gdf_departements = gpd.read_file(io.BytesIO(response.content))
    if 'code' in gdf_departements.columns:
        gdf_departements.rename(columns={'code': 'code_departement'}, inplace=True)
    else:
        print("AVERTISSEMENT: Colonne 'code' non trouvée. Vérifiez le nom de la colonne.")
    gdf_departements = gdf_departements[gdf_departements['code_departement'].str.len() <= 3]
    print(f"Chargement réussi. {len(gdf_departements)} départements chargés.")

except requests.exceptions.RequestException as e:
    print(f"Erreur de chargement des données géographiques : {e}")
    print("Impossible de continuer. Vérifiez votre connexion Internet.")
    raise SystemExit("Téléchargement du GeoJSON échoué.")


codes_dep = gdf_departements['code_departement'].unique()
vos_ventes = df_ventes_par_dept
vos_ventes.rename(columns={'count': 'Ventes_Immobilieres'}, inplace=True)
gdf_ventes = gdf_departements.merge(vos_ventes, on='code_departement', how='left')

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf_ventes.plot(
    column='Ventes_Immobilieres',  
    ax=ax,
    legend=True,
    cmap='Reds',                  
    linewidth=0.8,
    edgecolor='0.8',              
    missing_kwds={                
        "color": "lightgrey",
        "label": "Données manquantes",
    },
    legend_kwds={'label': "Nombre de Ventes Immobilières",
                 'orientation': "horizontal"}
)
ax.set_title('Volume des Ventes Immobilières par Département entre 2021 et 2024', fontsize=16)
ax.set_axis_off()
plt.show()




median_prix_dept = df_final.groupby('code_departement')['prix_au_m2'].median().sort_values(ascending=False)
print("\nTop 5 des départements (les plus chers) :")
print(median_prix_dept.head(5))
print("\nTop 5 des départements (les moins chers) :")
print(median_prix_dept.tail(5))
plt.figure(figsize=(12, 8))
sns.barplot(x=median_prix_dept.head(20).values, 
            y=median_prix_dept.head(20).index, 
            palette='viridis')
plt.title('Top 20 des Départements par Prix au m² Médian')
plt.xlabel('Prix au m² Médian (€)')
plt.ylabel('Code Département')
plt.show()


df_saison = df_final.copy()
df_saison['mois'] = df_saison['date_mutation'].dt.month
warnings.filterwarnings('ignore')
plt.figure(figsize=(14, 6))
sns.countplot(data=df_saison, 
              x='mois', 
              palette='viridis')
plt.title('Volume Total des Ventes par Mois (Toutes années confondues)')
plt.xlabel("Mois de l'année (1 = Jan, 12 = Dec)")
plt.ylabel("Nombre Total de Ventes")
plt.show()
plt.figure(figsize=(14, 7))
sns.boxplot(data=df_saison, 
            x='mois', 
            y='prix_au_m2', 
            palette='viridis')
plt.title('Distribution du Prix au m² par Mois (Toutes années confondues)')
plt.xlabel("Mois de l'année (1 = Jan, 12 = Dec)")
plt.ylabel("Prix au m² (€)")
plt.show()


# 1. Filtrer sur Paris (75)
df_paris = df_final.copy()
df_paris = df_paris[df_paris['Code departement'].astype(str).isin(['75', '75.0'])]
df_paris = df_paris[df_paris['Type local'].isin(['Appartement', 'Maison'])]
cols_to_keep = {
    'Valeur fonciere': 'valeur_foncière',
    'Surface reelle bati': 'surface_m2',
    'Nombre pieces principales': 'nombre_pieces_principales',
    'Code postal': 'code_postal',
    'Date mutation': 'date_mutation',
    'Nombre de lots': 'nombre_de_lots',
    'Type local': 'type_local'
}
df_paris = df_paris[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
df_paris = df_paris.dropna(subset=['valeur_foncière', 'surface_m2', 'code_postal'])
df_paris['valeur_foncière'] = df_paris['valeur_foncière'].astype(str).str.replace(',', '.').astype(float)
df_paris['code_postal'] = df_paris['code_postal'].astype(float).astype(int).astype(str)
df_paris['nombre_de_lots'] = df_paris['nombre_de_lots'].fillna(1)
df_paris['date_mutation'] = pd.to_datetime(df_paris['date_mutation'], dayfirst=True)
df_paris['annee'] = df_paris['date_mutation'].dt.year
df_paris['mois'] = df_paris['date_mutation'].dt.month
df_paris = df_paris[(df_paris['surface_m2'] > 9) & (df_paris['valeur_foncière'] > 50000)]

df_paris_clean = df_paris.copy()
df_paris_clean['prix_m2'] = df_paris_clean['valeur_foncière'] / df_paris_clean['surface_m2']
df_paris_clean = df_paris_clean[(df_paris_clean['prix_m2'] > 4000) & (df_paris_clean['prix_m2'] < 35000)]
df_paris_clean = df_paris_clean.drop(columns=['prix_m2'])


prix_m2_arr = df_paris_clean.groupby('code_postal').apply(
    lambda x: (x['valeur_foncière'] / x['surface_m2']).mean()
).sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=prix_m2_arr.values, y=prix_m2_arr.index, palette="viridis")

plt.title("Top 15 Arrondissements les plus chers (Prix moyen/m²)", fontsize=16, pad=20)
plt.xlabel("Prix moyen au m² (€)", fontsize=12)
plt.ylabel("Code Postal", fontsize=12)
for i, v in enumerate(prix_m2_arr.values):
    ax.text(v - 1500, i, f"{v:,.0f} €", color='white', va='center', fontweight='bold')
plt.show()


df_map_data = df_paris_clean.copy()
df_map_data['prix_m2'] = df_map_data['valeur_foncière'] / df_map_data['surface_m2']
prix_par_arrondissement = df_map_data.groupby('code_postal')['prix_m2'].median().reset_index()
prix_par_arrondissement.columns = ['c_ar', 'prix_m2_median'] 
url_geojson = "https://opendata.paris.fr/explore/dataset/arrondissements/download/?format=geojson"
paris_map = gpd.read_file(url_geojson)
prix_par_arrondissement['c_ar'] = prix_par_arrondissement['c_ar'].astype(str).str[-2:].astype(int)
paris_final = paris_map.merge(prix_par_arrondissement, on='c_ar', how='left')
fig, ax = plt.subplots(1, 1, figsize=(15, 10)).paris_final.plot(column='prix_m2_median', 
                 ax=ax, 
                 legend=True,
                 legend_kwds={'label': "Prix Médian au m² (€)", 'orientation': "horizontal"},
                 cmap='RdYlBu_r',
                 edgecolor='black',
                 linewidth=0.5)

for idx, row in paris_final.iterrows():
    if not pd.isna(row['prix_m2_median']):
        label = f"{row['c_ar']}e\n{row['prix_m2_median']/1000:.1f}k€"
        plt.annotate(text=label, 
                     xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                     horizontalalignment='center',
                     fontsize=9,
                     color='black',
                     fontweight='bold')

plt.title('Carte des Prix de l\'Immobilier Parisien (DVF 2021-2024)', fontsize=20, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()