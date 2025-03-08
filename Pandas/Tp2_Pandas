#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:25:53 2025

@author: Aleksandar Mihajlovic
"""

import pandas as pd
import numpy as np

# Exercice 1 : Creating and Modifying Series
print("= = = = = = = = = = Exercice 1 = = = = = = = = = =\n")
# Une Series est une structure de données unidimensionnelle similaire à une liste ou un tableau.
# Ici, nous créons une Series à partir d'un dictionnaire où les clés deviennent les index et les valeurs sont stockées dans la Series.

data = {'a': 100, 'b': 200, 'c': 300}
series = pd.Series(data)
print(f"Series:\n{series}")



# Exercice 2 : Creating DataFrames
print("\n= = = = = = = = = = Exercice 2 = = = = = = = = = =\n")
# Un DataFrame est une structure tabulaire à deux dimensions avec des étiquettes sur les lignes et colonnes.
# Nous créons un DataFrame à partir d'un dictionnaire où chaque clé est une colonne et chaque liste contient les valeurs associées.

df = pd.DataFrame({
    'A': [1, 4, 7],  # Colonne A avec 3 valeurs
    'B': [2, 5, 8],  # Colonne B avec 3 valeurs
    'C': [3, 6, 9]   # Colonne C avec 3 valeurs
})
print(f"DataFrame:\n{df}")

# Ajout d'une nouvelle colonne D avec des valeurs spécifiques
df['D'] = [10, 11, 12]
print(f"\nDataFrame après ajout de la colonne D:\n{df}")

# Suppression de la colonne B du DataFrame
df = df.drop(columns=['B'])
print(f"\nDataFrame après suppression de la colonne B:\n{df}")



# Exercice 3 : DataFrame Indexing and Selection
print("\n= = = = = = = = = = Exercice 3 = = = = = = = = = =\n")

df1 = pd.DataFrame({
    'A': [1, 4, 7],  # Colonne A avec 3 valeurs
    'B': [2, 5, 8],  # Colonne B avec 3 valeurs
    'C': [3, 6, 9]   # Colonne C avec 3 valeurs
})

# Sélection d'une seule colonne sous forme de Series
print("Sélection de la colonne B:")
print(df1['B'])

# Sélection de plusieurs colonnes en passant une liste des noms de colonnes
print("\nSélection des colonnes A et C:")
print(df1[['A', 'C']])

# Sélection d'une ligne avec un index spécifique en utilisant .loc
print("\nSélection de la ligne d'index 1 avec .loc:")
print(df1.loc[1])



# Exercice 4 : Adding and Removing DataFrame Elements
print("\n= = = = = = = = = = Exercice 4 = = = = = = = = = =\n")

# Ajout d'une colonne Sum qui additionne les colonnes A, B et C
df1['Sum'] = df1['A'] + df1['B'] + df1['C']
print("Ajout de la colonne Sum:")
print(df1)

# Suppression de la colonne Sum
df1 = df1.drop(columns=['Sum'])
print("\nDataFrame après suppression de Sum:")
print(df1)

# Ajout d'une colonne Random contenant des valeurs aléatoires générées par NumPy
# np.random.rand(len(df)) génère un nombre aléatoire entre 0 et 1 pour chaque ligne
df1['Random'] = np.random.rand(len(df1))
print("\nDataFrame après ajout de la colonne Random:")
print(df1)



# Exercice 5 : Merging DataFrames
print("\n= = = = = = = = = = Exercice 5 = = = = = = = = = =\n")

# Création de deux DataFrames avec une clé commune
left = pd.DataFrame({'key': [1, 2, 3], 'A': ['A1', 'A2', 'A3'], 'B': ['B1', 'B2', 'B3']})
right = pd.DataFrame({'key': [1, 2, 3], 'C': ['C1', 'C2', 'C3'], 'D': ['D1', 'D2', 'D3']})

# Fusion des DataFrames en utilisant la colonne 'key' comme clé de jointure
# Cela combine les deux DataFrames en associant les valeurs de 'key' correspondantes.
merged_df = pd.merge(left, right, on='key')
print("Fusion des DataFrames:")
print(merged_df)

# Modification en outer join
merged_outer = pd.merge(left, right, on='key', how='outer')
print("\nFusion Outer Join:")
print(merged_outer)

# Ajout de la colonne E et mise à jour de la fusion
right['E'] = ['E1', 'E2', 'E3']
merged_updated = pd.merge(left, right, on='key', how='outer')
print("\nFusion avec colonne E:")
print(merged_updated)



# Exercice 6 : Data Cleaning
print("\n= = = = = = = = = = Exercice 6 = = = = = = = = = =\n")

# Création d'un DataFrame avec des valeurs NaN (manquantes)
df_nan = pd.DataFrame({'A': [1.0, np.nan, 7.0], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, 6.0, np.nan]})
#print(f"{df_nan}\n")

# Remplacement des valeurs NaN par 0
df_filled = df_nan.fillna(0)
print("Remplacement des NaN par 0:")
print(df_filled)

# Remplacement des valeurs NaN par la moyenne de chaque colonne
df_filled_mean = df_nan.fillna(df_nan.mean())
print("\nRemplacement des NaN par la moyenne:")
print(df_filled_mean)

# Suppression des lignes avec des NaN avec un autre dataframe pour eviter le data frame vide
df_nan1 = pd.DataFrame({'A': [1.0, np.nan, 7.0], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, 6.0, 2.0]})
df_dropped = df_nan1.dropna()
print("\nSuppression des lignes contenant NaN:")
print(df_dropped)



# Exercice 7 : Grouping and Aggregation
print("\n= = = = = = = = = = Exercice 7 = = = = = = = = = =\n")

df_group = pd.DataFrame({'Category': ['A', 'B', 'A', 'B', 'A', 'B'], 'Value': [1, 2, 3, 4, 5, 6]})

# Calcul de la moyenne des valeurs par catégorie
grouped_mean = df_group.groupby('Category')['Value'].mean()
print("Moyenne par catégorie:")
print(grouped_mean)

# Calcul du total par catégorie
grouped_sum = df_group.groupby('Category')['Value'].sum()
print("\nSomme par catégorie:")
print(grouped_sum)

# Nombre d'entrées par catégorie
grouped_count = df_group.groupby('Category')['Value'].count()
print("\nNombre d'entrées par catégorie:")
print(grouped_count)



# Exercice 8 : Pivot Tables
print("\n= = = = = = = = = = Exercice 8 = = = = = = = = = =\n")

df_pivot = pd.DataFrame({'Category': ['A', 'A', 'A', 'B', 'B', 'B'], 'Type': ['X', 'Y', 'X', 'Y', 'X', 'Y'], 'Value': [1, 2, 3, 4, 5, 6]})

# Création d'une table pivot qui affiche la moyenne avec la marge pour la moyenne
table_pivot = df_pivot.pivot_table(values='Value', index='Category', columns='Type', aggfunc='mean', margins=True, margins_name='Moyenne')
print("\nTable pivot avec moyenne:")
print(table_pivot)

# Table pivot avec somme des valeurs
table_pivot_sum = df_pivot.pivot_table(values='Value', index='Category', columns='Type', aggfunc='sum')
print("\nTable pivot avec somme:")
print(table_pivot_sum)



# Exercice 9 : Time Series Data
print("\n= = = = = = = = = = Exercice 9 = = = = = = = = = =\n")

# Création d'une série temporelle avec un index de dates
date_rng = pd.date_range(start='2023-01-01', periods=6, freq='D')
df_time = pd.DataFrame({'Date': date_rng, 'Value': np.random.randint(1, 100, size=(6))})

# Définition de la colonne 'Date' comme index
df_time.set_index('Date', inplace=True)

# Resampling des données pour obtenir la somme des valeurs tous les 2 jours
df_resampled = df_time.resample('2D').sum()
print("Séries temporelles résumées par période de 2 jours:")
print(df_resampled)



# Exercice 10 : Handling Missing Data
print("\n= = = = = = = = = = Exercice 10 = = = = = = = = = =\n")

# Création d'un DataFrame avec des valeurs NaN
df_missing = pd.DataFrame({'A': [1.0, 2.0, np.nan], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, np.nan, 9.0]})
#print(f"{df_missing}\n")

# Interpolation des valeurs manquantes (remplissage intelligent des NaN)
df_interpolated = df_missing.interpolate()
print("Interpolation des valeurs manquantes:")
print(df_interpolated)

# Suppression des lignes avec NaN
df_dropped1 = df_missing.dropna()
print("\nSuppression des NaN:")
print(df_dropped1)

# Suppression des lignes avec NaN mais depuis le dataframe interpolé pour eviter d'avoir un dataframe vide
df_dropped2 = df_interpolated.dropna()
print("\nSuppression des NaN depuis Interpolé:")
print(df_dropped2)



# Exercice 11 : DataFrame Operations
print("\n= = = = = = = = = = Exercice 11 = = = = = = = = = =\n")

df_ops = pd.DataFrame({'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]})

# Calcul de la somme cumulée
df_cumsum = df_ops.cumsum()
print("Cumulative sum:")
print(df_cumsum)

# Calcul du produit cumulé
df_cumprod = df_ops.cumprod()
print("\nCumulative product:")
print(df_cumprod)

# Soustraction de 1 à chaque élément
df_minus_one = df_ops.map(lambda x: x - 1)
print("\nSoustraction de 1 à chaque élément:")
print(df_minus_one)

