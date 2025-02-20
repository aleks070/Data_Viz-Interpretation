
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:19:09 2025

@author: Aleksandar Mihajlovic
"""

#==============================================================================

"""
Réponses à NumPy_Quiz :

Question 1 : C) est vrai.
Question 2 : a) permet de creer un tableau 2D.
Question 3 : b) est vrai.
Question 4 : a) est vrai.
Question 5 : b) est vrai.
Question 6 : b) est vrai.
Question 7 : b) est vrai.
Question 8 : a) est vrai.
Question 9 : a) est vrai.
Question 10 : a) est vrai.
"""
#==============================================================================

"""
Réponses à NumPy_Course :
"""

import numpy as np

#Exercice1
print("Exercice 1\n")

# Créer un tableau 1D
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("Tableau 1D :", array_1d)
print('\n')

# Créer un tableau 2D à partir de la liste imbriquée
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Forme du tableau 2D :", array_2d.shape)
print("Taille du tableau 2D :", array_2d.size)
print('\n')

# Créer un tableau 3D avec des valeurs aléatoires
array_3d = np.random.rand(2, 3, 4)
print("Nombre de dimensions :", array_3d.ndim)
print("Forme du tableau 3D :", array_3d.shape)
print('\n')



#Exercice 2
print("\nExercice 2\n")

# Créer un tableau 1D avec les nombres de 0 à 9
array = np.arange(10)
reversed_array = array[::-1]
print("Tableau inversé :", reversed_array)
print('\n')

# Créer un tableau 2D avec les nombres de 0 à 11
array_2d = np.arange(12).reshape(3, 4)
subarray = array_2d[:2, 2:]
print("Sous-tableau :\n", subarray)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 0 et 10
array_5x5 = np.random.randint(0, 11, size=(5, 5))
array_5x5[array_5x5 > 5] = 0
print("Tableau modifié :\n", array_5x5)
print('\n')



#Exercice 3
print("\nExercice 3\n")

# Créer une matrice identité 3x3
identity_matrix = np.eye(3)
print("Nombre de dimensions (ndim) :", identity_matrix.ndim)
print("Forme (shape) :", identity_matrix.shape)
print("Taille (size) :", identity_matrix.size)
print("Taille de chaque élément (itemsize) :", identity_matrix.itemsize)
print("Taille totale en octets (nbytes) :", identity_matrix.nbytes)
print('\n')

# Créer un tableau de 10 nombres répartis uniformément entre 0 et 5
array_linspace = np.linspace(0, 5, 10)
print("Tableau :", array_linspace)
print("Type de données :", array_linspace.dtype)
print('\n')

# Créer un tableau 3D avec des valeurs aléatoires
array_3d_normal = np.random.randn(2, 3, 4)
print("Tableau 3D :\n", array_3d_normal)
print("Somme de tous les éléments :", np.sum(array_3d_normal))
print('\n')



#Exercice 4
print("\nExercice 4\n")

# Créer un tableau 1D avec des entiers aléatoires entre 0 et 50
array_1d = np.random.randint(0, 51, size=20)
selected_elements = array_1d[[2, 5, 7, 10, 15]]
print("Éléments sélectionnés :", selected_elements)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 0 et 30
array_2d = np.random.randint(0, 31, size=(4, 5))
elements_greater_than_15 = array_2d[array_2d > 15]
print("Éléments supérieurs à 15 :", elements_greater_than_15)
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre -10 et 10
array_1d_negative = np.random.randint(-10, 11, size=10)
array_1d_negative[array_1d_negative < 0] = 0
print("Tableau modifié :", array_1d_negative)
print('\n')



#Exercice 5
print("\nExercice 5\n")

# Créer deux tableaux 1D avec des entiers aléatoires entre 0 et 10
array1 = np.random.randint(0, 11, size=5)
array2 = np.random.randint(0, 11, size=5)
concatenated_array = np.concatenate((array1, array2))
print("Tableau concaténé :", concatenated_array)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 0 et 10
array_2d = np.random.randint(0, 11, size=(6, 4))
part1, part2 = np.vsplit(array_2d, 2)
print("Première partie :\n", part1)
print("Deuxième partie :\n", part2)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 0 et 10
array_2d = np.random.randint(0, 11, size=(3, 6))
part1, part2, part3 = np.hsplit(array_2d, 3)
print("Première partie :\n", part1)
print("Deuxième partie :\n", part2)
print("Troisième partie :\n", part3)
print('\n')



#Exercice 6
print("\nExercice 6\n")

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 100
array_1d = np.random.randint(1, 101, size=15)
mean = np.mean(array_1d)
median = np.median(array_1d)
std_dev = np.std(array_1d)
variance = np.var(array_1d)
print("Tableau :", array_1d)
print("Moyenne :", mean)
print("Médiane :", median)
print("Écart-type :", std_dev)
print("Variance :", variance)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 50
array_2d = np.random.randint(1, 51, size=(4, 4))
row_sums = np.sum(array_2d, axis=1)
column_sums = np.sum(array_2d, axis=0)
print("Tableau 2D :\n", array_2d)
print("Somme de chaque ligne :", row_sums)
print("Somme de chaque colonne :", column_sums)
print('\n')

# Créer un tableau 3D avec des entiers aléatoires entre 1 et 20
array_3d = np.random.randint(1, 21, size=(2, 3, 4))
max_axis0 = np.max(array_3d, axis=0)
min_axis0 = np.min(array_3d, axis=0)
max_axis1 = np.max(array_3d, axis=1)
min_axis1 = np.min(array_3d, axis=1)
max_axis2 = np.max(array_3d, axis=2)
min_axis2 = np.min(array_3d, axis=2)
print("Tableau 3D :\n", array_3d)
print("Max le long de l'axe 0 :\n", max_axis0)
print("Min le long de l'axe 0 :\n", min_axis0)
print("Max le long de l'axe 1 :\n", max_axis1)
print("Min le long de l'axe 1 :\n", min_axis1)
print("Max le long de l'axe 2 :\n", max_axis2)
print("Min le long de l'axe 2 :\n", min_axis2)
print('\n')



#Exercice 7
print("\nExercice 7\n")

# Créer un tableau 1D avec les nombres de 1 à 12
array_1d = np.arange(1, 13)
array_2d = array_1d.reshape(3, 4)
print("Tableau 2D redimensionné :\n", array_2d)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(3, 4))
transposed_array = array_2d.T
print("Tableau original :\n", array_2d)
print("Tableau transposé :\n", transposed_array)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(2, 3))
flattened_array = array_2d.flatten()
print("Tableau 2D original :\n", array_2d)
print("Tableau 1D aplati :", flattened_array)
print('\n')



#Exercice 8
print("\nExercice 8\n")

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(3, 4))
column_means = np.mean(array_2d, axis=0)
result = array_2d - column_means
print("Tableau original :\n", array_2d)
print("Moyennes des colonnes :", column_means)
print("Tableau après soustraction des moyennes :\n", result)
print('\n')

# Créer deux tableaux 1D avec des entiers aléatoires entre 1 et 5
array1 = np.random.randint(1, 6, size=4)
array2 = np.random.randint(1, 6, size=4)
outer_product = np.outer(array1, array2)
print("Tableau 1 :", array1)
print("Tableau 2 :", array2)
print("Produit externe :\n", outer_product)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(4, 5))
array_2d[array_2d > 5] += 10
print("Tableau modifié :\n", array_2d)
print('\n')



#Exercice 9
print("\nExercice 9\n")

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 20
array_1d = np.random.randint(1, 21, size=10)
sorted_array = np.sort(array_1d)
print("Tableau original :", array_1d)
print("Tableau trié :", sorted_array)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 50
array_2d = np.random.randint(1, 51, size=(3, 5))
sorted_array = array_2d[array_2d[:, 1].argsort()]
print("Tableau original :\n", array_2d)
print("Tableau trié par la deuxième colonne :\n", sorted_array)
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 100
array_1d = np.random.randint(1, 101, size=15)
indices = np.where(array_1d > 50)
print("Tableau :", array_1d)
print("Indices des éléments supérieurs à 50 :", indices[0])
print('\n')



#Exercice 10
print("\nExercice 10\n")

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2x2 = np.random.randint(1, 11, size=(2, 2))
determinant = np.linalg.det(array_2x2)
print("Tableau 2x2 :\n", array_2x2)
print("Déterminant :", determinant)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 5
array_3x3 = np.random.randint(1, 6, size=(3, 3))
eigenvalues, eigenvectors = np.linalg.eig(array_3x3)
print("Tableau 3x3 :\n", array_3x3)
print("Valeurs propres :", eigenvalues)
print("Vecteurs propres :\n", eigenvectors)
print('\n')

# Créer deux tableaux 2D avec des entiers aléatoires entre 1 et 10
array1 = np.random.randint(1, 11, size=(2, 3))
array2 = np.random.randint(1, 11, size=(3, 2))
matrix_product = np.dot(array1, array2)
print("Tableau 1 (2x3) :\n", array1)
print("Tableau 2 (3x2) :\n", array2)
print("Produit matriciel :\n", matrix_product)
print('\n')



#Exercice 11
print("\nExercice 11\n")

# Créer un tableau 1D avec 10 échantillons aléatoires d'une distribution uniforme sur [0, 1)
uniform_samples = np.random.rand(10)
print("Échantillons uniformes :", uniform_samples)
print('\n')

# Créer un tableau 2D avec des échantillons aléatoires d'une distribution normale
normal_samples = np.random.normal(loc=0, scale=1, size=(3, 3))
print("Échantillons normaux :\n", normal_samples)
#print(np.mean(normal_samples))
#print(np.std(normal_samples))
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 100
array_1d = np.random.randint(1, 101, size=20)
histogram, bin_edges = np.histogram(array_1d, bins=5)
print("Tableau :", array_1d)
print("Histogramme :", histogram)
print("Bords des bins :", bin_edges)
print('\n')



#Exercice 12
print("\nExercice 12\n")

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 20
array_2d = np.random.randint(1, 21, size=(5, 5))
diagonal_elements = np.diag(array_2d)
print("Tableau 2D :\n", array_2d)
print("Éléments diagonaux :", diagonal_elements)
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 50
array_1d = np.random.randint(1, 51, size=10)
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

prime_elements = array_1d[np.vectorize(is_prime)(array_1d)]
print("Tableau :", array_1d)
print("Éléments premiers :", prime_elements)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(4, 4))
even_elements = array_2d[array_2d % 2 == 0]
print("Tableau 2D :\n", array_2d)
print("Éléments pairs :", even_elements)
print('\n')



#Exercice 13
print("\nExercice 13\n")

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 10
array_1d = np.random.randint(1, 11, size=10).astype(float)
array_1d[np.random.randint(0, 10, size=3)] = np.nan
print("Tableau avec NaN :", array_1d)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 10
array_2d = np.random.randint(1, 11, size=(3, 4)).astype(float)
array_2d[array_2d < 5] = np.nan
print("Tableau 2D avec NaN :\n", array_2d)
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 20
array_1d = np.random.randint(1, 21, size=15).astype(float)
array_1d[np.random.randint(0, 15, size=5)] = np.nan
nan_indices = np.where(np.isnan(array_1d))
print("Tableau avec NaN :", array_1d)
print("Indices des éléments NaN :", nan_indices[0])
print('\n')



#Exercice 14
print("\nExercice 14\n")

import time

# Créer un grand tableau 1D avec des entiers aléatoires entre 1 et 100
large_array = np.random.randint(1, 101, size=1000000)
start_time = time.time()
mean = np.mean(large_array)
std_dev = np.std(large_array)
end_time = time.time()
print("Moyenne :", mean)
print("Écart-type :", std_dev)
print("Temps écoulé :", end_time - start_time, "secondes")
print('\n')

# Créer deux grands tableaux 2D avec des entiers aléatoires entre 1 et 10
array1 = np.random.randint(1, 11, size=(1000, 1000))
array2 = np.random.randint(1, 11, size=(1000, 1000))
start_time = time.time()
result = np.add(array1, array2)
end_time = time.time()
print("Temps écoulé pour l'addition :", end_time - start_time, "secondes")
print('\n')

# Créer un tableau 3D avec des entiers aléatoires entre 1 et 10
array_3d = np.random.randint(1, 11, size=(100, 100, 100))
start_time = time.time()
sum_axis0 = np.sum(array_3d, axis=0)
sum_axis1 = np.sum(array_3d, axis=1)
sum_axis2 = np.sum(array_3d, axis=2)
end_time = time.time()
print("Temps écoulé pour le calcul des sommes :", end_time - start_time, "secondes")
print('\n')



#Exercice 15
print("\nExercice 15\n")

# Créer un tableau 1D avec les nombres de 1 à 10
array_1d = np.arange(1, 11)
cumulative_sum = np.cumsum(array_1d)
cumulative_product = np.cumprod(array_1d)
print("Tableau :", array_1d)
print("Somme cumulative :", cumulative_sum)
print("Produit cumulatif :", cumulative_product)
print('\n')

# Créer un tableau 2D avec des entiers aléatoires entre 1 et 20
array_2d = np.random.randint(1, 21, size=(4, 4))
cumulative_sum_rows = np.cumsum(array_2d, axis=1)
cumulative_sum_columns = np.cumsum(array_2d, axis=0)
print("Tableau 2D :\n", array_2d)
print("Somme cumulative le long des lignes :\n", cumulative_sum_rows)
print("Somme cumulative le long des colonnes :\n", cumulative_sum_columns)
print('\n')

# Créer un tableau 1D avec des entiers aléatoires entre 1 et 50
array_1d = np.random.randint(1, 51, size=10)
min_value = np.min(array_1d)
max_value = np.max(array_1d)
sum_value = np.sum(array_1d)
print("Tableau :", array_1d)
print("Minimum :", min_value)
print("Maximum :", max_value)
print("Somme :", sum_value)
print('\n')



#Exercice 16
print("\nExercice 16\n")

# Créer un tableau de 10 dates commençant à partir d'aujourd'hui avec une fréquence quotidienne
dates = np.arange('2025-02-18', 10, dtype='datetime64[D]')
print("Tableau de dates :", dates)
print('\n')

# Créer un tableau de 5 dates commençant à partir du 1er janvier 2022 avec une fréquence mensuelle
dates = np.arange('2022-01-01', 5, dtype='datetime64[M]')
print("Tableau de dates :", dates)
print('\n')

# Créer un tableau 1D avec 10 timestamps aléatoires dans l'année 2023
start_date = np.datetime64('2023-01-01')
end_date = np.datetime64('2023-12-31')
total_days = (end_date - start_date).astype('timedelta64[D]').astype(int) + 1
random_days = np.random.randint(0, total_days, size=10)
timestamps = start_date + random_days.astype('timedelta64[D]')
print("Timestamps aléatoires en 2023 :", timestamps)
print('\n')



#Exercice 17
print("\nExercice 17\n")

# Définir un type de données personnalisé en binaire
custom_dtype = [('integer', 'i4'), ('binary', 'U10')]
array_custom = np.array([
    (1, '0001'),
    (2, '0010'),
    (3, '0011'),
    (4, '0100'),
    (5, '0101')
], dtype=custom_dtype)
print("Tableau avec type de données personnalisé :")
print(array_custom)
print('\n')

# Définir un type de données personnalisé pour les nombres complexes
complex_dtype = np.dtype([('real', 'f4'), ('imag', 'f4')])
array_complex = np.array([
    [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
    [(7.0, 8.0), (9.0, 10.0), (11.0, 12.0)],
    [(13.0, 14.0), (15.0, 16.0), (17.0, 18.0)]
], dtype=complex_dtype)
print("Tableau avec nombres complexes :")
print(array_complex)
print('\n')

# Définir un type de données structuré pour les livres
book_dtype = [('title', 'U50'), ('author', 'U50'), ('pages', 'i4')]
books = np.array([
    ('1984', 'George Orwell', 328),
    ('To Kill a Mockingbird', 'Harper Lee', 281),
    ('The Great Gatsby', 'F. Scott Fitzgerald', 180)
], dtype=book_dtype)
print("Tableau structuré des livres :")
print(books)
