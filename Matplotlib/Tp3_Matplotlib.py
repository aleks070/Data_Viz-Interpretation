#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 21:52:20 2025

@author: Aleksandar Mihajlovic 24012903
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Exercice 1: Polynomial Plot

# Création d'un tableau x avec 100 points linéairement espacés entre -10 et 10
x = np.linspace(-10, 10, 100)
# Calcul de y pour la fonction polynomiale y = 2x³ - 5x² + 3x - 7
y = 2*x**3 - 5*x**2 + 3*x - 7
# Création d'une figure de taille 10x6 pouces
plt.figure(figsize=(10, 6))
# Tracé de y en fonction de x avec une ligne bleue
plt.plot(x, y, 'b-', label='y = 2x³ - 5x² + 3x - 7')
# Ajout des étiquettes aux axes
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# Ajout d'un titre au graphique
plt.title("Exercice 1 - Polynomial Function Plot")
# Ajout d'une légende pour identifier la courbe
plt.legend()
# Affichage du graphique
plt.show()



# Exercice 2: Exponential and Logarithmic Plot

# Création d'un tableau x avec 500 points linéairement espacés entre 0.1 et 10
x = np.linspace(0.1, 10, 500)
# Calcul des fonctions exponentielle et logarithmique
y1 = np.exp(x)
y2 = np.log(x)
# Création d'une figure de taille 10x6 pouces
plt.figure(figsize=(10, 6))
# Tracé des fonctions avec des couleurs et styles de ligne différents
plt.plot(x, y1, 'r-', label='exp(x)')
plt.plot(x, y2, 'g--', label='log(x)')
# Ajout des étiquettes aux axes
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# Ajout d'un titre au graphique
plt.title("Exercice 2 - Exponential and Logarithmic Functions")
# Ajout d'une légende pour identifier les courbes
plt.legend()
# Ajout d'une grille pour faciliter la lecture
plt.grid()
# Sauvegarde du graphique en tant que fichier PNG avec une résolution de 100 DPI
plt.savefig("exp_log_plot.png", dpi=100)
# Affichage du graphique
plt.show()



# Exercice 3: Subplots and Histogram

# Création de tableaux x1 et x2 pour les différentes fonctions
x1 = np.linspace(-2*np.pi, 2*np.pi, 500)
y_tan = np.tan(x1)
y_arctan = np.arctan(x1)
x2 = np.linspace(-2, 2, 500)
y_sinh = np.sinh(x2)
y_cosh = np.cosh(x2)
# Création d'une figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Tracé des fonctions tan(x) et arctan(x) dans le premier sous-graphique
axes[0].plot(x1, y_tan, 'b-', label='tan(x)')
axes[0].plot(x1, y_arctan, 'r--', label='arctan(x)')
axes[0].set_title("Exercice 3 - tan(x) and arctan(x)")
axes[0].legend()
# Tracé des fonctions sinh(x) et cosh(x) dans le second sous-graphique
axes[1].plot(x2, y_sinh, 'g-', label='sinh(x)')
axes[1].plot(x2, y_cosh, 'm--', label='cosh(x)')
axes[1].set_title("Exercice 3 - sinh(x) and cosh(x)")
axes[1].legend()
# Ajustement de la disposition pour éviter les chevauchements
plt.tight_layout()
# Affichage du graphique
plt.show()

# Histogram
# Génération de 1000 valeurs aléatoires suivant une distribution normale
n = np.random.randn(1000)
# Création d'une figure de taille 8x5 pouces
plt.figure(figsize=(8, 5))
# Tracé de l'histogramme avec 30 bins
plt.hist(n, bins=30, color='purple', alpha=0.75)
# Ajout d'un titre au graphique
plt.title("Exercice 3 - Histogram of Normal Distribution")
# Définition des limites de l'axe x pour couvrir toute la plage des données
plt.xlim([n.min(), n.max()])
# Affichage du graphique
plt.show()



# Exercice 4: Scatter Plot

# Création d'un tableau x avec 500 points linéairement espacés entre 0 et 10
x = np.linspace(0, 10, 500)
# Calcul de y avec du bruit ajouté à la fonction sin(x)
y = np.sin(x) + np.random.normal(0, 0.1, 500)
# Création d'une figure de taille 8x5 pouces
plt.figure(figsize=(8, 5))
# Tracé d'un nuage de points avec des tailles et couleurs variables
plt.scatter(x, y, c=y, s=50, cmap='viridis', alpha=0.75)
# Ajout d'une grille pour faciliter la lecture
plt.grid()
# Masquage des graduations des axes
plt.xticks([])
plt.yticks([])
# Sauvegarde du graphique en tant que fichier PDF
plt.savefig("scatter_plot.pdf")
# Affichage du graphique
plt.show()



# Exercice 5: Contour Plot

# Création de tableaux x et y avec 200 points linéairement espacés entre -5 et 5
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
# Calcul de Z pour la fonction sin(sqrt(x^2 + y^2))
Z = np.sin(np.sqrt(X**2 + Y**2))
# Création d'une figure de taille 8x6 pouces
plt.figure(figsize=(8, 6))
# Tracé du graphique de contour avec une colormap 'coolwarm'
contour = plt.contourf(X, Y, Z, cmap='coolwarm')
# Ajout d'une barre de couleur pour indiquer les valeurs des contours
plt.colorbar()
# Ajout d'un titre au graphique
plt.title("Exercice 5 - Contour Plot")
# Affichage du graphique
plt.show()



# Exercice 6: 3D Surface Plot

# Création d'une figure de taille 10x6 pouces
fig = plt.figure(figsize=(10, 6))
# Ajout d'un sous-graphique 3D
ax = fig.add_subplot(111, projection='3d')
# Création de tableaux X et Y avec des valeurs allant de -5 à 5 avec un pas de 0.25
X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
# Calcul de Z pour la fonction cos(sqrt(X^2 + Y^2))
Z = np.cos(np.sqrt(X**2 + Y**2))
# Tracé du graphique de surface 3D avec une colormap 'plasma'
surf = ax.plot_surface(X, Y, Z, cmap='plasma')
# Ajout d'une barre de couleur
fig.colorbar(surf)
# Changement de l'angle de vue
ax.view_init(elev=30, azim=45)
# Ajout d'un titre au graphique
plt.title("Exercice 6 - 3D Surface Plot")
# Affichage du graphique
plt.show()



# Exercice 7: 3D Wireframe Plot

# Création d'une figure de taille 10x6 pouces
fig = plt.figure(figsize=(10, 6))
# Ajout d'un sous-graphique 3D
ax = fig.add_subplot(111, projection='3d')
# Calcul de Z pour la fonction sin(X) * cos(Y)
Z = np.sin(X) * np.cos(Y)
# Tracé du graphique en fil de fer 3D
ax.plot_wireframe(X, Y, Z, color='black')
# Ajout d'un titre au graphique
ax.set_title("Exercice 7 - 3D Wireframe Plot")
# Affichage du graphique
plt.show()



# Exercice 8: 3D Parametric Plot

# Création d'un tableau t avec 100 points linéairement espacés entre 0 et 2π
t = np.linspace(0, 2*np.pi, 100)
# Calcul des coordonnées X, Y, et Z
X, Y, Z = np.sin(t), np.cos(t), t
# Création d'une figure
fig = plt.figure()
# Ajout d'un sous-graphique 3D
ax = fig.add_subplot(111, projection='3d')
# Tracé du graphique paramétrique 3D
ax.plot(X, Y, Z, color='blue')
# Ajout d'un titre au graphique
ax.set_title("Exercice 8 - 3D Parametric Plot")
# Affichage du graphique
plt.show()



# Exercice 9: 3D Scatter Plot

# Création de trois tableaux x, y, et z avec 100 valeurs tirées d'une distribution normale
x, y, z = np.random.randn(100), np.random.randn(100), np.random.randn(100)
# Création d'une figure
fig = plt.figure()
# Ajout d'un sous-graphique 3D
ax = fig.add_subplot(111, projection='3d')
# Tracé d'un nuage de points 3D avec des couleurs basées sur les valeurs de z
sc = ax.scatter(x, y, z, c=z, cmap='coolwarm')
# Ajout d'une barre de couleur
fig.colorbar(sc)
# Ajout d'un titre au graphique
ax.set_title("Exercice 9 - 3D Scatter Plot")
# Affichage du graphique
plt.show()



# Exercice 10: 3D Density Plot

# Création de trois tableaux x, y, et z avec 1000 valeurs tirées d'une distribution normale
x, y, z = np.random.randn(1000), np.random.randn(1000), np.random.randn(1000)
# Création d'une figure
fig = plt.figure()
# Ajout d'un sous-graphique 3D
ax = fig.add_subplot(111, projection='3d')
# Tracé d'un nuage de points 3D avec des couleurs basées sur les valeurs de z
ax.scatter(x, y, z, c=z, cmap='inferno', alpha=0.6)
# Ajout d'un titre au graphique
ax.set_title("Exercice 10 - 3D Density Plot")
# Affichage du graphique
plt.show()



# Exercice 11: Box Plot

# Création d'une figure de taille 8x5 pouces
plt.figure(figsize=(8, 5))
# Création de quatre ensembles de données aléatoires
data = [np.random.randn(100) for _ in range(4)]
# Tracé d'un diagramme en boîte
plt.boxplot(data)
# Ajout d'un titre au graphique
plt.title("Exercice 11 - Box Plot Example")
# Affichage du graphique
plt.show()



# Exercice 12: Box Plot

# Création de cinq ensembles de données normales avec des moyennes différentes
data = [np.random.normal(loc=mu, scale=1, size=100) for mu in range(5)]
# Création d'une figure de taille 8x6 pouces
plt.figure(figsize=(8, 6))
# Tracé d'un diagramme en boîte avec des couleurs personnalisées
plt.boxplot(data, patch_artist=True, notch=True, boxprops=dict(facecolor="lightblue"))
# Ajout des étiquettes aux axes
plt.xlabel("Category")
plt.ylabel("Values")
# Ajout d'un titre au graphique
plt.title("Exercice 12 - Box Plot")
# Ajout d'une grille pour faciliter la lecture
plt.grid(True)
# Affichage du graphique
plt.show()



# Exercice 13: Violin Plot

# Création de cinq ensembles de données normales avec des moyennes différentes
data = [np.random.normal(loc=mu, scale=1, size=100) for mu in range(5)]
# Création d'une figure de taille 8x6 pouces
plt.figure(figsize=(8, 6))
# Tracé d'un diagramme en violon avec les moyennes et médianes affichées
plt.violinplot(data, showmeans=True, showmedians=True)
# Ajout des étiquettes aux axes
plt.xlabel("Category")
plt.ylabel("Values")
# Ajout d'un titre au graphique
plt.title("Exercice 13 - Violin Plot")
# Ajout d'une grille pour faciliter la lecture
plt.grid(True)
# Affichage du graphique
plt.show()



# Exercice 14: Pie Chart

# Création des étiquettes et des tailles pour le diagramme circulaire
labels = ['A', 'B', 'C', 'D']
sizes = [20, 30, 25, 25]
# Création d'une figure de taille 6x6 pouces
plt.figure(figsize=(6, 6))
# Tracé du diagramme circulaire avec les pourcentages affichés
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
# Ajout d'un titre au graphique
plt.title("Exercice 14 - Pie Chart Example")
# Affichage du graphique
plt.show()



# Exercice 15: Bar Chart

# Création d'une figure de taille 8x5 pouces
plt.figure(figsize=(8, 5))
# Tracé d'un diagramme en barres avec des couleurs personnalisées
plt.bar(labels, sizes, color=['blue', 'red', 'green', 'purple'])
# Ajout d'un titre au graphique
plt.title("Exercice 15 - Bar Chart Example")
# Affichage du graphique
plt.show()



# Exercice 16: Line Plot

# Création d'un tableau x avec 100 points linéairement espacés entre 0 et 10
x = np.linspace(0, 10, 100)
# Calcul de y pour la fonction sin(x)
y = np.sin(x)
# Tracé d'un graphique en ligne
plt.plot(x, y, label="sin(x)")
# Ajout d'un titre au graphique
plt.title("Exercice 16 - Line Plot")
# Ajout d'une légende pour identifier la courbe
plt.legend()
# Affichage du graphique
plt.show()



# Exercice 17: Step Plot

# Tracé d'un graphique en escalier pour la fonction sin(x)
plt.step(x, y, label="step function")
# Ajout d'un titre au graphique
plt.title("Exercice 17 - Step Plot")
# Ajout d'une légende pour identifier la courbe
plt.legend()
# Affichage du graphique
plt.show()



# Exercice 18: Area Plot

# Tracé d'un graphique de zone pour la fonction sin(x)
plt.fill_between(x, y, color="skyblue", alpha=0.5)
# Ajout d'un titre au graphique
plt.title("Exercice 18 - Area Plot")
# Affichage du graphique
plt.show()



# Exercice 19: Polar Plot

# Création d'un tableau theta avec 100 points linéairement espacés entre 0 et 2π
theta = np.linspace(0, 2*np.pi, 100)
# Calcul de r pour la fonction 1 + sin(4*theta)
r = 1 + np.sin(4*theta)
# Tracé d'un graphique polaire
plt.polar(theta, r)
# Ajout d'un titre au graphique
plt.title("Exercice 19 - Polar Plot")
# Affichage du graphique
plt.show()



# Exercice 20: Hexbin Plot

# Création de deux tableaux x et y avec 1000 valeurs tirées d'une distribution normale
x = np.random.randn(1000)
y = np.random.randn(1000)
# Tracé d'un graphique hexbin pour visualiser la densité des points
plt.hexbin(x, y, gridsize=50, cmap='Blues')
# Ajout d'une barre de couleur pour indiquer les valeurs des hexagones
plt.colorbar()
# Ajout d'un titre au graphique
plt.title("Exercice 20 - Hexbin Plot")
# Affichage du graphique
plt.show()
