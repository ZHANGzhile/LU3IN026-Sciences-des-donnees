# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ 
# ------------------------ REPRENDRE ICI LES FONCTIONS SUIVANTES DU TME 2:
# genere_dataset_uniform:
def genere_dataset_uniform(p, n: int, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    data2_desc = np.random.uniform(binf, bsup, (int(n*p),p))
    data2_label = np.asarray([-1 for i in range(0,int( n ))] + [+1 for i in range(0,int( n ))])
    
    return data2_desc, data2_label

# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    n = int(nb_points)
    desc_negative = np.random.multivariate_normal(negative_center, negative_sigma, n)
    desc_positive = np.random.multivariate_normal(positive_center, positive_sigma, n)

    desc = np.concatenate((desc_negative, desc_positive))
    
    labels = np.asarray([-1 for i in range(0,int( n ))] + [+1 for i in range(0,int( n ))])
    
    return desc, labels

# plot2DSet:
def plot2DSet(desc,labels):    
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    desc_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    desc_positifs = desc[labels == +1]
    
    # Affichage de l'ensemble des exemples :
    plt.scatter(desc_negatifs[:,0],desc_negatifs[:,1],marker='o', color="red") # 'o' rouge pour la classe -1
    plt.scatter(desc_positifs[:,0],desc_positifs[:,1],marker='x', color="blue") # 'x' bleu pour la classe +1

# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    x1grid,x2grid=np.meshgrid(np.linspace(mmin[0],mmax[0],step),np.linspace(mmin[1],mmax[1],step))
    grid=np.hstack((x1grid.reshape(x1grid.size,1),x2grid.reshape(x2grid.size,1)))
    
    # calcul de la prediction pour chaque point de la grille
    res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    res=res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid,x2grid,res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

    # ------------------------ COMPLETER LES INSTRUCTIONS DANS CETTE BOITE 
def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    print("Nb iterations: ", len(les_variations))
    
    var_00 = np.array([[var, 0], [0, var]])
    
    desc1, label1 = genere_dataset_gaussian(np.array([1,1]), var_00, np.array([0,0]), var_00, n)
    desc2, label2 = genere_dataset_gaussian(np.array([0,1]), var_00, np.array([1,0]), var_00, n)
    return np.concatenate((desc1,desc2),axis=0),np.asarray([-1]*len(desc1)+[1]*len(desc2))
    