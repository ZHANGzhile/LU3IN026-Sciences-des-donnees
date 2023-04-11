# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd

from copy import deepcopy

import scipy.cluster.hierarchy

def dist_euclidienne(p1, p2):
    #print(f"{p1.shape} | {p2.shape}")
    #print(f"{p1} | {p2}")
    
    return np.linalg.norm(np.asarray(p1)-np.asarray(p2))

def dist_centroides(v1, v2):
    center1 = np.asarray(centroide(v1))
    center2 = np.asarray(centroide(v2))
    return dist_euclidienne(center1, center2)

def initialise_CHA(df):
    return dict([(i, [i]) for i in range(len(df))])

def fusionne(df, P0, verbose=False):
    P1 = deepcopy(P0)
    idx1 = 0
    idx2 = 0
    min_dist = 10
    keys = list(P1.keys())
    
    for c1 in keys:
        for c2 in keys:
            if c2 > c1:
                dist = dist_centroides(df.loc[P1[c1]], df.loc[P1[c2]])
                if dist < min_dist:
                    min_dist = dist
                    idx1 = c1
                    idx2 = c2

    fused = P1[idx1] + P1[idx2]
    P1[(len(df) * 2) - len(P1)] = fused
    #print(P1)
    P1.pop(idx2)
    P1.pop(idx1)
    
    if verbose:
        print(f"Distance mininimale trouvée entre [{idx1}, {idx2}] = {min_dist}")
    
    return (P1, idx1, idx2, min_dist)

def CHA_centroid(df, verbose=False, dendrogramme=False):
    df = df.copy(deep=True)
    fused = []
    P0 = initialise_CHA(df)
    
    while len(P0) > 1:
        P0, idx1, idx2, dist = fusionne(df, P0, verbose)
        fused.append([idx1, idx2, dist, len(list(P0.values())[-1])])
        
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            fused, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        plt.show()
        
    return fused

# ------------------------ 

def clustering_hierarchique_complete(v1, v2):
    max_dist = 0
    
    for _, c1 in v1.iterrows():
        for _, c2 in v2.iterrows():
            dist = dist_euclidienne(c1, c2)
            if dist > max_dist:
                max_dist = dist

    return max_dist

def clustering_hierarchique_simple(v1, v2):
    min_dist = 100
    
    for _, c1 in v1.iterrows():
        for _, c2 in v2.iterrows():
            dist = dist_euclidienne(c1, c2)
            if dist < min_dist:
                min_dist = dist

    return min_dist

def clustering_hierarchique_average(v1, v2):
    sum_dist = 0
    
    for _, c1 in v1.iterrows():
        for _, c2 in v2.iterrows():
            dist = dist_euclidienne(c1, c2)
            sum_dist += dist

    return sum_dist / (len(v1) * len(v2))

def fusionne_methode(df, P0, methode, verbose=False):
    P1 = deepcopy(P0)
    idx1 = 0
    idx2 = 0
    min_dist = 10
    keys = list(P1.keys())
    
    for c1 in keys:
        for c2 in keys:
            if c2 > c1:
                dist = methode(df.loc[P1[c1]], df.loc[P1[c2]])
                if dist < min_dist:
                    min_dist = dist
                    idx1 = c1
                    idx2 = c2

    fused = P1[idx1] + P1[idx2]
    P1[(len(df) * 2) - len(P1)] = fused
    #print(P1)
    P1.pop(idx2)
    P1.pop(idx1)
    
    if verbose:
        print(f"Distance mininimale trouvée entre [{idx1}, {idx2}] = {min_dist}")
    
    return (P1, idx1, idx2, min_dist)

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    if linkage == "centroid":
        return CHA_centroid(DF, verbose, dendrogramme)
    
    methodes = {"complete": clustering_hierarchique_complete, "simple": clustering_hierarchique_simple, "average": clustering_hierarchique_average}
    methode = methodes[linkage]
    
    P0 = initialise_CHA(DF)
    fused = []
    
    while len(P0) > 1:
        P0, idx1, idx2, dist = fusionne_methode(DF, P0, methode, verbose)
        fused.append([idx1, idx2, dist, len(list(P0.values())[-1])])
        
    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            fused, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )
        plt.show()
        
    return fused