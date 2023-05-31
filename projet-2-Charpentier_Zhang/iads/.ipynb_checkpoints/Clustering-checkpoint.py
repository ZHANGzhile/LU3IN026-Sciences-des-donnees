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
import math
import random
import matplotlib.pyplot as plt

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

def centroide(df):
    df = df.to_numpy() if type(df) == pd.DataFrame else df
    df = np.transpose(df)
    #print(type(df))
    #print(pd.DataFrame)
    means = []
    #print(df)

    for i in range(len(df)):
        c = df[i]
        means.append(c.mean())

    return means

def CHA_centroid(df, verbose=False, dendrogramme=False, labels=[]):
    df = df.copy(deep=True)
    fused = []
    P0 = initialise_CHA(df)

    while len(P0) > 1:
        P0, idx1, idx2, dist = fusionne(df, P0, verbose)
        fused.append([idx1, idx2, dist, len(list(P0.values())[-1])])

    if dendrogramme:
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Exemple", fontsize=18)
        plt.xticks([i for i in range(len(df))], labels)
        plt.ylabel('Distance', fontsize=18)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            fused,
            #labels=labels,
            leaf_font_size=18.,  # taille des caractères de l'axe des X
        )

        #plt.show()

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

def fusionne_methode(df, P0, methode, verbose=False, labels=[]):
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

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False, labels=[]):
    """  ##### donner une documentation à cette fonction
    """
    if linkage == "centroid":
        return CHA_centroid(DF, verbose, dendrogramme, labels)

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
        plt.xlabel("Indice d'exemple", fontsize=18)
        plt.ylabel('Distance', fontsize=18)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            fused,
            #labels=labels,
            leaf_font_size=18.,  # taille des caractères de l'axe des X
        )
        #plt.show()

    return fused

def min_max(df):
    ranges = []

    for c in df.columns:
        cmin = df[c].min()
        cmax = df[c].max()
        #print(f"{c}  min: {cmin}")
        #print(f"{c}  max: {cmax}")
        ranges.append([cmin, cmax])

    return ranges

def normalisation(df):
    df2 = df.copy(deep=True)
    ranges = min_max(df)

    for i in range(len(df.columns)):
        c = df[df.columns[i]]
        df2[df.columns[i]] = (c - ranges[i][0]) / (ranges[i][1] - ranges[i][0])

    return df2

def inertie_cluster(Ens):
    center = centroide(Ens)
    inertia = 0

    for _, row in Ens.iterrows():
        dist = dist_euclidienne(row, center)
        inertia += dist ** 2

    return inertia


def init_kmeans(K,Ens):
    indices = np.random.choice(Ens.shape[0], size=K, replace=False)
    sample = Ens.iloc[indices].values

    return sample

def plus_proche(Exe,Centres):
    min_dist = float(100)
    min_index = len(Centres)

    for c in range(len(Centres)):
        dist = dist_euclidienne(Exe, Centres[c])
        if dist < min_dist:
            min_dist = dist
            min_index = c

    return min_index

def affecte_cluster(Base,Centres):
    mat = np.zeros((len(Base), len(Centres)))
    res = {}

    for i in range(len(Centres)):
        res[i] = []

    for i in range(0,len(Base)):
        pproche = plus_proche(Base.iloc[i],Centres)
        res[pproche].append(i)
        #mat[i][pproche] = 1
    #print("L'exemple ",i," est le plus proche du centroide ",pproche)
    return res

def nouveaux_centroides(Base,U):
    res = []

    for k,v in U.items():
        cent = centroide(Base.iloc[U[k]])
        res.append(cent)

    return res

def inertie_globale(Base, U):
    inerties = []

    for k, v in U.items():
        inerties.append(inertie_cluster(Base.iloc[U[k]]))

    return np.sum(inerties)

def kmoyennes(K, Base, epsilon, iter_max, verbose=False):
    curr_iter = 0
    Centroides = init_kmeans(K, Base)
    prev_inertie = 1

    while curr_iter < iter_max:
        DictAffect = affecte_cluster(Base, Centroides)

        Centroides = nouveaux_centroides(Base, DictAffect)
        inertie = inertie_globale(Base, DictAffect)

        diff = abs(inertie - prev_inertie)
        prev_inertie = inertie

        if verbose:
            print(f"Iteration {curr_iter}    Inertie: {inertie}   Difference: {diff}")

        if diff < epsilon:
            return (np.asarray(Centroides), DictAffect,inertie)

        curr_iter += 1

def affiche_resultat(Base,Centres,Affect):
    #plt.scatter(data_2D_norm['X1'],data_2D_norm['X2'],color='b')
    couleurs = cm.tab20(np.linspace(0, 1, len(Affect)))

    for k, cluster in Affect.items():
        for index in cluster:
            plt.scatter(Base.iloc[index][0],Base.iloc[index][1],color=couleurs[k])

    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
