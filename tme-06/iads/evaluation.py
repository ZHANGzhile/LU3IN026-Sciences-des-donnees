# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
def crossval_strat(X, Y, n_iterations, iteration):
    balanced = {}
    for y in np.unique(Y):
        balanced[y] = []
    
    for i in range(len(Y)):
        balanced[Y[i]].append(X[i])
                                        
    for y in balanced.keys() :
        balanced[y] = np.asarray(balanced[y])
        
    #print(balanced)
    Xshape = X.shape
    Yshape = Y.shape
    
    Xtest, Ytest, Xapp, Yapp = np.empty([0, Xshape[1]], dtype=int), np.array([]), np.empty([0, Xshape[1]], dtype=int), np.array([])
    #print(type(Xtest))
    
    for k, v in balanced.items():
        proportion = len(v) / n_iterations
        
        lowc = int(iteration * proportion)
        highc = int((iteration+1) * proportion)
        
        #print("x: ", Xtest, "v: ", v[lowc:highc])

        Xtest = np.concatenate((Xtest, v[lowc:highc]))
        Ytest = np.concatenate((Ytest, np.linspace(k, k, highc-lowc)))

        Xapp = np.concatenate((Xapp, np.concatenate((v[:lowc], v[highc:]))))
        Yapp = np.concatenate((Yapp, np.linspace(k, k, lowc + (len(v) - highc))))
    
    return Xapp, Yapp, Xtest, Ytest

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = np.mean(L)
    ecart_type = np.std(L)
    
    return moyenne, ecart_type
