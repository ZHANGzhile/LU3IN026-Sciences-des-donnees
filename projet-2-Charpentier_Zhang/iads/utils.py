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
import copy

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

# plot2D:
def plot2D(desc, labels, xlabel, ylabel, title):
    plt.plot(desc, labels)  
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

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
    
def zip_x(mat, index1, index2):
    # permet d'extraire deux rangees d'une matrice
    return np.asarray([mat[index1], mat[index2]])
    
# plot_frontiere a N dimensions:
def plot_frontiere_ND(desc_set, label_set, classifier, nb_dim, dim_1, dim_2, step=30):
    """ desc_set * label_set * Classifier * int * float -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    #print(f"desc_set_shape: {desc_set.shape}")
    
    mmax=desc_set.max(0)
    mmin=desc_set.min(0)
    mmeans=desc_set.mean(0)
    
    #print(f"desc_set maxs: {mmax}")
    #print(f"desc_set mins: {mmin}")
    #print(f"desc_set means: {mmeans}")
    
    #grid=np.hstack([xgrids[i].reshape(xgrids[i].size,1) for i in range(len(xgrids))])
    #print(f"grid shape: {grid.shape}")
    
    # calcul de la prediction pour chaque point de la grille
    #res=np.array([classifier.predict(grid[i,:]) for i in range(len(grid)) ])
    #res=res.reshape(xgrids[0].shape)

    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    xgrids = np.array(np.meshgrid(np.linspace(mmin[dim_1],mmax[dim_1],step),np.linspace(mmin[dim_2],mmax[dim_2],step)))
    
    #grid = np.array([5, len(desc_set_t)])
    #grid.fill(point_fixe)
    #grid[i] = desc_set_t[i]
    #grid[j] = desc_set_t[j]
    grid = np.hstack([xgrids[i].reshape(xgrids[i].size,1) for i in range(len(xgrids))])
    final_grid = np.array([mmeans[:] for i in range(len(grid))])
    #print(f"final_grid shape: {final_grid.shape}")

    for p in range(len(grid)):
        final_grid[p][dim_1] = grid[p][0]
        final_grid[p][dim_2] = grid[p][1]
    
    #print(f"grid shape: {grid.shape}")
    #print(f"grid: {grid}")

    res = np.array([classifier.predict(final_grid[i,:]) for i in range(len(grid)) ])
    res = res.reshape(xgrids[0].shape)

    plt.contourf(xgrids[0],xgrids[1],res,colors=["darksalmon","skyblue"],levels=[-1000,0,1000])

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

def crossval(X, Y, n_iterations, iteration):
    lowc = int(iteration * (len(X) / n_iterations))
    highc = int((iteration+1) * (len(X) / n_iterations))
    
    Xtest = X[lowc:highc]
    Ytest = Y[lowc:highc]
    
    Xapp = np.concatenate((X[:lowc], X[highc:]))
    Yapp = np.concatenate((Y[:lowc], Y[highc:]))
    
    return Xapp, Yapp, Xtest, Ytest


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
    
    Xtest, Ytest, Xapp, Yapp = np.empty([0, Xshape[1]], dtype=int), np.array([], dtype=int), np.empty([0, Xshape[1]], dtype=int), np.array([], dtype=int)
    #print(type(Xtest))
    
    for k, v in balanced.items():
        #print(f"{k}: {v}")
        proportion = len(v) / n_iterations
        
        lowc = int(iteration * proportion)
        highc = int((iteration+1) * proportion)
        
        #print("x: ", Xtest, "v: ", v[lowc:highc])

        Xtest = np.concatenate((Xtest, v[lowc:highc]))
        Ytest = np.concatenate((Ytest, np.linspace(k, k, highc-lowc,dtype = int)))

        Xapp = np.concatenate((Xapp, np.concatenate((v[:lowc], v[highc:]))))
        Yapp = np.concatenate((Yapp, np.linspace(k, k, lowc + (len(v) - highc),dtype = int)))
    
    return Xapp, Yapp, Xtest, Ytest

def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    moyenne = np.mean(L)
    ecart_type = np.std(L)
    
    return moyenne, ecart_type

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    newC = copy.deepcopy(C)
    
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(X, Y, nb_iter, i)
        newC.train(Xapp, Yapp)
        acc = newC.accuracy(Xtest, Ytest)
        perf.append(acc)
        print(i, " : ", "taille app.= ", len(Xapp)," taille test = ", len(Xtest)," Accuracy: ", acc)

    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

