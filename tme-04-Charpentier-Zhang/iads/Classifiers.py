# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd

# ---------------------------
class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        return
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI : 
        predicted = [self.predict(desc_set[i]) for i in range(len(desc_set))]
        #print(predicted)
        correct = 0
        for i in range(len(predicted)):
            correct += int(predicted[i] == label_set[i])
        #correct = len(predicted[predicted == label_set]) / 2
        
        return (correct) / len(label_set)

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.intput_dimension = input_dimension
        self.k = k
        self.desc_set = np.asarray([])
        self.label_set = np.asarray([])
        
        #raise NotImplementedError("Please Implement this method")
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        def distance(p1, p2):
            return np.linalg.norm(p1-p2)
            #return math.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2 )
        
        # trouver les distances
        dist = np.array([distance(x, y) for y in self.desc_set])
        dist_fusion = np.dstack((dist, self.label_set))
        dist_sorted = np.argsort(dist_fusion, axis=1)
        
        #print(dist_sorted)
        dist_sorted = dist_sorted[:,:self.k]
        """
        nb_close = 0
        for i in range(self.k):
            if dist_fusion[0][dist_sorted[0][i][0]][1] == +1:
                nb_close += 1
        """
        close = dist_sorted[0][self.label_set[dist_sorted[0]] == +1]
        
        nb_close = len(close)
        
        proportion = nb_close / self.k
        
        return 2*proportion - 1
        
        #dist = [distance(x, y) for y in self.desc_set]
        #dist.sort()
        
        #close = [dist[d] for d in range(self.k)
    
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        pro = (self.score(x) + 1)/2
        
        if pro == 0.5:
            return 0
        elif pro > 0.5:
            return 1
        elif pro < 0.5:
            return -1
        else:
            print("OUT OF RANGE")
        

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
        
        #raise NotImplementedError("Please Implement this method")
    
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True ):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        if init:
            self.w = np.asarray([0 for i in range(input_dimension)])
        else:
            scale_factor = 0.001
            self.w = np.asarray([((np.random.uniform(0, 1) * 2) - 1) * scale_factor for i in range(input_dimension)])
        self.allw =[self.w.copy()] # stockage des premiers poids
        
        #raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        # sample
        len_desc = len(desc_set)
        l =[i for i in range(0,len_desc)]
        np.random.shuffle(l)
        
        # calculate prediction
        for i in l:
            #print(desc_set[i] * self.w)
            error = np.sign(np.dot(desc_set[i],self.w) * label_set[i])
            #print(f"got dot: {np.dot(desc_set[i],self.w)}")

            # correct w
            if error < 1:
                #print(f"Correction: {self.learning_rate * label_set[i] * desc_set[i]}")
                self.w = self.w + self.learning_rate * label_set[i] * desc_set[i]
        
                # save w
                self.allw.append(self.w.copy())
             
    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        differences = []
        
        # train_step
        for i in range(nb_max):
            frozen_w = self.w.copy()
            
            self.train_step(desc_set, label_set)
        
            # if convergence
            correction = np.linalg.norm(self.w - frozen_w)
            differences.append(correction)
            
            #print(correction)
            if correction < seuil:
                break
        
        return differences
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
            
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return -1 if self.score(x) < 0 else 1
        
        #raise NotImplementedError("Please Implement this method")
    
    def get_allw(self):
        return self.allw

    
class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """
    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)
        # Affichage pour information (décommentez pour la mise au point)
        # print("Init perceptron biais: w= ",self.w," learning rate= ",learning_rate)
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        # sample
        len_desc = len(desc_set)
        l =[i for i in range(0,len_desc)]
        np.random.shuffle(l)
                
        # calculate prediction
        for i in l:
            #print(desc_set[i] * self.w)
            error = self.score(desc_set[i]) * label_set[i]
            #print(f"got dot: {np.dot(desc_set[i],self.w)}")

            # correct w
            if error < 1:
                #print(f"Correction: {self.learning_rate * label_set[i] * desc_set[i]}")
                self.w = self.w + self.learning_rate * (label_set[i] - self.score(desc_set[i])) * desc_set[i]
                
                # save w
                self.allw.append(self.w.copy())
