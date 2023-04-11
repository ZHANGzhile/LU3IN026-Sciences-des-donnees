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
                
                
import copy

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    newC = copy.deepcopy(C)
    
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = ev.crossval_strat(X, Y, nb_iter, i)
        newC.train(Xapp, Yapp)
        acc = newC.accuracy(Xtest, Ytest)
        perf.append(acc)
        print(i, " : ", "taille app.= ", len(Xapp)," taille test = ", len(Xtest)," Accuracy: ", acc)

    (perf_moy, perf_sd) = ev.analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    max = 0
    majori_class = -100
    for i in range(len(valeurs)):
        if nb_fois[i] > max:
            max = nb_fois[i]
            majori_class = valeurs[i]
    return majori_class

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    labels = np.unique(Y)
    entropie_ens = entropie(Y)
    if (entropie_ens <= epsilon):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        min_entropie = 1.1
        i_best = -1
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui minimise l'entropie
        # min_entropie : la valeur de l'entropie minimale
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur de l'entropie de la classe pour chaque attribut.
        
        
        ############
        #特征数量
        numFeatures = len(X[0])
        for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
            featList = [example[i] for example in X]
        #创建set集合{}，元素不可重复
            valeurs, nb_fois = np.unique(featList,return_counts=True)
            #print(valeurs,nb_fois)
            P_vi = [nb_fois[i]/len(Y) for i in range(len(valeurs))]

            new_entropie = 0.0
            d = {}
            for v in valeurs:
                d[v] = {}
            for j in range(len(Y)):
                if Y[j] in d[X[j][i]].keys():
                    d[X[j][i]][Y[j]] += 1
                else:
                    d[X[j][i]][Y[j]] = 1
            for k in range(len(valeurs)): #value = 0,1,2,3
            #subDataSet划分后的子集
            #subDataSet = splitDataSet(dataSet, i, value)
            #计算子集的概率
            #prob = len(subDataSet) / float(len(dataSet))
            #根据公式计算经验条件熵
                P_nb = list(d[valeurs[k]].values())
                P_taux = [P_nb[g]/sum(P_nb) for g in range(len(P_nb))]
                #print(P_nb,P_taux)
                new_entropie += P_vi[k] * shannon(P_taux)
        #信息增益
        #infoGain = baseEntropy - newEntropy
        #打印每个特征的信息增益
            #print("第%d个特征的增益为%.3f" % (i, new_entropie))
        #计算信息增益
            if (new_entropie < min_entropie):
            #更新信息增益，找到最大的信息增益
                min_entropie = new_entropie
            #记录信息增益最大的特征的索引值
                i_best = i
                
        featList = [example[i_best] for example in X]
        Xbest_valeurs = np.unique(featList)
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud

import math
def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
        rem: la fonction utilise le log dont la base correspond à la taille de P
    """
    Hs = 0
    for i in range(1, len(P)+1):
        if len(P) == 1 or P[i-1] == 0:
            Hs = Hs + 0.0
        else:
            Hs = Hs + P[i-1]* math.log(P[i-1],len(P))
    if Hs == 0.0:
        return Hs
    else:
        return -Hs
    
def entropie(Y):
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    P = [nb_fois[i]/len(Y) for i in range(len(valeurs))]
    
    return shannon(P)

import graphviz as gv
class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g

class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD(desc_set,label_set,self.epsilon, self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)