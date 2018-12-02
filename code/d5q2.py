# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 2
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - Repérez les commentaires commençant par TODO : ils indiquent une tâche que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, à la fonction
#       checkTime, ni aux conditions vérifiant le bon fonctionnement de votre
#       code. Ces structures vous permettent de savoir rapidement si vous ne
#       respectez pas les requis minimum pour une question en particulier.
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer
#       la note de zéro (0) pour la partie implémentation!
#
###############################################################################

import itertools
import time
import numpy
import warnings

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from matplotlib import pyplot

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, RFE, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split


from sklearn.mixture import GaussianMixture

# Fonctions utilitaires liées à l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")

# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q2Aall = 0.5
TMAX_Q2Achi = 0.5
TMAX_Q2Amut = 60
TMAX_Q2B = 150

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 2
    # Chargement des données et des noms des caractéristiques
    # IMPORTANT : ce code assume que vous avez préalablement téléchargé
    # l'archive csdmc-spam-binary.zip à l'adresse
    # http://vision.gel.ulaval.ca/~cgagne/enseignement/apprentissage/A2018/donnees/csdmc-spam-binary.zip
    # et que vous l'avez extrait dans le répertoire courant, de telle façon
    # qu'un dossier nommé "csdmc-spam-binary" soit présent.

    X = numpy.loadtxt("data/csdmc-spam-binary/data", delimiter=",")
    y = numpy.loadtxt("data/csdmc-spam-binary/target", delimiter=",")
    with open("data/csdmc-spam-binary/features", "r") as f:
        features = [line[:-1] for line in f]

    # Division du jeu en entraînement / test
    # Ne modifiez pas la random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    _times.append(time.time())
    # TODO Q2A
    # Entraînez un classifieur SVM linéaire sur le jeu de données *complet*
    # et rapportez sa performance en test
    svm = LinearSVC()
    svm.fit(X,y)
    #y_pred = svm.predict(X)
    accuracy = 1-svm.score(X,y)
    print("Score du svm sur tout le jeu de données : " , accuracy)
    _times.append(time.time())
    checkTime(TMAX_Q2Aall, "2A (avec toutes les variables)")

    # TODO Q2A
    # Entraînez un classifieur SVM linéaire sur le jeu de données
    # en réduisant le nombre de caractéristiques (features) à 10 en
    # utilisant le chi² comme métrique et rapportez sa performance en test
    ch2 = SelectKBest(chi2, k=10)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    svm.fit(X_train,y_train)
    accuracy = 1-svm.score(X_test,y_test)
    print("Score du svm sur avec sélection de caractéristiques : ", accuracy)
    _times.append(time.time())
    checkTime(TMAX_Q2Achi, "2A (avec sous-ensemble de variables par chi2)")

    # TODO Q2A
    # Entraînez un classifieur SVM linéaire sur le jeu de données
    # en réduisant le nombre de caractéristiques (features) à 10 en utilisant
    # l'information mutuelle comme métrique et rapportez sa performance en test
    
    _times.append(time.time())
    checkTime(TMAX_Q2Amut, "2A (avec sous-ensemble de variables par mutual info)")

    # TODO Q2B
    # Entraînez un classifieur SVM linéaire sur le jeu de données
    # en réduisant le nombre de caractéristiques (features) à 10 par
    # sélection séquentielle arrière et rapportez sa performance en test

    _times.append(time.time())
    checkTime(TMAX_Q2B, "2B")

# N'écrivez pas de code à partir de cet endroit
