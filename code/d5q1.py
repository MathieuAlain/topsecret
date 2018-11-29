# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 1
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

import time
import numpy
import warnings

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
warnings.filterwarnings("ignore", category=FutureWarning)

from matplotlib import pyplot

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.datasets import load_breast_cancer
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
TMAX_Q1A = 15
TMAX_Q1B = 20
TMAX_Q1C = 20

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # Question 1
    # Chargement des données de Breast Cancer Wisconsin
    # Pas de division entraînement / test pour cette question
    X, y = load_breast_cancer(return_X_y=True)

    _times.append(time.time())
    # TODO Q1A
    # Écrivez ici une fonction nommée `evalKmeans(X, y, k)`, qui prend en paramètre
    # un jeu de données (`X` et `y`) et le nombre de clusters à utiliser `k`, et qui
    # retourne un tuple de 3 éléments, à savoir les scores basés sur :
    # - L'indice de Rand ajusté
    # - L'information mutuelle
    # - La mesure V
    # Voyez l'énoncé pour plus de détails sur ces scores.
    def evalKmeans(X, y, k):
        km = KMeans(n_clusters=k)
        y_predict = km.fit_predict(X,y)
        mesureV = v_measure_score(y,y_predict)
        randAjustee = adjusted_rand_score(y,y_predict)
        infoMutuelle = adjusted_mutual_info_score(y,y_predict)
        
        return randAjustee, infoMutuelle, mesureV
    
    pyplot.figure()
    # TODO Q1A
    # Évaluez ici la performance de K-means en utilisant la fonction `evalKmeans`
    # définie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrément de 2. Tracez les résultats obtenus sous forme de courbe
    kVec = []
    randVec = []
    mutualVec = []
    vmeasureVec = []

    for k in range(2, 51, 2):
        rand, mutual_information, v_measure = evalKmeans(X, y, k)
        randVec.append(rand)
        mutualVec.append(mutual_information)
        vmeasureVec.append(v_measure)
        kVec.append(k)
    pyplot.xlabel('Nombre de clusters k')
    pyplot.ylabel('Score')
    
    a = pyplot.plot(kVec,randVec,label='adjusted_rand_score')
    b = pyplot.plot(kVec,mutualVec,label='adjusted_rand_score')
    c =pyplot.plot(kVec,vmeasureVec,label='adjusted_rand_score')
    
    pyplot.legend([a,b,c], ['adjusted_rand_score', 'adjusted_mutual_info_score', 'v_measure_score'])
    
    _times.append(time.time())
    checkTime(TMAX_Q1A, "1A")

    # On affiche la courbe obtenue
    pyplot.show()

    _times.append(time.time())
    # TODO Q1B
    # Écrivez ici une fonction nommée `evalEM(X, y, k, init)`, qui prend en paramètre
    # un jeu de données (`X` et `y`), le nombre de clusters à utiliser `k`
    # et l'initialisation demandée ('random' ou 'kmeans') et qui
    # retourne un tuple de 3 éléments, à savoir les scores basés sur :
    # - L'indice de Rand ajusté
    # - L'information mutuelle
    # - La mesure V
    # Voyez l'énoncé pour plus de détails sur ces scores.
    # N'oubliez pas que vous devez d'abord implémenter les équations fournies à
    # la question 1B pour déterminer les étiquettes de classe, avant de passer
    # les résultats aux différentes métriques!
    def evalEM(X, y, k, init):
        GM = GaussianMixture(n_clusters=k)
        y_predict = GM.fit_predict(X,y)
        mesureV = v_measure_score(y,y_predict)
        randAjustee = adjusted_rand_score(y,y_predict)
        infoMutuelle = adjusted_mutual_info_score(y,y_predict)
        return randAjustee, infoMutuelle, mesureV

    pyplot.figure()
    # TODO Q1B
    # Évaluez ici la performance de EM en utilisant la fonction `evalEM`
    # définie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrément de 2 et en utilisant une initialisation aléatoire.
    # Tracez les résultats obtenus sous forme de courbe
    # Notez que ce calcul est assez long et devrait requérir au moins 120 secondes;
    # la limite de temps qui vous est accordée est beaucoup plus laxiste.
    for k in range(2, 51, 2):
        rand, mutual_information, v_measure = evalEM(X, y, k, 'random')

    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")

    # On affiche la courbe obtenue
    pyplot.show()

    _times.append(time.time())
    pyplot.figure()
    # TODO Q1C
    # Évaluez ici la performance de EM en utilisant la fonction `evalEM`
    # définie plus haut, en faisant varier le nombre de clusters entre 2 et 50
    # par incrément de 2 et en utilisant une initialisation par K-means.
    # Tracez les résultats obtenus sous forme de courbe
    # Notez que ce calcul est assez long et devrait requérir au moins 120 secondes;
    # la limite de temps qui vous est accordée est beaucoup plus laxiste.

    _times.append(time.time())
    checkTime(TMAX_Q1C, "1C")

    # On affiche la courbe obtenue
    pyplot.show()


# N'écrivez pas de code à partir de cet endroit
