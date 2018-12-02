# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 4
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

from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.datasets import load_digits

from scipy.spatial.distance import cdist


def plot_clustering(X_red, labels, title, savepath):
    # Tiré de https://scikit-learn.org/stable/auto_examples/cluster/plot_digits_linkage.html
    # Auteur : Gael Varoquaux
    # Distribué sous license BSD
    #
    # X_red doit être un array numpy contenant les caractéristiques (features)
    #   des données d'entrée, réduit à 2 dimensions
    #
    # labels doit être un array numpy contenant les étiquettes de chacun des
    #   éléments de X_red, dans le même ordre
    #
    # title est le titre que vous souhaitez donner à la figure
    #
    # savepath est le nom du fichier où la figure doit être sauvegardée
    #
    x_min, x_max = numpy.min(X_red, axis=0), numpy.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    pyplot.figure(figsize=(9, 6), dpi=160)
    for i in range(X_red.shape[0]):
        pyplot.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                    color=pyplot.cm.nipy_spectral(labels[i] / 10.),
                    fontdict={'weight': 'bold', 'size': 9})

    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.title(title, size=17)
    pyplot.axis('off')
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.95])
    pyplot.savefig(savepath)
    pyplot.close()

# Ne modifiez rien avant cette ligne!


if __name__ == "__main__":
    # On charge le jeu de données "digits"
    X, y = load_digits(return_X_y=True)

    # TODO Q4
    # Écrivez le code permettant de projeter le jeu de données en 2 dimensions
    # avec les classes scikit-learn suivantes : PCA, MDS et TSNE
    #savepath = 
    title = "Données réduites à 2 dimensions"
    pca = PCA(n_component = 2)
    X_red = pca.fit_transform(X)
    labels = numpy.zeros(X_red.shape[0])
    #labels = 
    
    # TODO Q4
    # Calculez le ratio entre la distance moyenne intra-classe et la distance moyenne
    # inter-classe, pour chacune des classes, pour chacune des méthodes, y compris
    # le jeu de données original. Utilisez une distance euclidienne.
    # La fonction cdist importée plus haut pourrait vous être utile

    # TODO Q4
    # Utilisez la fonction plot_clustering pour afficher les résultats des
    # différentes méthodes de réduction de dimensionnalité
    # Produisez également un graphique montrant les différents ratios de
    # distance intra/inter classe pour toutes les méthodes

# N'écrivez pas de code à partir de cet endroit
