# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 5, Question 3
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
# import matplotlib2tikz

import warnings
# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
warnings.filterwarnings("ignore", category=FutureWarning)

from matplotlib import pyplot

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles, make_moons

# Fonctions utilitaires liées à l'évaluation
_times = []


def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps à s'exécuter! ".format(question) +
              "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration, duration) +
              "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple à show()) dans cette boucle!")

# Définition des durées d'exécution maximales pour chaque sous-question
TMAX_Q3Ai = 60
TMAX_Q3Aii = 80
TMAX_Q3C = 130

# Ne modifiez rien avant cette ligne!

def train_adaboost(max_depth):
    print(f"\nTraining AdaBoost with decision trees of depth {max_depth}")
    n_estimators = 50
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    scores = []
    n_estimators_count = 1
    for score in clf.staged_score(X_train, y_train):
        print(f"AdaBoost train score with {n_estimators_count} estimators: {score}.")
        scores.append(score)
        n_estimators_count += 1
    fig, ax = pyplot.subplots()
    ax.plot(range(1, n_estimators + 1), scores)
    ax.set_xlabel("Nombre de classifieurs de base")
    ax.set_ylabel("Performance en entraînement")
    # matplotlib2tikz.save("train_scores_{max_depth}.pgf")
    ax.set_title(f"Performance en entraînement avec AdaBoost utilisant des arbres de profondeur {max_depth}")
    fig.show()
    scores = []
    n_estimators_count = 1
    for score in clf.staged_score(X_test, y_test):
        print(f"AdaBoost test score with {n_estimators_count} estimators: {score}.")
        scores.append(score)
        n_estimators_count += 1
    fig, ax = pyplot.subplots()
    ax.plot(range(1, n_estimators + 1), scores)
    ax.set_xlabel("Nombre de classifieurs de base")
    ax.set_ylabel("Performance en test")
    # matplotlib2tikz.save("test_scores_{max_depth}.pgf")
    ax.set_title(f"Performance en test avec AdaBoost utilisant des arbres de profondeur {max_depth}")
    fig.show()
    n_estimators_count = 1
    for predictions in clf.staged_predict(X_test):
        if n_estimators_count % (n_estimators // 3) == 0:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = numpy.meshgrid(
                numpy.arange(x_min, x_max, 0.1),
                numpy.arange(y_min, y_max, 0.1)
            )
            fig, ax = pyplot.subplots()
            Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.4)
            ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
            # matplotlib2tikz.save(f"decision_regions_{n_estimators_count}_{max_depth}.pgf")
            ax.set_title(f"Régions de décision générées par AdaBoost avec {n_estimators_count} estimateurs et des arbres de profondeur {max_depth}")
            fig.show()
        n_estimators_count += 1

if __name__ == "__main__":
    # Question 3
    # Création du jeu de données
    # Ne modifiez pas ces lignes
    X1, y1 = make_gaussian_quantiles(cov=2.2,
                                     n_samples=600, n_features=2,
                                     n_classes=2, random_state=42)
    X2, y2 = make_moons(n_samples=300, noise=0.25, random_state=42)
    X3, y3 = make_moons(n_samples=300, noise=0.3, random_state=42)
    X2[:, 0] -= 3.5
    X2[:, 1] += 3.5
    X3[:, 0] += 4.0
    X3[:, 1] += 2.0
    X = numpy.concatenate((X1, X2, X3))
    y = numpy.concatenate((y1, y2 + 2, y3 + 1))

    # Division du jeu en entraînement / test
    # Ne modifiez pas la random seedbase_estimator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

    _times.append(time.time())
    # TODO Q3A
    # Entraînez un classifieur par ensemble de type AdaBoost sur le jeu de données (X_train, y_train)
    # défini plus haut, en utilisant des souches de décision comme classifieur de base.
    # Rapportez les résultats et figures tel que demandé dans l'énoncé, sur
    # les jeux d'entraînement et de test.
    train_adaboost(max_depth=1)

    _times.append(time.time())
    checkTime(TMAX_Q3Ai, "3A avec souches")

    # TODO Q3A
    # Entraînez un classifieur par ensemble de type AdaBoost sur le jeu de données (X_train, y_train)
    # défini plus haut, en utilisant des arbres de décision de profonduer 3 comme
    # classifieur de base. Rapportez les résultats et figures tel que demandé dans l'énoncé, sur
    # les jeux d'entraînement et de test.
    train_adaboost(max_depth=3)

    _times.append(time.time())
    checkTime(TMAX_Q3Aii, "3A avec arbres de profondeur 3")

    # TODO Q3C
    # Entraînez un classifieur par ensemble de type Random Forest sur le jeu de données (X_train, y_train)
    # défini plus haut. Rapportez les résultats et figures tel que demandé dans l'énoncé, sur
    # les jeux d'entraînement et de test.

    _times.append(time.time())
    checkTime(TMAX_Q3C, "3C")

# N'écrivez pas de code à partir de cet endroit
