# encoding: utf-8
# module python_perceptron.functions


"""
     Ercan Can
     Program icin gerekli olan fonksiyonlari icerir
"""

__author__ = "ercanc"

# imports
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from matplotlib.colors import ListedColormap

# functions

# 0:Iris-setosa [50], 1:Iris-versicolor[100], 2:Iris-virginica[150]
def select_data(X, type):
    if type is SelectType.setosa_50_versicolor_0:
        print "Verinin ilk 50 kaydi olan Iris-setosa [50] secildi..."
        return X[0:50]
    elif type is SelectType.setosa_10_versicolor_10:
        print "Veri icinde Iris-setosa [10], Iris-versicolor[10] adet secildi"
        return np.concatenate((X[0:10], X[50:60]), axis=0)
    elif type is SelectType.setosa_20_versicolor_20:
        print "Veri icinde Iris-setosa [20], Iris-versicolor[20] adet secildi"
        return np.concatenate((X[0:20], X[50:70]), axis=0)
    elif type is SelectType.setosa_30_versicolor_30:
        print "Veri icinde Iris-setosa [30], Iris-versicolor[30] adet secildi"
        return np.concatenate((X[0:30], X[50:80]), axis=0)
    elif type is SelectType.setosa_50_versicolor_50:
        print "Verinin ilk 50 kaydi olan Iris-setosa [50] ve  Iris-versicolor[50]  secildi..."
        return X[0:100]
    elif type is SelectType.setosa_versicolor_virginica_all:
        print "Verinin tamami secildi..."
        return X[0:150]
    else:
        print "Veri secilmedi !!!"


# Iris datasinin standardize edilmesi
def standardize(X):
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mu) / (std + 0.2)


"""
    iris[:,0] = (iris[:,0] - iris[:,0].mean()) / iris[:,0].std()
    iris[:,1] = (iris[:,1] - iris[:,1].mean()) / iris[:,1].std()
    iris[:,2] = (iris[:,2] - iris[:,2].mean()) / iris[:,2].std()
    iris[:,3] = (iris[:,3] - iris[:,3].mean()) / iris[:,3].std()
"""

# Grafik cizme
def plot_draw(X, y, pcn):
    colors = 'green,gray,yellow'  # renk listesi
    markers = 's^oxv<>'  # plot markerlar
    res = 0.01  # plot hassasiyet parametresi
    ax = plt.gca()

    marker_gen = cycle(list(markers))

    # make color map
    n_classes = np.unique(y).shape[0]
    colors = colors.split(',')
    cmap = ListedColormap(colors[:n_classes])

    # plot the decision surface
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    Z = pcn.predict(np.array([xx.ravel(), yy.ravel()]).T)

    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)

    ax.axis(xmin=xx.min(), xmax=xx.max(), y_min=yy.min(), y_max=yy.max())

    # plot class samples
    for c in np.unique(y):
        y_data = X[y == c, 1]

        ax.scatter(x=X[y == c, 0],
                   y=y_data,
                   alpha=0.8,
                   c=cmap(c),
                   marker=next(marker_gen),
                   label=c)

    legend = plt.legend(loc=4,
                        fancybox=True,
                        framealpha=0.3,
                        scatterpoints=1,
                        handletextpad=-0.25,
                        borderaxespad=0.9)
    ax.add_artist(legend)

    return ax


# classes

# Verinin secimi kriterleri
class SelectType(Enum):
    setosa_50_versicolor_0 = 1
    setosa_10_versicolor_10 = 2
    setosa_20_versicolor_20 = 3
    setosa_30_versicolor_30 = 4
    setosa_50_versicolor_50 = 5
    setosa_versicolor_virginica_all = 6