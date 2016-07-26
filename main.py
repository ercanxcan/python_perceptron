# encoding: utf-8
# module python_perceptron.main

"""
     Ercan Can
     Iris datasinin perceptron algoritmasi kullanilarak siniflandirilmasi
"""

__author__      = "ercanc"

# imports
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
from functions import plot_draw,standardize,select_data,SelectType

dim1 = 0              # Verinin 1 boyutu sepal length
dim2 = 2              # Verinin 3 boyutu  petal length
learning_rate=0.01    # Hedefe varmasi icin gereken adim buyuklugu
epochs=50            # Agirliklari guncelleme sayisi
showGraph=True        # Grafigi goster

iris = np.loadtxt('iris_proc.data',delimiter=',')
print('Yuklenen data : %s' % iris[0:5,:])

# Veri secimi SelectType taki seceneklere gore gerceklesiyor.
input_data = select_data(iris,SelectType.setosa_50_versicolor_50)

# Verinin son columu gercek output olarak y degiskenine aktariliyor.
y = input_data[:,4]

# Veriler normalize ediliyor.
input_data = standardize(input_data)

print('Normalize edilmis data : %s' % input_data[0:5,:])

# Xi input verisi icin boyut secimi gerceklesiyor
xi = input_data[:,[dim1,dim2]]

# Perceptron algoritmasi cagriliyor
ppn = Perceptron(epochs=epochs, eta=learning_rate)

# Perceptron ogrenme metodu calistiriliyor.
ppn.train(xi, y)
print('Agirliklar: %s' % ppn.w_)
print('Hatalar : %s' % ppn.errors_)

if showGraph:
    plot_draw(X=xi, y=y, pcn=ppn)
    plt.title('Perceptron Algoritmasi setosa_50_versicolor_50')
    plt.xlabel('SepalUzunluk[cm]')
    plt.ylabel('PetalUzunluk[cm]')
    plt.show()
