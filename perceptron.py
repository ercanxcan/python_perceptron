# encoding: utf-8
# module python_perceptron.main

"""
     Ercan Can
     Perceptron algoritmasi
"""

__author__      = "ercanc"

# imports
import numpy as np

# classes

class Perceptron(object):
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        # +1 ile bias tanimi yapilir. X.shape[1] ile 2 w belirtilir.
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        # Wij (t+1) = Wij (t) + eta . err (p)
        for i in range(self.epochs):
            errors = 0

            for xi, target in zip(X, y):
                err = target - self.predict(xi)
                calculation = self.eta * err

                # w1,w2 weights
                self.w_[1:] += calculation * xi

                # bias weight
                self.w_[0] += calculation

                errors += int(calculation != 0.0)

            self.print_weights(self.w_, i)

            self.errors_.append(errors)

        return self

    # net input hesaplama sigma(xi*wi)+bias
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def print_weights(self, w, i):
        print("%s. %s" % (i, w))

