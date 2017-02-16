#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 15 14:38:54 2017

@author: salma
Multiclass SVMs (Crammer-Singer).
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def simplex_proj(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w
    

class CrammerSingerSVM():
    def __init__(self, C=1.0, max_iter=100, epsilon=0.01):
        self.C = C
        self.max_iter = max_iter
        self.epsilon = epsilon

    def _gradi(self, X, y, i):
        # TODO: replace here x_i^x_j by k(x_i, x_j) for a non-linear kernel (I think do the update using alpha instead of
        # W, but not sure (equation 4 in the paper)
        g = np.dot(X[i], self.W.T) + 1
        g[y[i]] -= 1
        return g

    def _getvi(self, g, y, i):
        min_side = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.alpha[k, i] >= self.C:
                continue
            elif k != y[i] and self.alpha[k, i] >= 0:
                continue

            min_side = min(min_side, g[k])
        return g.max() - min_side

    def _solve_dual(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.alpha[:, i]) + g / norms[i]
        z = self.C * norms[i]
        beta = simplex_proj(beta_hat, z)
        delta = Ci - self.alpha[:, i] - (beta / norms[i])
        return delta

    def _dual_decomp(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self.alpha = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.W = np.zeros((n_classes, n_features))
        # TODO non-linear kernel: replace ||x_i||^2 by k(x_i, x_i) (equation 6 in the paper)
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        ind = np.arange(n_samples)
        np.random.shuffle(ind)
        v_init = None
        for iter in range(self.max_iter):
            vsum = 0
            for ix in range(n_samples):
                i = ind[ix]
                if norms[i] == 0:
                    continue
                g = self._gradi(X, y, i)
                v = self._getvi(g, y, i)
                vsum += v
                if v < 1e-7:
                    continue
                delta = self._solve_dual(g, y, norms, i)
                self.W += (delta * X[i][:, np.newaxis]).T
                self.alpha[:, i] += delta
            if iter == 0:
                v_init = vsum
            vmax = vsum / v_init
            if vmax < self.epsilon:
                print("Convergence")
                break
        return self

    def fit(self, X, y):
        self._dual_decomp(X, y)

    def predict(self, X):
        predictions = np.argmax(np.dot(X, self.W.T), axis=1)
        return predictions


if __name__ == '__main__':

    # TODO: Refactor data related stuff
    X = np.genfromtxt('../data/Xtr.csv', delimiter=',')
    y = np.genfromtxt('../data/Ytr.csv', delimiter=',')
    X_sub = np.genfromtxt('../data/Xte.csv', delimiter=',')

    X = X[:, :-1]
    X_sub = X_sub[:, :-1]
    y = y[1:, 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    clf = CrammerSingerSVM(C=0.1, epsilon=0.01, max_iter=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("test score", accuracy_score(y_test, y_pred))
    print("it sucks")

