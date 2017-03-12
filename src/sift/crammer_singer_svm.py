#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Feb 15 14:38:54 2017
@author: salma
Multiclass SVMs (Crammer-Singer).
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from data_utils import cross_validation
from kernels import rbf_kernel, linear_kernel
import operator
from KernelPCA import KernelPCA

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
    def __init__(self, C=1.0, kernel=linear_kernel, max_iter=500, epsilon=0.0001, param_kernel=None):
        self.C = C
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.param_kernel = param_kernel
        self.kernel = kernel

    def _gradi(self, X, y, i):
        # TODO: replace here x_i^x_j by k(x_i, x_j) for a non-linear kernel (I think do the update using alpha instead of
        # W, but not sure (equation 4 in the paper)
        g = np.dot(self.alpha, self.K[:,i]) + 1
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
        norms = np.zeros(len(X))
        K = self.kernel(X,X, self.param_kernel)
        self.K = K
        for i in range(len(X)):
            norms[i] = np.sqrt(K[i,i])
        self.X_train = X
        ind = np.arange(n_samples)
        np.random.shuffle(ind) 
        v_init = None
        for iter in range(self.max_iter):
            if iter % 50 == 0:
                print("**iter ",iter)
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
        K = self.kernel(X, self.X_train, self.param_kernel)
        predictions = np.argmax(np.dot(K, self.alpha.T), axis=1)
        return predictions

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def grid_search_crammer_singer(X_train, y_train, dict_param, nb_folds, verbose=True):
    parameters = dict()
    for C in dict_param['C']:
        for kernel_param in dict_param['kernel_param']:
            for nb_components in dict_param['nb_components']:
                for kernel_param_pca in dict_param['kernel_param_pca']:
                    accuracies_folds = list()
                    for X_train_train, y_train_train, X_valid, y_valid in cross_validation(X_train, y_train, nb_folds):
                        kpca = KernelPCA(kernel=dict_param['kernel_pca'], param_kernel=kernel_param_pca, n_components=nb_components, apply=dict_param['apply_pca'])
                        X_train_train = kpca.fit_transform(X_train_train)
                        X_valid = kpca.transform(X_valid)
                        clf = CrammerSingerSVM(C=C, kernel=dict_param['kernel'], param_kernel=kernel_param)
                        clf.fit(X_train_train, y_train_train)
                        acc = clf.score(X_valid, y_valid)
                        accuracies_folds.append(acc)
                    if verbose:
                        print("\tC = ", C, "kernel param = ", kernel_param, "kernel param for PCA = ", kernel_param_pca, "nb_components = ", nb_components, "---- score = ", np.mean(accuracies_folds))
                    parameters[("C", C, "kernel_param", + kernel_param, "kernel_param_pca", kernel_param_pca, "nb_components", nb_components)] = np.mean(accuracies_folds)
    best_param = max(parameters.items(), key=operator.itemgetter(1))[0]
    if verbose:
        print("\tThe best set of parameters is: ", best_param)
    return parameters, best_param

if __name__ == '__main__':
    pass


