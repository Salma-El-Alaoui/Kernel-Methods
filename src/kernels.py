#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:28:15 2017

@author: salma
"""

import numpy as np


def linear_kernel(X, Y, param=None):
    return np.dot(X, Y.T)


def chi2_kernel(X, Y, gamma=1.):
    """
    Chi^2 kernel, 
    K(x, y) = exp( -gamma * SUM_i (x_i - y_i)^2 / (x_i + y_i) )
    https://lear.inrialpes.fr/pubs/2007/ZMLS07/ZhangMarszalekLazebnikSchmid-IJCV07-ClassificationStudy.pdf
    (page 6)
    """
    kernel = np.zeros((X.shape[0], Y.shape[0]))

    for d in range(X.shape[1]):
        column_1 = X[:, d].reshape(-1, 1)
        column_2 = Y[:, d].reshape(-1, 1)
        kernel += (column_1 - column_2.T) ** 2 / (column_1 + column_2.T)

    return np.exp(gamma * kernel)


def min_kernel(X, Y, param=None):
    """
    Min kernel (Histogram intersection kernel)
    K(x, y) = SUM_i min(x_i, y_i)
    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """
    kernel = np.zeros((X.shape[0], Y.shape[0]))

    for d in range(X.shape[1]):
        column_1 = X[:, d].reshape(-1, 1)
        column_2 = Y[:, d].reshape(-1, 1)
        kernel += np.minimum(column_1, column_2.T)

    return kernel


def gen_min_kernel(X, Y, alpha=1.):
    """
    Generalized histogram intersection kernel
    K(x, y) = SUM_i min(|x_i|^alpha, |y_i|^alpha)
    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """
    return min_kernel(np.abs(X) ** alpha, np.abs(Y) ** alpha)


def euclidean_dist(X, Y):
    """
    matrix of pairwise squared Euclidean distances
    """
    norms_1 = (X ** 2).sum(axis=1)
    norms_2 = (Y ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(X, Y.T))


def rbf_kernel(X, Y, gamma):
    dists = euclidean_dist(X, Y)
    return np.exp(-gamma * dists)


def laplacian_kernel(X,Y, sigma=1):
    dists = euclidean_dist(X, Y)
    return np.exp(-1 / sigma * np.sqrt(dists))


def mother_wavelet(X):
    return np.cos(1.75 * X) * np.exp(-X ** 2)


def simple_wavelet_kernel(X, Y, param=0):
    mat= mother_wavelet((X-Y)/param)
    return np.prod(mat, axis=1)
    

