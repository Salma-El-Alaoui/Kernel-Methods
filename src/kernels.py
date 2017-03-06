#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 19:28:15 2017

@author: salma
"""

import numpy as np


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
        kernel += (column_1 - column_2.T)**2 / (column_1 + column_2.T)
    
    return np.exp(gamma * kernel)


def min_kernel(X, Y):
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
