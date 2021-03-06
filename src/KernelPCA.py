#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:47:29 2017

@author: camillejandot
"""

from kernels import linear_kernel, laplacian_kernel, rbf_kernel
from numpy.linalg import eigh
import numpy as np


class KernelPCA():
    
    def __init__(self, n_components=500, kernel=rbf_kernel, param_kernel=0.6, apply=True):
        self.n_components = n_components
        self.param_kernel = param_kernel
        self.X = None
        self.first_eigen_vectors = None
        self.pca_kernel = kernel
        self.K_ones = None
        self.K_mean = None
        self.first_eigenvalues = None
        self.apply = apply
        
    def fit_transform(self, X):
        if not self.apply:
            return X
        else:
            self.X = X
            # Build and center Gram matrix
            gram = self.pca_kernel(X, X, self.param_kernel)
            n = X.shape[0]
            ones = np.ones((n,n)) / float(n)
            identity = np.eye(n)

            self.K_ones = gram.mean(axis=1)
            self.K_mean = gram.mean()
            K = np.dot(np.dot(identity-ones,gram),+identity - ones)

            # Compute the first eigenvectors and eigenvalues
            eigen_values, eigen_vectors = eigh(K)
            eigen_values = np.flipud(eigen_values)
            eigen_vectors = np.fliplr(eigen_vectors)

            eigen_values_pos = eigen_values[eigen_values > 0]
            n_pos_ev = len(eigen_values_pos)

            if n_pos_ev < self.n_components:
                print("Too many components. Only keeping those associated to positive eigenvalues.")

            n_components = min(self.n_components,n_pos_ev)
            first_eigen_values = eigen_values[:n_components]
            first_eigen_vectors = eigen_vectors[:,:n_components]


            # Normalize the eigenvectors
            self.first_eigen_vectors = first_eigen_vectors
            first_eigen_vectors = first_eigen_vectors / np.sqrt(first_eigen_values)
            self.first_eigen_values = first_eigen_values

            # Transform
            return np.dot(K, first_eigen_vectors)

    
    def transform(self,X):
        if not self.apply:
            return X
        else:
            K = self.pca_kernel(self.X, X, self.param_kernel).T
            K_cols = (np.sum(K, axis=1)/self.K_ones.shape[0])[:,np.newaxis]
            K_c = K - self.K_ones
            K_c = K_c - K_cols
            K_c = K_c + self.K_mean
            return np.dot(K_c, self.first_eigen_vectors/np.sqrt(self.first_eigen_values))

    
#%%
if __name__ == '__main__':
    X = np.random.rand(20, 15)/10
    X_test = np.random.rand(4, 15)
    
    kpca_ours = KernelPCA(kernel=rbf_kernel, param_kernel=0.6, n_components=2)
    X_pca_ours = kpca_ours.fit_transform(X)
    X_test_pca_ours = kpca_ours.transform(X_test)
    
    from sklearn.decomposition import KernelPCA
    
    kpca = KernelPCA(kernel="rbf", gamma=0.6, n_components=2)
    X_pca = kpca.fit_transform(X)
    X_test_pca = kpca.transform(X_test)
    
