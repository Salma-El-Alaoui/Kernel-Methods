#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:47:29 2017

@author: camillejandot
"""

from kernels import linear_kernel,laplacian_kernel,rbf_kernel
from numpy.linalg import eigh
import numpy as np

class KernelPCAOurs():
    
    def __init__(self,n_components=500,kernel='rbf',gamma=0.6):
        self.n_components = n_components
        self.kernel = kernel 
        self.gamma = gamma
        self.X = None
        self.first_eigen_vectors = None
        self.pca_kernel = None
        self.K_ones = None
        self.K_mean = None
        self.first_eigenvalues = None
        
    def fit_transform(self,X):
        if self.kernel == 'rbf':
            pca_kernel = rbf_kernel
        elif self.kernel == 'linear':
            pca_kernel = linear_kernel
        elif self.kernel == 'laplacian':
            pca_kernel = laplacian_kernel
        
        self.pca_kernel = pca_kernel
        self.X = X
        
        # Build and center Gram matrix
        gram = pca_kernel(X,X,gamma=self.gamma)
        n = X.shape[0]
        ones = np.ones((n,n)) / float(n)
        identity = np.eye(n)
        
        self.K_ones = gram.mean(axis=1)
        self.K_mean = gram.mean()
        K = np.dot(np.dot(identity-ones,gram),+identity - ones)
        
        
        # Compute the first eigenvectors and eigenvalues
        eigen_values, eigen_vectors = eigh(K)
        print("eig0 ",eigen_values)
        eigen_values = np.flipud(eigen_values)
        print("eig ",eigen_values)
        eigen_vectors = np.fliplr(eigen_vectors)
        
        eigen_values_pos = eigen_values[eigen_values > 0]
        n_pos_ev = len(eigen_values_pos)
        
        if n_pos_ev != self.n_components:
            print("Too many components. Only keeping those associated to positive eigenvalues.")
        n_components = min(self.n_components,n_pos_ev)
        first_eigen_values = eigen_values[:n_components]
        first_eigen_vectors = eigen_vectors[:,:n_components]
        
        print(first_eigen_values)
        # Normalize the eigenvectors

        first_eigen_vectors = first_eigen_vectors / np.sqrt(first_eigen_values)
        self.first_eigen_vectors = first_eigen_vectors
        self.first_eigen_values = first_eigen_values
        # Transform
        return np.dot(K, first_eigen_vectors)
        
    
    def transform(self,X):
        K = self.pca_kernel(self.X,X,gamma=self.gamma).T
        n = X.shape[0]
        print(n)
        K_cols = (np.sum(K,axis=1)/self.K_ones.shape[0])[:,np.newaxis]
        print(K_cols.shape)
        #print(K.shape)
        print(K)
        K_c = K - self.K_ones
        print(K_c)
        print(self.K_ones.shape)
        K_c = K - K_cols 
        print(K_c)
        K_c = K + self.K_mean
        print(K_c.shape)
        print((self.first_eigen_vectors).shape)
        return np.dot(K,self.first_eigen_vectors)
    
    
#%%
if __name__ == '__main__':
    X = np.random.rand(20,15)/10
    X_test = np.random.rand(4,15)
    
    kpca_ours = KernelPCAOurs(kernel="rbf", gamma = 0.6, n_components=2)
    X_pca_ours = kpca_ours.fit_transform(X)
    X_test_pca_ours = kpca_ours.transform(X_test)
    
    from sklearn.decomposition import KernelPCA
    
    kpca = KernelPCA(kernel="rbf", gamma = 0.6, n_components=2)
    X_pca = kpca.fit_transform(X)
    X_test_pca = kpca.transform(X_test)
    

    