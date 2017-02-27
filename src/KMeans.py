#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:48:41 2017

@author: camillejandot
"""
import numpy as np
from random import shuffle
from numpy.linalg import norm
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self,n_clusters=200,n_iter=500):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centroids = 0
        
    def _assign_to_nearest_centroid(self,X,centroids):
        (n,d) = X.shape 
        assigned_centroids = np.zeros(n)
        # For each object...
        for i in range(n):
            obj = X[i]
            dists = np.zeros(self.n_clusters)
            # ... compute distance of the object to each centroid,...
            for i_centroid in range(len(centroids)):
                centroid = centroids[i_centroid]
                dists[i_centroid] = norm(obj-centroid,2)
            #... and assign obj to nearest centroid.
            assigned_centroids[i] = np.argmin(dists)
        return assigned_centroids
        
    def fit(self,X):
        # Initialization of centroids with points of the distribution.
        (n,d) = X.shape 
        indices = np.arange(n)
        shuffle(indices)
        centroids = X[indices[:self.n_clusters]]
        for it in range(self.n_iter):
            # Assign points to nearest centroid
            assigned_centroids = self._assign_to_nearest_centroid(X,centroids)
                    
             # For each cluster, update centroid.
            for i_centroid in range(len(centroids)):
                 indices_centroid = np.where(assigned_centroids==i_centroid)
                 points_in_cluster = X[indices_centroid]
                 centroids[i_centroid] = points_in_cluster.mean(axis=0)
                 
        self.centroids = centroids
        return self
        
    def predict(self,X):
        return self._assign_to_nearest_centroid(X,self.centroids)
            
        
if __name__ == '__main__':
    # Testing KMeans implementation
    from sklearn.datasets import make_blobs
    X,y = make_blobs()
    kmeans = Kmeans(n_clusters=3,n_iter=100)
    print(kmeans.n_clusters)
    kmeans.fit(X)

    clusters = kmeans.predict(X)
    print(y)
    print(clusters)
    plt.figure()
    for i in range(len(X)):
        if clusters[i]==0:
            plt.scatter(X[i,0],X[i,1],c='b')
        elif clusters[i]==1:
            plt.scatter(X[i,0],X[i,1],c='r')
        else:
            plt.scatter(X[i,0],X[i,1],c='g')