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
from sklearn.metrics.pairwise import rbf_kernel
from equalization import equalize_item
from image_utils import load_data
from HoG import hog
import pandas as pd
from data_utils import cross_validation

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
    def __init__(self, C=1.0, kernel='linear', max_iter=100, epsilon=0.01, gamma=1.0):
        self.C = C
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.gamma = gamma
        self.kernel = kernel

        
    def _gradi(self, X, y, i):
        # TODO: replace here x_i^x_j by k(x_i, x_j) for a non-linear kernel (I think do the update using alpha instead of
        # W, but not sure (equation 4 in the paper)
        if self.kernel == 'linear':
            g = np.dot(X[i], self.W.T) + 1
        elif self.kernel == 'gaussian':
            g = np.dot(self.alpha,self.K[:,i]) + 1
        else:
            print('Only linear and gaussian kernels implemented. Using linear kernel.')
            g = np.dot(X[i], self.W.T) + 1
        g[y[i]] -= 1
        return g

    def _gaussian_kernel(self,x,y):
        return np.exp(-self.gamma*np.linalg.norm(x-y)**2)
        
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
        #print("true norm ", np.sqrt(np.sum(X ** 2, axis=1)))
        if self.kernel == 'linear':
            norms = np.sqrt(np.sum(X ** 2, axis=1))
        elif self.kernel == 'gaussian':
            norms = np.zeros(len(X))
#            K = np.zeros((len(X),len(X)))
#            for i in range(len(X)):
#                for j in range(len(X)):
#                    K[i,j] = self._gaussian_kernel(X[i],X[j])
            K = rbf_kernel(X,X)
            self.K = K
            for i in range(len(X)):
                norms[i] = np.sqrt(K[i,i]) 
            self.X_train = X
        else:
            print('Only linear and gaussian kernels implemented. Using linear kernel.')
            norms = np.sqrt(np.sum(X ** 2, axis=1))
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
                if self.kernel == 'linear':
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
        if self.kernel == 'linear':
            predictions = np.argmax(np.dot(X, self.W[:, :-1].T) + self.W[:, -1], axis=1)
            
        elif self.kernel == 'gaussian':
            K = rbf_kernel(X, self.X_train)
            predictions = np.argmax(np.dot(K,self.alpha.T), axis=1) 
        return predictions



def cross_validation(X, y, nb_folds):
    subset_size = int(len(X) / nb_folds)
    for k in range(nb_folds):
        X_train = np.concatenate((X[:k * subset_size], X[(k + 1) * subset_size:]), axis=0)
        X_test = X[k * subset_size:][:subset_size]
        y_train =  np.concatenate((y[:k * subset_size], y[(k + 1) * subset_size:]), axis=0) 
        y_test = y[k * subset_size:][:subset_size]
        yield X_train, y_train, X_test, y_test


if __name__ == '__main__':
#%%
    X_train, X_test, y_train = load_data()

    hist_train = []
    for id_img in range(len(X_train)):
        image = X_train[id_img]
        img = equalize_item(image, verbose=False)
        hist_train.append(hog(img, visualise=False))


    hist_test = []
    for id_img in range(len(X_test)):
        image = X_test[id_img]
        img = equalize_item(image, verbose=False)
        hist_test.append(hog(img, visualise=False))

    hist_train_np = np.array(hist_train)
    hist_test_np = np.array(hist_test)

    X_train = np.zeros((hist_train_np.shape[0], hist_train_np.shape[1] * hist_train_np.shape[2] * hist_train_np.shape[3]))
    X_test = np.zeros((hist_test_np.shape[0], hist_test_np.shape[1] * hist_test_np.shape[2] * hist_test_np.shape[3]))

    for i in range(hist_train_np.shape[0]):
        X_train[i] = hist_train_np[i].reshape(hist_train_np.shape[1] * hist_train_np.shape[2] * hist_train_np.shape[3])

    for i in range(hist_test_np.shape[0]):
        X_test[i] = hist_test_np[i].reshape(hist_test_np.shape[1] * hist_test_np.shape[2] * hist_test_np.shape[3])

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    list_C = [0.1, 1, 10, 1000]
    #list_gamma = [0.001,0.01,0.1,1.,10.,100.]
#%%
    parameters = dict()
    for C in list_C:
        accuracies_folds = list()
        for X_train_train, y_train_train, X_valid, y_valid in cross_validation(X_train, y_train, 5):
            #X_train_train = np.concatenate((X_train_train, np.ones((len(X_train_train), 1))), axis=1)
            clf = CrammerSingerSVM(C=C, epsilon=0.0001, max_iter=500, kernel='gaussian')
            clf.fit(X_train_train, y_train_train)
            y_pred = clf.predict(X_valid)
            acc = accuracy_score(y_valid, y_pred)
            print("test score for fold", acc, ' C ', C)
            accuracies_folds.append(acc)
        print("test score for C =", C, np.mean(accuracies_folds))
        parameters[C] = np.mean(accuracies_folds)

    print(parameters)
#%%
X_train = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
clf = CrammerSingerSVM(C=0.016, epsilon=0.0001, max_iter=2000, kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#%%
df = pd.DataFrame(y_pred, columns=['Prediction'])
df.index += 1 
df['Id'] = df.index
cols = df.columns.tolist()
cols = cols[-1:] + cols[0:-1]
df = df[cols]
df.to_csv("../submission.csv", index=False)


