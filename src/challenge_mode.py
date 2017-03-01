#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:48:54 2017

@author: camillejandot
"""

import numpy as np
import matplotlib.pyplot as plt
from image_utils import load_data
from equalization import equalize_item
#%%
# Load data
X_train, X_test, y_train = load_data()

#%%
# Equalization of histogram

X_train_eq = np.zeros((len(X_train),32*32))
X_test_eq = np.zeros((len(X_test),32*32))

for i in range(len(X_train)):
    X_train_eq[i] = equalize_item(X_train[i], verbose=False).reshape(32*32)

for i in range(len(X_test)):
    X_test_eq[i] = equalize_item(X_test[i], verbose=False).reshape(32*32)
    
#%%

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GridSearchCV

kpca = KernelPCA()
svc = SVC(kernel='rbf')

pipeline = Pipeline([('kpca', kpca), ('svc', svc)])

#%%

def train_test_model_clf_CV(X, y):
    X_train = X.copy()                                  
    y_train = y.copy()
    NUM_TRIALS = 4
    for i in range(NUM_TRIALS):
        print(i)
        inner_cv = KFold(n_splits=4, shuffle=True,random_state=i)
        pipe = make_pipeline(SVC(kernel='rbf')) #KernelPCA(kernel="rbf"),
        params = dict( svc__gamma=[100,1000], svc__C=[100,10000]) #kernelpca__n_components=[500],kernelpca__gamma = [10],
        print("here")
        grid_search = GridSearchCV(pipe, param_grid=params,cv=inner_cv)
        grid_search.fit(X_train,y_train)

        print(grid_search.best_score_)
        print(grid_search.best_params_)
        
train_test_model_clf_CV(X_train_eq,y_train)

#%%
plt.figure()
#mat = np.ones((256,256))
X = []
for i in range(256):
    for j in range(256):
        X.append(i)
Y = []
for i in range(256):
    for j in range(256):
        Y.append(j)
plt.imshow(np.ones((256,256)),cmap="Oranges")
plt.scatter(X,Y,s=0.2,color='b')
