#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 17:16:11 2017

@author: camillejandot
"""


from svm import OneVsOneSVM
from kernels import rbf_kernel
from KernelPCA import KernelPCA
from data_utils import write_submission, load_hog_features



signed = True
equalize = True
rgb = True
n_cells_hog = 4

kernel = rbf_kernel  
kernel_pca =  rbf_kernel
classifier = "one_vs_one"

submission_name = "Yte"

print("Computing Features...")
X_train, X_test, y_train = load_hog_features(rgb=rgb, equalize=equalize, n_cells_hog=n_cells_hog)

print("Performing k-PCA...")
kpca = KernelPCA(kernel=kernel_pca, param_kernel=1.0, n_components=500)
X_train_pca = kpca.fit_transform(X_train)
X_test_pca = kpca.transform(X_test)

print("Fitting classifier on all training data...")
clf = OneVsOneSVM(C=100, kernel=kernel, kernel_param=3)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("Writing submission...")
write_submission(y_pred, submission_name)