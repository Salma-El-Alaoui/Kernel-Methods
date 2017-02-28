#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:32:29 2017

@author: camillejandot
"""
from KMeans import Kmeans
from scipy.stats import histogram
import numpy as np

class BagOfFeatures():
    
    def __init__(self,n_words):
        self.n_words = n_words
    
    def _compute_tf(self,X):
        """
        Should be computed on train for train and on test for test
        tf_doc(word) = n_occ_word_in_doc / n_words_in_doc
        TODO: see if we want to do take a log of frequency (cf. Wikipedia tf-idf)
        """
        tfs = []
        for i_img in range(len(X)):
            img = X[i_img]
            hist = histogram(img,numbins=self.n_words)[0]
            hist /= float(len(img))
            tfs.append(hist)
        return tfs
        
    def _compute_idf(self,X):
        """
        Should be computed on train 
        idf(word) = log(1+n_doc/n_doc_where_word_appears)
        """
        n_img = len(X)
        idfs = np.zeros(self.n_words)
        for i_word in range(self.n_words):
            for i_img in range(n_img):
                # If word of idx i_word appears in image, increment idf
                if i_word in X[i_img]:
                    idfs[i_word] += 1.
        idfs = np.log(1. + n_img * np.ones(len(idfs)) / idfs)
        return idfs
            
    def _words_pred_per_img(self,words_pred,n_ft_per_img):
        count_words = []
        begin = 0
        for i_n_features in range(len(n_ft_per_img)):
            n_features = n_ft_per_img[i_n_features]
            count_words.append(words_pred[begin:begin+n_features])
            begin += n_features
        return count_words
        
    def tf_idf(self,X_train,X_test,n_ft_per_img_train,n_ft_per_img_test):
        kmeans = Kmeans(n_clusters=self.n_words)
        kmeans.fit(X_train)
        words_pred_train = kmeans.predict(X_train)
        words_pred_test = kmeans.predict(X_test)
        words_train_per_img = np.array(self._words_pred_per_img(words_pred_train,n_ft_per_img_train))
        words_test_per_img = np.array(self._words_pred_per_img(words_pred_test,n_ft_per_img_test))
        tfs_train = self._compute_tf(words_train_per_img)
        tfs_test = self._compute_tf(words_test_per_img)
        idfs = np.nan_to_num(self._compute_idf(words_train_per_img))
        tf_idf_train = np.array(tfs_train) * np.array(idfs)
        tf_idf_test = np.array(tfs_test) * np.array(idfs)
        return tf_idf_train,tf_idf_test
   
if __name__ == '__main__':
    
    X_train = np.array([[1,2,4,6,2,1,10],[1,1,19,98,11,12,12],[1,5,2,3,3,2,4],[4,0,0,9,-3,-8,8],[12,12,12,9,-9,0,0],[12,12,12,9,-9,1,1],[12,12,12,9,-9,2,1]])
    X_test = np.array([[1,7,9,6,23,1,13],[13,-1,19,98,11,102,12],[-1,5,2,31,3,2,4],[4,10,0,9,-3,-8,8],[12,2,12,9,-9,0,0]])
    tfidf = BagOfFeatures(n_words=2)
    tf_idf_train,tf_idf_test = tfidf.tf_idf(X_train,X_test,[2,2,1],[3,2])
    print(tf_idf_train)
    print(tf_idf_test)