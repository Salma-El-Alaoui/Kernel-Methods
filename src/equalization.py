#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:06:24 2017

@author: camillejandot
"""
import numpy as np
import matplotlib.pyplot as plt

def img_hist(img):
    shape = img.shape
    hist = np.zeros(256)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hist[img[i,j]] += 1
    return hist/(shape[0]*shape[1])

def cum_sum(hist):
    return [sum(hist[:i+1]) for i in range(len(hist))]

def equal_hist(img0):
    img = img0.copy()
    shape = img.shape
    hist = img_hist(img)
    cum_dist = np.array(cum_sum(hist)) 
    transf = np.uint8(255 * cum_dist) 
    img_after_eq = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            img_after_eq[i,j] = transf[img[i,j]]
    hist_after_eq = img_hist(img_after_eq)
    return img_after_eq,hist,hist_after_eq
    
def plot_histograms(hist,hist_after_eq):
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram before equalization') 
    
    plt.figure()
    plt.plot(hist_after_eq)
    plt.title('Histogram after equalization')
    
    plt.show()