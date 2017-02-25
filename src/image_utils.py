#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:07:38 2017

@author: camillejandot
"""
import numpy as np
import matplotlib.pyplot as plt

def show_image(img,incr_contrast=True):
    image = img.copy()
    r = image[:1024].reshape(32,32)
    g = image[1024:2048].reshape(32,32)
    b = image[2048:].reshape(32,32)
    
    def increase_contrast_rescale(channel,incr_contrast):
        threshold = 0.5 * np.ones((32,32))
        channel += threshold
        if incr_contrast:
            delta = np.power(channel,2.5)
            channel += delta 
        n_over = 0
        n_under = 0
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if channel[i,j]>1.:
                    channel[i,j] = 1.
                    n_over += 1
                elif channel[i,j]<0.:
                    channel[i,j] = 0.
                    n_under += 1
        #print("Contrast over ",n_over)
        #print("Contrast under ",n_under)
        return channel
    
    r = increase_contrast_rescale(r,incr_contrast)
    g = increase_contrast_rescale(g,incr_contrast)
    b = increase_contrast_rescale(b,incr_contrast)
    
    threshold = 0.5 * np.ones((32,32))
    image_3 = np.zeros((32,32,3))
    image_3[:,:,0] = r #+ threshold
    image_3[:,:,1] = g #+ threshold
    image_3[:,:,2] = b #+ threshold
    
    plt.figure()
    plt.imshow(image_3)
    
    
def show_channel(img,channel):
    """
    channel 0,1,2,3 for r,g,b,all (resp.)
    -1 : only one channel in output
    """
    image = img.copy()
    if channel==0:
        r = image[:1024].reshape(32,32)
        plt.figure()
        plt.imshow(r, cmap='gray')
        plt.title('red channel')
    elif channel==1:
        g = image[1024:2048].reshape(32,32)
        plt.figure()
        plt.imshow(g, cmap='gray')
        plt.title('green channel')
    elif channel==2:
        b = image[2048:].reshape(32,32)
        plt.figure()
        plt.imshow(b, cmap='gray')
        plt.title('blue channel')
    elif channel == 3:
        r = image[:1024].reshape(32,32)
        plt.figure()
        plt.imshow(r, cmap='gray')
        plt.title('red channel')
        g = image[1024:2048].reshape(32,32)
        plt.figure()
        plt.imshow(g, cmap='gray')
        plt.title('green channel')
        b = image[2048:].reshape(32,32)
        plt.figure()
        plt.imshow(b, cmap='gray')
        plt.title('blue channel')
    else:
        plt.figure()
        plt.imshow(image, cmap='gray')

def load_data():
    X_train = np.genfromtxt('../data/Xtr.csv',delimiter=',')
    print("X_train loaded")
    y_train = np.genfromtxt('../data/Ytr.csv',delimiter=',')
    print("y_train loaded")
    X_test = np.genfromtxt('../data/Xte.csv',delimiter=',') 
    print("X_test loaded")
    
    X_train = X_train[:,:-1]
    X_test = X_test[:,:-1]
    y_train = y_train[1:,1]
    
    return X_train, X_test, y_train