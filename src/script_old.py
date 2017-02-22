#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:10:07 2017

@author: camillejandot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from image_utils import show_image, show_channel

#%%
X_train = np.genfromtxt('../data/Xtr.csv',delimiter=',')
y_train = np.genfromtxt('../data/Ytr.csv',delimiter=',')
X_test = np.genfromtxt('../data/Xte.csv',delimiter=',') 

#%%   
# Last column is a column of nans, we remove it and thus obtain 3072 columns as precised in the documentation
X_train = X_train[:,:-1]
X_test = X_test[:,:-1]
y_train = y_train[1:,1]

#%%
id_img = 70
img = X_train[id_img]
print(y_train[id_img])
show_image(img,incr_contrast=True)
show_channel(img,3)

#%%

# Convert image to grey 
image_test = X_train[id_img].copy()
#threshold = np.ones((32,32))
#print(image_test.reshape(32,32,3).mean(axis=-1).shape)
#grey_img_test = image_test.reshape(32,32,3).mean(axis=-1)
#plt.figure
#plt.imshow(grey_img_test+threshold,cmap='gray')
#original_size = image_test.shape

#%%

def img_hist(img):
    shape = img.shape
    hist = np.zeros(256)
    for i in range(shape[0]):
        for j in range(shape[1]):
            hist[img[i, j]]+= 1
    return hist/(shape[0]*shape[1])

def cum_sum(hist):
    return [sum(hist[:i+1]) for i in range(len(hist))]

def histeq(img0):
    img = img0.copy()
    shape = img.shape()
    hist = img_hist(img0)
    cum_dist = np.array(cum_sum(hist)) 
    sk = np.uint8(255 * cum_dist) #finding transfer function values ????
    Y = np.zeros_like(img)
    for i in range(shape[0]):
        for j in range(shape[1]):
            Y[i,j] = sk[img[i,j]]
    H = img_hist(Y)
    return Y,hist,H,sk
 #%%
 
#import matplotlib.image as mpimg
import numpy as np
# load image to numpy arrayb
# matplotlib 1.3.1 only supports png images
# use scipy or PIL for other formats
img = image_test.reshape(32,32,3)
min_r = np.ones((32,32))*img[:,:,0].min() 
min_g = np.ones((32,32))*img[:,:,1].min() 
min_b = np.ones((32,32))*img[:,:,2].min() 
img[:,:,0] -= min_r
img[:,:,1] -= min_g
img[:,:,2] -= min_b
show_channel(img.reshape(32*32*3),3)
#%%

img[:,:,0] *= 255.
img[:,:,1] *= 255.
img[:,:,2] *= 255.

show_channel(img.reshape(32*32*3),3)
#%%
img_r = np.uint8(img[:,:,0])
img_g = np.uint8(img[:,:,1])
img_b = np.uint8(img[:,:,2])


# convert to grayscale
# do for individual channels R, G, B, A for nongrayscale images


#img = np.uint8((0.2126* img[:,:,0]) + \
#  		np.uint8(0.7152 * img[:,:,1]) +\
#			 np.uint8(0.0722 * img[:,:,2]))


# use hist module from hist.py to perform histogram equalization
#input_r = img_r.copy()
#new_img_r, h_r, new_h_r, sk_r = histeq(input_r)

#%%
show_channel(img.reshape(32*32*3),0)

diff = img.reshape(32*32*3)[:1024].reshape(32,32)[0,31] - img[:,:,0][0,31]
print(img[:,:,0][1,10])
print(img.reshape(32*32*3)[:1024].reshape(32,32)[1,10])
#%%
# show old and new image
# show original image
plt.figure()
plt.imshow(img[:,:,0],cmap='gray')
plt.title('original image')
#%%
# show original image
plt.subplot(122)
plt.imshow(new_img_r,cmap='gray')
plt.title('hist. equalized image')
plt.show()

# plot histograms and transfer function
fig = plt.figure()
fig.add_subplot(221)
plt.plot(h_r)
plt.title('Original histogram') # original histogram

fig.add_subplot(222)
plt.plot(new_h_r)
plt.title('New histogram') #hist of eqlauized image

fig.add_subplot(223)
plt.plot(sk_r)
plt.title('Transfer function') #transfer function

plt.show()

#%%
A = np.zeros((2,2,3))
A[:,:,0] = np.array([0,1,2,3]).reshape(2,2)
#print(A[:,:,0])
A[:,:,1] = np.array([4,5,6,7]).reshape(2,2)
A[:,:,2] = np.array([8,9,10,11]).reshape(2,2)
print(A[:,:,0].shape)
#%%
def show_image_2(img,incr_contrast=True):
    image = img.copy()
    r = image[:4].reshape(2,2)
    g = image[4:8].reshape(2,2)
    b = image[8:].reshape(2,2)
    print (r,g,b)
    
    def increase_contrast_rescale(channel,incr_contrast):
        threshold = 0.5 * np.ones((2,2))
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
    
    threshold = 0.5 * np.ones((2,2))
    image_3 = np.zeros((2,2,3))
    image_3[:,:,0] = r #+ threshold
    image_3[:,:,1] = g #+ threshold
    image_3[:,:,2] = b #+ threshold
    
    plt.figure()
    plt.imshow(image_3)
    
    
def show_channel_2(img,channel):
    """
    channel 0,1,2,3 for r,g,b,all (resp.)
    -1 : only one channel in output
    """
    image = img.copy()
    if channel==0:
        r = image[:4].reshape(2,2)
        plt.figure()
        plt.imshow(r, cmap='gray')
        plt.title('red channel')
    elif channel==1:
        g = image[4:8].reshape(2,2)
        plt.figure()
        plt.imshow(g, cmap='gray')
        plt.title('green channel')
    elif channel==2:
        b = image[4:8].reshape(2,2)
        plt.figure()
        plt.imshow(b, cmap='gray')
        plt.title('blue channel')
    elif channel == 3:
        r = image[:4].reshape(2,2)
        plt.figure()
        plt.imshow(r, cmap='gray')
        plt.title('red channel')
        g = image[4:8].reshape(2,2)
        plt.figure()
        plt.imshow(g, cmap='gray')
        plt.title('green channel')
        b = image[8:].reshape(2,2)
        plt.figure()
        plt.imshow(b, cmap='gray')
        plt.title('blue channel')
    else:
        plt.figure()
        plt.imshow(image, cmap='gray')

#%%
A = np.array([0.,1.,0.,0.4,0.1,0.1,0.,0.3,0.2,0.,0.,1.])
print(A)

show_image_2(A,incr_contrast=False)