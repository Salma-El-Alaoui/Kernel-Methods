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
from equalization import img_hist, cum_sum, equal_hist, plot_histograms
    

#%%
X_train = np.genfromtxt('../data/Xtr.csv',delimiter=',')
y_train = np.genfromtxt('../data/Ytr.csv',delimiter=',')
X_test = np.genfromtxt('../data/Xte.csv',delimiter=',') 

#%%   
# Last column is a column of nans, we remove it and thus obtain 3072 columns as
# precised in the documentation
X_train = X_train[:,:-1]
X_test = X_test[:,:-1]
y_train = y_train[1:,1]

#%%
id_img = 108#189
img = X_train[id_img]
print(y_train[id_img])
show_image(img,incr_contrast=True)
show_channel(img,3)


#%%

image_test = X_train[id_img].copy()

 #%%
 
import numpy as np
img = np.zeros((32,32,3))

img[:,:,0] = image_test[:1024].reshape(32,32)
img[:,:,1] = image_test[1024:2048].reshape(32,32)
img[:,:,2] = image_test[2048:].reshape(32,32)


min_r = np.ones((32,32))*img[:,:,0].min() 
min_g = np.ones((32,32))*img[:,:,1].min() 
min_b = np.ones((32,32))*img[:,:,2].min() 
img[:,:,0] -= min_r
img[:,:,1] -= min_g
img[:,:,2] -= min_b
show_channel(image_test,3)
#%%

img[:,:,0] *= 255.
img[:,:,1] *= 255.
img[:,:,2] *= 255.


#%%
img_r = np.uint8(img[:,:,0])
img_g = np.uint8(img[:,:,1])
img_b = np.uint8(img[:,:,2])

input_r = img_r.copy()
input_g = img_g.copy()
input_b = img_b.copy()

eq_img_r, hist_r, eq_hist_r = equal_hist(input_r)
eq_img_g, hist_g, eq_hist_g = equal_hist(input_g)
eq_img_b, hist_b, eq_hist_b = equal_hist(input_b)

#%%
# Original image - R channel
plt.figure()
plt.imshow(input_r,cmap='gray')
plt.title('Original image (R channel)')
#%%
# Corrected image - R channel
plt.figure()
plt.imshow(eq_img_r,cmap='gray')
plt.title('Image after histogram equalization (R channel)')
plt.show()
#%%
# Original versus corrected RGB image
plt.figure()
plt.imshow(img/255.,interpolation='bilinear')
plt.title('Image before histogram equalization (RGB)')

RGB_hist_eq = np.zeros((32,32,3))
RGB_hist_eq[:,:,0] = eq_img_r / 255.
RGB_hist_eq[:,:,1] = eq_img_g / 255.
RGB_hist_eq[:,:,2] = eq_img_b / 255.

plt.figure()
plt.imshow(RGB_hist_eq,interpolation='bilinear')
plt.title('Image after histogram equalization (RGB)')

#%%
# plot histograms and transfer function
plot_histograms(hist_r,eq_hist_r)

#%%

# Gray scale - Histogram equalization
id_img = 108
image_test = X_train[id_img].copy()
img = np.zeros((32,32,3))

img[:,:,0] = image_test[:1024].reshape(32,32)
img[:,:,1] = image_test[1024:2048].reshape(32,32)
img[:,:,2] = image_test[2048:].reshape(32,32)

min_r = np.ones((32,32))*img[:,:,0].min() 
min_g = np.ones((32,32))*img[:,:,1].min() 
min_b = np.ones((32,32))*img[:,:,2].min() 
img[:,:,0] -= min_r
img[:,:,1] -= min_g
img[:,:,2] -= min_b
img[:,:,0] *= 255.
img[:,:,1] *= 255.
img[:,:,2] *= 255.

img_grey = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
input_grey = img_grey.copy()

eq_img_grey, hist_grey, eq_hist_grey = equal_hist(input_grey)

# Histogram equalized image
plt.figure()
plt.imshow(img_grey,cmap='gray',interpolation="bilinear")
plt.title('Before equalization (grayscale)')
plt.figure()
plt.imshow(eq_img_grey,cmap='gray',interpolation="bilinear")
plt.title('After equalization (grayscale)')

#%%
# Visualize histograms
plot_histograms(hist_grey,eq_hist_grey)