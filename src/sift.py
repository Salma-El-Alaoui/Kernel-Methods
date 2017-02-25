#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""

from scipy.misc import imresize,imread
from equalization import equalize_item
from HarrisCorner import HarrisCorner
from EdgeDetection import EdgeDetection
from LowContrast import LowContrast
from Pyramid import Pyramid
from image_utils import load_data
import matplotlib.pyplot as plt


  
if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    id_img =  108
    
    equalized_item = equalize_item(X_train[id_img],verbose=True)
    im_res = imresize(equalized_item,(256,256),interp="bilinear")
    
    pyramid = Pyramid(sigma=1.6)
    output = pyramid.create_diff_gauss(im_res)
    for i in range(len(output)):
        for j in range(len(output[0])):
            plt.figure()
            plt.imshow(output[i][j],cmap='gray')
            plt.title('Octave n° '+ str(i)+ ' Scale n° '+ str(j))
            
    test_image = output[0][3]
    harris = HarrisCorner(threshold=0.98)
    idx_corners = harris.get_corners(test_image)
    edges = EdgeDetection().find_edges(test_image)
    contrast = LowContrast().get_low_contrast(test_image)
    plt.figure()
    #plt.imshow(test_image, cmap='gray')
    idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
    idx_contr_x, idx_contr_y = [i[0] for i in contrast], [i[1] for i in contrast]
    plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='r', s=0.1)
    plt.scatter(edges[1],edges[0], marker='o', c='r', s=0.1)
    plt.scatter(idx_contr_y, idx_contr_x, marker='o', c='g', s=0.1)


#%% Load toy image

#test_zz = np.zeros((256,256))
#for i in range(50,100):
#    for j in range(200,250):
#        test_zz[i,j]=1

test_zz = imread('test.jpg')
test_zz = imresize(test_zz,(256,256,3)).mean(axis=-1)

#%% Test corners, edges and low contrast points detection o toy image
harris = HarrisCorner(threshold=0.01)
idx_corners = harris.get_corners(test_zz)
idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
edges = EdgeDetection().find_edges(test_zz)
contrast = Contrast().get_low_contrast(test_zz)
idx_contr_x, idx_contr_y = [i[0] for i in contrast], [i[1] for i in contrast]
print(len(edges[0]))
plt.figure()
plt.imshow(test_zz, cmap="gray")    
plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='b', s=0.1)
plt.scatter(edges[1],edges[0], marker='o', c='r', s=0.1)
plt.scatter(idx_contr_y, idx_contr_x, marker='o', c='g', s=0.1)


#%%       
#test_image = output[0][3]
#harris = HarrisCorner(threshold=0.99)
#idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
#idx_corners = harris.get_corners(test_image)
#plt.figure()
#plt.imshow(im_res, cmap='gray')
#plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='r', s=2)