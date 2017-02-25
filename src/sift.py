#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""
import numpy as np
from scipy.misc import imresize,imread
from scipy.ndimage.filters import gaussian_filter
from equalization import equalize_item
from image_utils import load_data
import matplotlib.pyplot as plt
import time

class Pyramid():
    def __init__(self,sigma=1.6,n_oct=4,k=np.sqrt(2),n_scales=5):
        self.sigma = sigma
        self.n_oct = n_oct
        self.k = k
        self.n_scales = n_scales

    def create_diff_gauss(self,img): 
        octaves = self._create_octaves(self._normalize(img))
        output = self._diff(self._blur_gauss(octaves))
        return output
        
    def _create_octaves(self,img):
        im_shape = img.shape
        list_img = []
        for i in range(self.n_oct):
            size = (int(im_shape[0]/2**i),int(im_shape[1]/2**i))
            image_l = imresize(img,size)
            list_img.append(image_l)
        return list_img
        
    def _blur_gauss(self, octaves):
        list_scaled_octave = []
        for o in octaves:
            list_scales = []
            for i in range(self.n_scales):
                list_scales.append(gaussian_filter(o,self.sigma * (self.k**(i-3)))) # a changer eventuellement
            list_scaled_octave.append(list_scales)
        return list_scaled_octave
        
    def _diff(self,list_scaled_octave):
        all_diffs = []
        for o in list_scaled_octave:
            diffs = []
            for i in range(len(o)-1):
                diffs.append(o[i+1] - o[i])
            all_diffs.append(diffs)
        return all_diffs
      
    def _normalize(self,img):
        img = img/(img.max()/255.)
        return img
        
class HarrisCorner():
    def __init__(self,sigma=1.,threshold=0.9):
        self.epsilon = 1e-5
        self.sigma = sigma
        self.threshold = threshold
    
    def get_corners(self, image):
        harris_response = self._harris_response(image)
        idx_corners = self._find_harris_points(harris_response)
        return idx_corners
        
    def _harris_response(self,img):
        im_x = gaussian_filter(img,(self.sigma,self.sigma),(0,1))
        im_y = gaussian_filter(img,(self.sigma,self.sigma),(1,0))
        W_xx = gaussian_filter(im_x * im_x,self.sigma)
        W_xy = gaussian_filter(im_x * im_y,self.sigma)
        W_yy = gaussian_filter(im_y * im_y,self.sigma)
        det = W_xx * W_yy - W_xy**2
        trace = W_xx + W_yy
        return det / (trace + self.epsilon)
    
    def _find_harris_points(self, harris_response, distance_to_borders=3):
        
        corner_threshold = harris_response.max() * self.threshold   
        idx_above_thresh = np.where((harris_response > corner_threshold))
        
        not_borders = np.zeros(harris_response.shape)
        not_borders[distance_to_borders:-distance_to_borders,distance_to_borders:-distance_to_borders] = 1
        
        idx_not_border = np.where(not_borders != 0)
        
        not_border = np.zeros(harris_response.shape)
        above_t = np.zeros(harris_response.shape)
        
        for i in range(len(idx_not_border[0])):
            not_border[idx_not_border[0][i],idx_not_border[1][i]] = 1

        for j in range(len(idx_above_thresh[0])):
            above_t[idx_above_thresh[0][j],idx_above_thresh[1][j]] = 1

        res = not_border * above_t
        ret = np.where(res != 0)
        idx_valid = list(zip(ret[0],ret[1]))
        return idx_valid     
        
class EdgeDetection():
    def __init__(self,sigma=1.,threshold=1e-6):
        self.epsilon = 1e-5
        self.sigma = sigma
        self.threshold = threshold
        
    def _find_ratio(self,img):
        im_x = gaussian_filter(img,(self.sigma,self.sigma),(0,1))
        im_y = gaussian_filter(img,(self.sigma,self.sigma),(1,0))
        W_xx = gaussian_filter(im_x * im_x,self.sigma)
        W_xy = gaussian_filter(im_x * im_y,self.sigma)
        W_yy = gaussian_filter(im_y * im_y,self.sigma)

        det = W_xx * W_yy - W_xy**2
        trace = W_xx + W_yy
        return trace * trace / (det + self.epsilon)
    
    
    def find_edges(self,img):
        ratio = self._find_ratio(img) 
        above_threshold = np.where(ratio > self.threshold * ratio.max())      
        return above_threshold
                
class Contrast():
    def __init__(self,threshold=0.3, eps=10**(-6)):
        self.threshold = threshold
        self.eps = eps
        
    def get_low_contrast(self, img):
        low_contrast=[]
        shape = img.shape
        for i in range(1,shape[0]-1):
            for j in range(1,shape[1]-1):
                window = img[i-1:i+2,j-1:j+2]
                mean = window.mean()
                std = window.std()
                if np.abs(img[i,j]-mean)/(std+self.eps)<self.threshold:
                    low_contrast.append((i,j))
        return low_contrast
                
#%%        
if __name__ == '__main__':
    #X_train, X_test, y_train = load_data()
    id_img =  108
    
    equalized_item = equalize_item(X_train[id_img],verbose=True)
    im_res = imresize(equalized_item,(256,256),interp="bilinear")
    
    pyramid = Pyramid(sigma=1.6)
    output = pyramid.create_diff_gauss(im_res)
    for i in range(len(output)):
        for j in range(len(output[0])):
            plt.figure()
            plt.imshow(output[i][j],cmap='gray')
            plt.title('Octave n° '+str(i)+ ' Scale n° '+str(j))
            
    test_image = output[0][3]
    harris = HarrisCorner(threshold=0.98)
    idx_corners = harris.get_corners(test_image)
    edges = EdgeDetection().find_edges(test_image)
    contrast = Contrast().get_low_contrast(test_image)
    plt.figure()
    #plt.imshow(test_image, cmap='gray')
    idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
    idx_contr_x, idx_contr_y = [i[0] for i in contrast], [i[1] for i in contrast]
    plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='r', s=0.1)
    plt.scatter(edges[1],edges[0], marker='o', c='r', s=0.1)
    plt.scatter(idx_contr_y, idx_contr_x, marker='o', c='g', s=0.1)


#%%
test_zz = np.zeros((256,256))
for i in range(50,100):
    for j in range(200,250):
        test_zz[i,j]=1

test_zz = imread('test.jpg')
test_zz = imresize(test_zz,(256,256,3)).mean(axis=-1)

#%%
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
test_image = output[0][3]
harris = HarrisCorner(threshold=0.99)
idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
idx_corners = harris.get_corners(test_image)
plt.figure()
plt.imshow(im_res, cmap='gray')
plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='r', s=2)