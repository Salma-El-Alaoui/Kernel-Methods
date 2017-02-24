#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from equalization import equalize_item
from image_utils import load_data
import matplotlib.pyplot as plt

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
    def __init__(self,sigma=1.,threshold=1e-3):
        self.epsilon = 1e-5
        self.sigma = sigma
        self.threshold = threshold
        
    def _harris_response(self,img):
        im_x = gaussian_filter(img,(self.sigma,self.sigma),(0,1))
        im_y = gaussian_filter(img,(self.sigma,self.sigma),(1,0))
        W_xx = gaussian_filter(im_x * im_x,self.sigma)
        W_xy = gaussian_filter(im_x * im_y,self.sigma)
        W_yy = gaussian_filter(im_y * im_y,self.sigma)
        det = W_xx * W_yy - W_xy**2
        trace = W_xx + W_yy
        return det / (trace + self.epsilon) 
    
    
    def _find_harris_points(self,harris_response,distance_to_borders=3):
        corner_threshold = harris_response.max() * self.threshold
        above_threshold = (harris_response > corner_threshold) #* 1

        # get coordinates of candidates
        coords = np.array(above_threshold.nonzero()).T

        # ...and their values
        candidate_values = [harris_response[c[0],c[1]] for c in coords]

        # sort candidates
        index = np.argsort(candidate_values)

        # store allowed point locations in array
        not_borders = np.zeros(harris_response.shape)
        
        not_borders[distance_to_borders:-distance_to_borders,distance_to_borders:-distance_to_borders] = 1

        # select the best points taking min_distance into account
        filtered_coords = []
        for coord in index:
            if not_borders[coords[i,0],coords[i,1]] == 1:
                filtered_coords.append(tuple(coords[i]))
                
                not_borders[(coords[i,0] - distance_to_borders):(coords[i,0] + distance_to_borders),
                        (coords[i,1] - distance_to_borders):(coords[i,1] + distance_to_borders)] = 0
        
        return filtered_coords      
    
    
if __name__ == '__main__':
    #X_train, X_test, y_train = load_data()
    id_img =  40
    
    equalized_item = equalize_item(X_train[id_img],verbose=True)
    im_res = imresize(equalized_item,(256,256),interp="bilinear")
    
    pyramid = Pyramid(sigma=1.6)
    output = pyramid.create_diff_gauss(im_res)
    for i in range(len(output)):
        for j in range(len(output[0])):
            plt.figure()
            plt.imshow(output[i][j],cmap='gray')
            plt.title('Octave n° '+str(i)+ ' Scale n° '+str(j))
    print(y_train[id_img])
        
    