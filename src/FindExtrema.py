#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:59:07 2017

@author: camillejandot
"""
#import Pyramid
from equalization import equalize_item
from image_utils import load_data
from scipy.misc import imresize
from Pyramid import Pyramid
from matplotlib import pyplot as plt
import numpy as np

class FindExtrema():
    
    def __init__(self):
        pass

    def _neighbors_values(self,x,y,dog,dog_below,dog_above):
        neighbors_in_dog = [dog[x-1,y-1],dog[x-1,y],dog[x-1,y+1],dog[x,y-1],dog[x,y+1],dog[x+1,y-1],dog[x+1,y],dog[x+1,y+1]]
        neighbors_in_dog_above = [dog_above[x-1,y-1],dog_above[x-1,y],dog_above[x-1,y+1],dog_above[x,y-1],dog_above[x,y],dog_above[x,y+1],dog_above[x+1,y-1],dog_above[x+1,y],dog_above[x+1,y+1]]
        neighbors_in_dog_below = [dog_below[x-1,y-1],dog_below[x-1,y],dog_below[x-1,y+1],dog_below[x,y-1],dog_below[x,y],dog_below[x,y+1],dog_below[x+1,y-1],dog_below[x+1,y],dog_below[x+1,y+1]]  
        neighbors_in_dog.extend(neighbors_in_dog_below)
        neighbors_in_dog.extend(neighbors_in_dog_above)
        return neighbors_in_dog
        
    def find_extrema(self,octaves):
        all_maxima = []
        all_minima = []
        all_extrema = list()
        for i_oct,octave in enumerate(octaves):
            maxima = []
            minima = []
            for i_dog,dog in enumerate(octave[1:-1]):
                i_dog += 1
                
                for i in range(1,len(dog)-1):
                    for j in range(1,len(dog)-1):
                        neighbors_values = self._neighbors_values(i,j,dog,octave[i_dog-1],octave[i_dog+1])
                        max_neighbors_values = max(neighbors_values)
                        min_neighbors_values = min(neighbors_values)
                        # TODO: should we scale back or not?
                        if dog[i,j] >= max_neighbors_values:
                            maximum = (i*2**i_oct,j*2**i_oct)
                            maxima.append(maximum)
                            all_extrema.append((i_oct, i_dog, maximum))
                        if dog[i,j] <= min_neighbors_values:
                            minimum = (i*2**i_oct,j*2**i_oct)
                            minima.append(minimum)
                            all_extrema.append((i_oct, i_dog, minimum))
                            
            all_maxima.extend(maxima)
            all_minima.extend(minima)
            
        return all_maxima,all_minima, all_extrema
        
class ReferenceOrientation:

    def __init__(self, lambda_ori=1.5):
        self.lambda_ori = lambda_ori

    def _get_patch(self, keypoint, pyramid):
        """
        pyramid is of shape (list(ocatves)(list_scales))
        """
        dogs = pyramid.create_diff_gauss()
        scales = pyramid.get_scales()
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        delta = octave + 1
        sigma = pyramid.get_sigma() * 2**(scale / pyramid.get_nscales() - 3)
        patch = list()
        for m in range(scales[octave][scale].shape[0]):
            for n in range(scales[octave][scale].shape[1]):
                if np.max(np.abs(delta * m - x), np.abs(delta * n - y)) <= 3 * self.lambda_ori * sigma:
                    patch.append((m, n))
        return patch

    def _is_inborder(self, keypoint, pyramid):
        scales = pyramid.get_scales()
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        sigma = pyramid.get_sigma() * 2 ** (scale / pyramid.get_nscales() - 3)
        h, w = scales[octave][scale].shape[0], scales[octave][scale].shape[1]

        inf_x = 3 * self.lambda_ori * sigma
        sup_x = h - 3 * self.lambda_ori * sigma
        inf_y = 3 * self.lambda_ori * sigma
        sup_y = w - 3 * self.lambda_ori * sigma
        bool = (x >= inf_x ) and (x <= sup_x) and (y >= inf_y) and (y <= sup_y)
        return bool
    
    def get_histograms(self, keypoint, pyramid, gradient):
        n_bins = 36
        hist = np.zeros(n_bins)
        patch = self._get_patch(self, keypoint, pyramid)
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        delta = octave + 1
        sigma = pyramid.get_sigma() * 2**(scale / pyramid.get_nscales() - 3)
        grad_m, grad_n = gradient[octave-1][scale-1]
        for m,n in patch:
            c = np.exp(-((np.abs(m*delta-x))**2-np.abs(n*delta-y))**2)/(2* self.lambda_ori**2 * sigma**2)\
            * np.sqrt((np.abs(grad_m[m,n]))**2 + (np.abs(grad_n[m,n]))**2)
            b = n_bins/(2*np.pi) * ((np.arctan2(grad_m[m,n], grad_n[m,n]))%(2* np.pi))
            hist[int(b)]+=c
        for i in range(6):
            hist = np.convolve(hist,np.array([1,1,1])/3)
        return hist

    def gradient(self, scales):
        l = []
        for octave in scales:
            for scale in octave[1:3]:
                list_scales=[]
                grad_m = np.zeros((scale.shape[0]-2,scale.shape[0]-2))
                grad_n = np.zeros(grad_m.shape)
                for m in range(scale.shape[0]-2):
                    for n in range(scale.shape[1]-2):
                        grad_m[m,n] = (scale[m+1,n]- scale[m-1,n])/2.
                        grad_n[m,n] = (scale[m,n+1]- scale[m,n-1])/2.
                list_scales.append((grad_m, grad_n))
            l.append(list_scales)
        return l
                    
                        
                        
                        
# %%
if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    id_img =  108
    
    equalized_item = equalize_item(X_train[id_img],verbose=True)
    im_res = imresize(equalized_item,(256,256),interp="bilinear")
    
    pyramid = Pyramid(img=im_res, sigma=1.6, n_oct=4)
    output = pyramid.create_diff_gauss()
#    for i in range(len(output)):
#        for j in range(len(output[0])):
#            plt.figure()
#            plt.imshow(output[i][j],cmap='gray')
#            plt.title('Octave n° '+str(i)+ ' Scale n° '+str(j))         
    find_extrema = FindExtrema()
    maxima, minima, extrema = find_extrema.find_extrema(output)
       
#%%
plt.figure()
plt.imshow(im_res, cmap='gray')
idx_max_x, idx_max_y = [i[0] for i in maxima ], [i[1] for i in maxima]
idx_min_x, idx_min_y = [i[0] for i in minima ], [i[1] for i in minima]
plt.scatter(idx_max_y, idx_max_x, marker='o', c='r', s=1)
plt.scatter(idx_min_y, idx_min_x, marker='o', c='g', s=1)
plt.scatter([50],[100], marker='o', c='y', s=10)
#%%
ref = ReferenceOrientation()
pyramid = Pyramid(img=im_res, sigma=1.6, n_oct=4)
#output = pyramid.create_diff_gauss()
scales = pyramid.get_scales()
res = ref.gradient(scales)
print(res.shape)