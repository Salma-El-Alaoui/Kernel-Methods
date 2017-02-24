#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""
import numpy as np
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter


class pyramid(object):
    def __init__(self):
        self.sigma = 1.6
        self.n_oct = 4
        self.k= np.sqrt(2)
        self.n_scales = 5

    def create_diff_gauss(self,img): 
        octaves = self.create_octaves(img)
        output = self.diff(self.blur_gauss(octaves))
        return output
        
    def _create_octaves(self,img):
        im_shape = img.shape
        list_img = []
        for i in range(self.n_oct):
            size = (im_shape[0]/2**i,im_shape[1]/2**i)
            image_l = imresize(img,size)
            list_img.append(image_l)
        return list_img
        
    def blur_gauss(self, octaves):
        list_scaled_octave = []
        for o in octaves:
            list_scales = []
            for i in range(self.n_scales):
                list_scales.append(gaussian_filter(o,self.sigma * (self.k**(i)))) # a changer eventuellement
            list_scaled_octave.append(list_scales)
        return list_scaled_octave
        
        
    