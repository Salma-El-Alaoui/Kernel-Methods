#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:55:34 2017

@author: camillejandot
"""

import numpy as np
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter

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