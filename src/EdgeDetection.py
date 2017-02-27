#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:59:36 2017

@author: camillejandot
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter


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
        idx_valid = list(zip(above_threshold[0],above_threshold[1]))
        return idx_valid
                
