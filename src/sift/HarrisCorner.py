#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:58:00 2017

@author: camillejandot
"""
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class HarrisCorner():
    def __init__(self,sigma=1.,threshold=0.1):
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