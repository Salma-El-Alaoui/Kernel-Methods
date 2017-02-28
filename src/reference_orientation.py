#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:12:09 2017

@author: salma
"""

import numpy as np
import time



class ReferenceOrientation:

    def __init__(self, pyramid, lambda_ori=1.5):
        self.lambda_ori = lambda_ori
        self.scales = pyramid.get_scales()
        self.dogs = pyramid.create_diff_gauss()
        self.sigma_min = pyramid.get_sigma()
        self.n_scales = pyramid.get_nscales()

    def _is_inborder(self, keypoint):
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        sigma = self.sigma_min * 2 ** (scale / self.n_scales - 3)
        print("border",octave-1, scale )
        h, w = self.scales[octave-1][scale].shape[0], self.scales[octave-1][scale].shape[1]

        inf_x = 3 * self.lambda_ori * sigma
        sup_x = h - 3 * self.lambda_ori * sigma
        inf_y = 3 * self.lambda_ori * sigma
        sup_y = w - 3 * self.lambda_ori * sigma
        bool = (x >= inf_x ) and (x <= sup_x) and (y >= inf_y) and (y <= sup_y)
        return bool
    
    def get_histogram(self, keypoint, gradient):
        #time1=time.time()
        n_bins = 36
        hist = np.zeros(n_bins)
        #patch = self._get_patch(keypoint, pyramid)
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        delta = octave + 1
        sigma = self.sigma_min * 2 ** (scale / self.n_scales - 3)
        print("gradient",octave-1, scale )
        grad_m, grad_n = gradient[octave-1][scale]
        infx = 3 * self.lambda_ori * sigma
        for m in range(int((x-infx)/delta), int((x+infx)/delta)):
            for n in range(int((y-infx)/delta), int((y+infx)/delta)):
                c = np.exp(-((np.abs(m*delta-x))**2-np.abs(n*delta-y))**2)/(2* self.lambda_ori**2 * sigma**2)\
                * np.sqrt((np.abs(grad_m[m,n]))**2 + (np.abs(grad_n[m,n]))**2)
                b = n_bins/(2*np.pi) * ((np.arctan2(grad_m[m,n], grad_n[m,n]))%(2* np.pi))
                hist[int(b)]+=c
        for i in range(6):
            hist = np.convolve(hist,np.array([1,1,1])/3)
        #print("keypoint ",time.time()-time1)
        return hist
    
    
    def gradient(self):
        l = []
        for octave in self.scales:
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
        
    def compute_orientation(hist, t):
        pass
        

    