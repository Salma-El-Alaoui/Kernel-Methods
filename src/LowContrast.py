#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:02:23 2017

@author: camillejandot
"""
import numpy as np

class LowContrast():
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
                