#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:19:43 2017

@author: camillejandot
"""
import numpy as np
from numpy import fliplr, flipud

class HistogramOrientedGradient():
    
    def __init__(self,n_cells=8,cell_size=4,n_bins=18):
        self.n_cells = n_cells #nb of cells (one 1 axis)
        self.cell_size = cell_size #nb of pixel in cell (along 1 axis)
        self.n_bins = n_bins
        self.img_size = 32
        
    def _grad(self,img):
        # Compute gradient wrt 1st dimension
        grad_x = np.zeros(img.shape)
        grad_x[:,1:-1] = img[:,2:] - img[:,:-2]
        grad_x[:,0] = img[:,1] - img[:,0]
        grad_x[:,-1] = img[:,-1] - img[:,-2]
        
        # Compute gradient wrt 2nd dimension
        grad_y = np.zeros(img.shape)
        grad_y[1:-1,:] = img[2:,:] - img[:-2,:]
        grad_y[0,:] = img[1,:] - img[0,:]
        grad_y[-1,:] = img[-1,:] - img[-2,:]
        
        return grad_x, grad_y
    
    def _compute_magnitude(self,grad_x,grad_y):
        return np.sqrt(grad_x**2 + grad_y**2)
    
    def _compute_orientation(self,grad_x,grad_y):
        orientation_rad = np.arctan2(grad_y,grad_x)
        orientation_deg = (orientation_rad * 180 / (2*np.pi)) % 360
        return orientation_deg
    
    def _bilinear_interpolation(self):
        cell_size = self.cell_size
        aux = np.zeros((cell_size,cell_size))
        for i in range(cell_size):
            for j in range(cell_size):
                aux[i,j] = (cell_size - i - 0.5)*(cell_size - j - 0.5)
        coefs = np.zeros((2*cell_size,2*cell_size))
        
        coefs[cell_size:,cell_size:] = aux
        aux_90 = np.rot90(aux)
        coefs[:cell_size,cell_size:] = aux_90
        aux_180 = np.rot90(aux_90)
        coefs[:cell_size,:cell_size] = aux_180
        aux_270 = np.rot90(aux_180)
        coefs[cell_size:,:cell_size] = aux_270
        
        coefs_for_image = np.zeros((self.img_size + cell_size,self.img_size + cell_size))
        for i in range(0,len(coefs_for_image)-cell_size,cell_size):
            for j in range(0,len(coefs_for_image)-cell_size,cell_size):
                coefs_for_image[i:i + 2*cell_size,j:j + 2*cell_size] += coefs
        
        return coefs, coefs_for_image
    
    def _orientation_interpolation(self,orientation):
        n_bins = self.n_bins
        angle_bin = 360. / n_bins
        nums_bin_low = orientation // angle_bin
        nums_bin_high = (nums_bin_low + 1.) % n_bins
        
        weights_high = (orientation % angle_bin)/float(angle_bin)
        temp_hist = np.zeros((orientation.shape[0],orientation.shape[1],n_bins))
        
        for i in range(n_bins):
            temp_hist[:,:,i] = np.where(nums_bin_low==i,(1. - weights_high),0)
            temp_hist[:,:,i] += np.where(nums_bin_high==i,weights_high,0)
        
        return temp_hist
        
    def _raw_histogram(self):
                # For testing purpose
        tableau = np.zeros((n_cells,n_cells,n_bins))
        coefs,_ = self._bilinear_interpolation
        pass
    
    def _normalize_histogram(self,raw_histogram):
        pass
    
    def build_histogram(self):
        #self.img_size = 32 # todo
        pass
        
#%%
import matplotlib.pyplot as plt
hog = HistogramOrientedGradient()
interp = hog._bilinear_interpolation()
plt.figure()
plt.imshow(interp)
print(interp[10:15,10:15])
#orientation = np.ones((32,32))*47
#print(np.unique(hog._bilinear_interpolation().flatten()).shape)
#print(hog._orientation_interpolation(orientation)[0,0,:])    