#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:43:38 2017

@author: camillejandot
"""

import numpy as np

class ComputeDescriptors():

    def __init__(self, pyramid,t=0.8,n_hist=4,n_ori=8,lambda_descr=6.):
        self.scales = pyramid.get_scales()
        self.dogs = pyramid.create_diff_gauss()
        self.n_hist = n_hist
        self.lambda_descr = lambda_descr
        self.n_ori = n_ori
        self.sigma_min = pyramid.get_sigma()
        self.n_scales = pyramid.get_nscales()
        
    def is_in_border(self, keypoint):
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        sigma = self.sigma_min * 2 ** (scale / self.n_scales - 3)
        h, w = self.scales[octave-1][scale].shape[0], self.scales[octave-1][scale].shape[1]

        inf_x = np.sqrt(2) * self.lambda_descr * sigma*(self.n_hist+1)/self.n_hist
        sup_x = h - np.sqrt(2) * self.lambda_descr * sigma*(self.n_hist+1)/self.n_hist
        inf_y = np.sqrt(2) * self.lambda_descr * sigma*(self.n_hist+1)/self.n_hist
        sup_y = w - np.sqrt(2) * self.lambda_descr * sigma*(self.n_hist+1)/self.n_hist
        bool = (x >= inf_x ) and (x <= sup_x) and (y >= inf_y) and (y <= sup_y)
        return bool
    

    
    def build_keypoint_descriptor(self,keypoint,gradient):
        octave = keypoint[0] + 1
        dog = keypoint[1]
        scale = dog - 1
        x = keypoint[2][0]
        y = keypoint[2][1]
        theta_ref = keypoint[-1]
        delta = octave + 1
        sigma = self.sigma_min * 2 ** (scale / self.n_scales - 3)
        grad_m, grad_n = gradient[octave-1][scale]
        infx = np.sqrt(2) * self.lambda_descr * sigma*(self.n_hist+1)/self.n_hist
        upper_bound = 2*self.lambda_descr/self.n_hist
        
        hist = np.zeros((self.n_hist,self.n_hist,self.n_ori))
        
        #print("delta ", delta)
        #print("infx", infx)
        for m in range(int(round((x-infx)/delta)), int(round((x+infx)/delta)+1)):
            for n in range(int(round((y-infx)/delta)), int(round((y+infx)/delta)+1)):
                #print(m,n)
                x_hat = 1./sigma * ((m*delta - x)*np.cos(theta_ref) + (n*delta - y)*np.sin(theta_ref))
                y_hat = 1./sigma * (-(m*delta - x)*np.sin(theta_ref) + (n*delta - y)*np.cos(theta_ref))
                
                if max(np.abs(x_hat),np.abs(y_hat)) < self.lambda_descr * (self.n_hist + 1) / (self.n_hist):
                    print('here')
                    theta = (np.arctan2(grad_m[m,n],grad_n[m,n]) - theta_ref) % (2*np.pi)
                    c = np.exp(-((np.abs(m*delta-x))**2-np.abs(n*delta-y))**2)/(2* self.lambda_descr**2 * sigma**2)\
                    * np.sqrt((np.abs(grad_m[m,n]))**2 + (np.abs(grad_n[m,n]))**2)
                    print(theta,c)
                    for i in range(self.n_hist):
                        x_hat_i = (i - (1. + self.n_hist)/2) * 2 * self.lambda_descr/self.n_hist
                        for j in range(self.n_hist):
                            y_hat_j = (j - (1. + self.n_hist)/2) * 2 * self.lambda_descr/self.n_hist
                            if (np.abs(x_hat_i - x_hat)<upper_bound) and (np.abs(y_hat_j - y_hat)<upper_bound):
                                for k in range(self.n_ori):
                                    theta_hat_k = 2.*np.pi *(k-1)/self.n_ori
                                    if (np.abs((theta_hat_k - theta)%(2*np.pi)))< 2 * np.pi/(self.n_ori):
                                        hist[i,j,k] += (1. - 1./upper_bound * np.abs(x_hat - x_hat_i)) \
                                        *(1. - 1./upper_bound * np.abs(y_hat - y_hat_j)) * (1.-self.n_ori/(2*np.pi)*np.abs((theta_hat_k - theta)%(2*np.pi)))*c
                    print("hist ", hist)
        f = np.zeros(self.n_hist*self.n_hist*self.n_ori)             
        for i in range(self.n_hist):
            for j in range(self.n_hist):
                for k in range(self.n_ori):
                    f[i*self.n_hist*self.n_ori + j*self.n_ori + k] = hist[i,j,k]
        
        norm_f = np.linalg.norm(f)
        for l in range(len(f)):
            f[l] = min(f[l],0.2*norm_f)
            f[l] = min(np.floor(512*f[l]/norm_f),255)
            
        return f
            
        #return keypoint + (f,)
            
        
                    
        

        
