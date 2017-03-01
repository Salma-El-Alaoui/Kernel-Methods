#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""

from scipy.misc import imresize,imread
from equalization import equalize_item
from HarrisCorner import HarrisCorner
from EdgeDetection import EdgeDetection
from LowContrast import LowContrast
from FindExtrema import FindExtrema
from Pyramid import Pyramid
from reference_orientation import ReferenceOrientation
from image_utils import load_data
import numpy as np
import matplotlib.pyplot as plt
from ComputeDescriptors import ComputeDescriptors
#%%
class Sift():
     
    def __init__(self, interp_size=256, thresh_contrast=2., thresh_corner=0.1):
        self.interp_size = interp_size
        self.sigma_min = 1.5
        self.n_octaves = 4
        self.thresh_contrast = thresh_contrast
        self.thresh_corner = thresh_corner
  
 
    def perform_sift(self, image, verbose=False):
        equalized_item = equalize_item(image, verbose=False)
        im_res = imresize(equalized_item, (self.interp_size, self.interp_size), interp="bilinear") 
        pyramid = Pyramid(img=im_res, sigma=self.sigma_min, n_oct=self.n_octaves)
        dogs = pyramid.create_diff_gauss() 
        find_extrema = FindExtrema()
        extrema, extrema_flat = find_extrema.find_extrema(dogs)
        print(".................Done computing extrema")
        
        # find bad points
        bad_points = []
        for o, octave in enumerate(dogs):
            list_oct = []
            for sc, img in enumerate(octave[1:3]) :
                list_scale = []
                harris = HarrisCorner(threshold=self.thresh_corner)
                idx_corners = harris.get_corners(img)
                list_scale.extend(idx_corners)
                edges = EdgeDetection().find_edges(img)
                list_scale.extend(edges)
                contrast = LowContrast(threshold=self.thresh_contrast).get_low_contrast(img)
                list_scale.extend(contrast)
                
                # plot bad points
                
                if verbose:
                    plt.figure()
                    plt.imshow(img, cmap="gray")
                    idx_corners_x, idx_corners_y = [i[0] for i in idx_corners], [i[1] for i in idx_corners]
                    idx_edges_x, idx_edges_y = [i[0] for i in edges], [i[1] for i in edges]
                    idx_contr_x, idx_contr_y = [i[0] for i in contrast], [i[1] for i in contrast]
                    # corners in blue, edges in red, low contrast points i  green
                    s = 0.08 * 2**o
                    plt.scatter(idx_edges_y,idx_edges_x, marker='o', c='r', s=s)
                    plt.scatter(idx_contr_y, idx_contr_x, marker='o', c='g', s=s)
                    plt.scatter(idx_corners_y, idx_corners_x, marker='o', c='b', s=s)
                    plt.title(" Bad points for Octave " + str(o) +" Scale " + str(sc+1))
                list_oct.append(list_scale)
            bad_points.append(list_oct)       
        print(".................Done computing bad points")
        
        # remove bad points from extrema
        for i in range(len(bad_points)):
            for j in range(len(bad_points[0])):
                img = dogs[i][j+1] 
                a = set(extrema[i][j])
                b = set(bad_points[i][j])
                extrema[i][j] = list(a - b)
                if verbose:
                    plt.figure()
                    s = 0.1 * 2**i
                    plt.axis('equal')
                    idx_extrema_x, idx_extrema_y = [ind[0] for ind in extrema[i][j]], [ind[1] for ind in extrema[i][j]]
                    plt.scatter(-1 * np.array(idx_extrema_y), idx_extrema_x, marker='o', c='b', s=s)
                    plt.title("Extrema for Octave " + str(i) +" scale " + str(j+1))

        print(".................Done Computing keypoints")
        return pyramid, extrema_flat
       
            
#%%               
if __name__ == '__main__':

    #X_train, X_test, y_train = load_data()
    id_img =  108
    image = X_train[id_img]
#    test_zz = imread('carre.jpg')
#    print (test_zz[:,:,0].shape)
#    image = np.zeros(32*32*3)
#    image[:1024] = imresize(test_zz[:,:,0],(32,32)).reshape(32*32)
#    image[1024:2048] = imresize(test_zz[:,:,1],(32,32)).reshape(32*32)
#    image[2048:] = imresize(test_zz[:,:,2],(32,32)).reshape(32*32)
    sift = Sift(thresh_contrast=100, thresh_corner =0.9)
    pyramid, extrema_flat = sift.perform_sift(image, verbose=True)
    #%%
    ref = ReferenceOrientation(pyramid)
    gradient = ref.gradient()
    count = 0
    count_points = 0
    #sum_hist = np.zeros(36)
    keypoints = []
    for i, ext in enumerate(extrema_flat):
        print(i)
        print ("keypoint")
        if ref._is_inborder(ext):
            count_points += 1
            
            keypoint = ref.get_histogram(ext, gradient)
            print(keypoint)
            #hist = ref.get_histogram(ext, gradient)
            #if len(hist[hist!=0]) == 0:
            #    count += 1
            if keypoint != []:
                comp0 = ComputeDescriptors(pyramid)
                if comp0.is_in_border(keypoint[0]):
                    comp = comp0.build_keypoint_descriptor(keypoint[0],gradient)
                    print ("comp ",comp)
                    print (comp[comp!=0])

    #print("keypoints ",keypoints)    
    # Load toy image
    #test_zz = imread('test.jpg')
    #test_zz = imresize(test_zz,(256,256,3)).mean(axis=-1)
    


