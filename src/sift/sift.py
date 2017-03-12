#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:44:22 2017

@author: camillejandot
"""
import sys
import os
from scipy.misc import imresize,imread
from HarrisCorner import HarrisCorner
from EdgeDetection import EdgeDetection
from LowContrast import LowContrast
from FindExtrema import FindExtrema
from Pyramid import Pyramid
from reference_orientation import ReferenceOrientation
import numpy as np
import matplotlib.pyplot as plt
from ComputeDescriptors import ComputeDescriptors
import time


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from data_utils import load_data
from  equalization import equalize_item

class Sift():
     
    def __init__(self, interp_size=128, thresh_contrast=2., thresh_corner=0.1):
        self.interp_size = interp_size
        self.sigma_min = 1.5
        self.n_octaves = 3
        self.thresh_contrast = thresh_contrast
        self.thresh_corner = thresh_corner
  
 
    def perform_sift(self, image, verbose=False):
        equalized_item = equalize_item(image, verbose=False)
        im_res = imresize(equalized_item, (self.interp_size, self.interp_size), interp="bilinear") 
        plt.figure()
        plt.imshow(im_res, cmap="gray")
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
                    #corners in blue, edges in red, low contrast points i  green
                    s = 2 * 2**o #0.8* 2**o
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
                    s = 2 * 2**i #0.1 * 2**i
                    plt.axis('equal')
                    idx_extrema_x, idx_extrema_y = [ind[0] for ind in extrema[i][j]], [ind[1] for ind in extrema[i][j]]
                    plt.scatter(-1 * np.array(idx_extrema_y), idx_extrema_x, marker='o', c='b', s=s)
                    plt.title("Extrema for Octave " + str(i) +" scale " + str(j+1))

        print(".................Done Computing keypoints")
        return pyramid, extrema_flat



