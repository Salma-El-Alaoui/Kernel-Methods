#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 10:59:07 2017

@author: camillejandot
"""

class FindExtrema():
    
    def __init__(self):
        pass

    def _neighbors_values(self,x,y,dog,dog_below,dog_above):
        neighbors_in_dog = [dog[x-1,y-1],dog[x-1,y],dog[x-1,y+1],dog[x,y-1],dog[x,y+1],dog[x+1,y-1],dog[x+1,y],dog[x+1,y+1]]
        neighbors_in_dog_above = [dog_above[x-1,y-1],dog_above[x-1,y],dog_above[x-1,y+1],dog_above[x,y-1],dog_above[x,y],dog_above[x,y+1],dog_above[x+1,y-1],dog_above[x+1,y],dog_above[x+1,y+1]]
        neighbors_in_dog_below = [dog_below[x-1,y-1],dog_below[x-1,y],dog_below[x-1,y+1],dog_below[x,y-1],dog_below[x,y],dog_below[x,y+1],dog_below[x+1,y-1],dog_below[x+1,y],dog_below[x+1,y+1]]  
        neighbors_in_dog.extend(neighbors_in_dog_below)
        neighbors_in_dog.extend(neighbors_in_dog_above)
        return neighbors_in_dog
        
    def find_extrema(self,octaves):
        all_extrema_flat = []
        all_extrema = list()
        for i_oct,octave in enumerate(octaves):
            extrema_oct = []
            for i_dog,dog in enumerate(octave[1:-1]):
                i_dog += 1
                extr =[]
                for i in range(1,len(dog)-1):
                    for j in range(1,len(dog)-1):
                        neighbors_values = self._neighbors_values(i,j,dog,octave[i_dog-1],octave[i_dog+1])
                        max_neighbors_values = max(neighbors_values)
                        min_neighbors_values = min(neighbors_values)
                        # TODO: should we scale back or not?
                        if dog[i,j] >= max_neighbors_values:
                            #maximum = (i*2**i_oct,j*2**i_oct)
                            maximum = (i,j)
                            extr.append(maximum)
                            all_extrema_flat.append((i_oct, i_dog, maximum))
                        if dog[i,j] <= min_neighbors_values:
                            #minimum = (i*2**i_oct,j*2**i_oct)
                            minimum = (i,j)
                            extr.append(minimum)
                            all_extrema_flat.append((i_oct, i_dog, minimum))                
                extrema_oct.append(extr)
            all_extrema.append(extrema_oct)
            
        return all_extrema, all_extrema_flat
        
