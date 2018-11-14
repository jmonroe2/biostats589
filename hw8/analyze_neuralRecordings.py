#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:31:43 2018

@author: jmonroe

This script exists to process nueral recording data and find groups of neurons.
"""
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
from scipy import cluster as clust



def test_stuff():
   read_dict = io.loadmat("bint_fishmovie32_100.mat")
   data = read_dict['bint']
   
   ## look at data
   # data is 297 x 160 x 953
   # corresponding to 297 iterations of 160 neurons for 953 time steps
   average_iter = np.mean(data, axis=0)
   if False:
       plt.figure()
       plt.imshow(average_iter)
   
   ## blindly FFT the data
   do_fft = False
   if do_fft:
       plt.figure()
       fourierTransform = np.fft.fft(average_iter,axis=1)
       plt.plot(np.abs(fourierTransform[0]))
       plt.xlabel("Fourier mode")
       plt.ylabel("|Fourier amplitude|")
   
   ## blindly apply k-means: look at dendrogram
   show_dendrogram = False
   if show_dendrogram:
       plt.figure()
       linked = clust.hierarchy.linkage(average_iter, 'single')
       clust.hierarchy.dendrogram(linked, orientation='top', no_labels=True, distance_sort='descending')
       plt.xlabel("Cluster index")
       plt.ylabel("Distance threshold")
       
   ## Let's look at a single cluster
   if False:
       plt.figure()
       k = 5
       scaled = clust.vq.whiten(average_iter)
       clusters,distortion = clust.vq.kmeans(scaled, k)
       for i,c in enumerate(clusters):
           plt.plot(c+i*2)
       plt.xlabel("Time")
       plt.ylabel("Cluster average firing")
       
       
   if True:
       plt.figure()
       for k in range(1,40):
           scaled = clust.vq.whiten(average_iter)
           clusters,distortion = clust.vq.kmeans(scaled, k)
           plt.plot(k,distortion,'ok')
       plt.xlabel("Num clusters")
       plt.ylabel("$\Sigma (x_i - c_i)^2$") 
       ## not sure what distortion means
   return 0;
##END test_stuff
   

def show_allTraces():
    read_dict = io.loadmat("bint_fishmovie32_100.mat")
    data = read_dict['bint']
    # data is 297 x 160 x 953
    # corresponding to 297 iterations of 160 neurons for 953 time steps
   
    traceOut_iter = np.mean(data,axis=0) ## cells across time
    #print(traceOut_iter.shape)
    plt.figure()
    plt.imshow(traceOut_iter)
    plt.ylabel("Nuerons"); plt.xlabel("time")
    
    traceOut_time = np.mean(data,axis=2) ## cell firing across time
    #print(traceOut_time.shape)
    plt.figure()
    plt.imshow(traceOut_time)
    plt.ylabel("Iteration"); plt.xlabel("cell")
    
    traceOut_cells = np.mean(data,axis=1) ## movie activity 
    #print(traceOut_cells.shape)
    plt.figure()
    plt.imshow(traceOut_cells)
    plt.ylabel("iteration"); plt.xlabel("time")
##END show_allTraces()
    
def main():
    return 0;
    show_allTraces()
##END main
    

if __name__ == '__main__':
    foo = main()
