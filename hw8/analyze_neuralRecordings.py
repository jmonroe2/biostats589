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
#plt.style.use('jtm_style')
=======
plt.style.use("default")

>>>>>>> f3f0bcbd92e4ce1216f09409b6df384177033919


def test_stuff(data):
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
       
   ## find appropriate number of k      
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
   

def show_allTraces(show=False):
   
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

    if show: plt.show()
##END show_allTraces()

def corr(x1, x2, num_avg, max_tau,offset=0):
    ## x1 needs to be at least num_avg + max_tau
    ## x2 needs to be at least num_avg
    out = np.zeros(max_tau)
    for i in range(max_tau):
        pairwise_product = x1[offset+i:num_avg+i] * x2[offset:num_avg]
        out[i] = sum(pairwise_product) 
    return out
##END corr
  

def cf_rank(data):
    avg_over_runs = np.mean(data,axis=0)
    #plt.imshow(avg_over_runs)

    def avg_first_n(xs, n=200):
        return np.mean(xs[710:730])

    sorted_avg = list(avg_over_runs)
    sorted_avg.sort(key=avg_first_n, reverse=True)
    sorted_avg= np.array(sorted_avg)

    ## show sorted list
    plt.figure()
    plt.imshow(sorted_avg)
    plt.xlabel("Time step")
    plt.ylabel("Neuron index (sorted")

    ## measure some correlations
    plt.figure()
    num_neurons = sorted_avg.shape[0]
    corr_image = np.zeros((num_neurons,30))
    base = sorted_avg[0]
    for i,neuron in enumerate(sorted_avg):
        line = corr(base, neuron, num_avg=20,max_tau = 30, offset=0)
        corr_image[i] = line
    
    plt.imshow(corr_image)
    plt.show()
## cf_rank


def regional_fft(data):
    avg_over_runs = np.mean(data,axis=0)

    ## cut out region 
    region_min = 125
    region_max = 325
    sliced = avg_over_runs[:, region_min:region_max]
    print(sliced.shape)

<<<<<<< HEAD
    plt.figure()
    for i,neuron in enumerate(sliced[:5]):
        plt.plot( neuron )
=======
    fig, ax_time = plt.subplots()
    fig, ax_freq = plt.subplots()
    
>>>>>>> f3f0bcbd92e4ce1216f09409b6df384177033919
    plt.show()
##END regional_fft


def question1(data):
    '''
    V   find a block
    V   measure correlation funciton of blocks
	rank based on averaged correlation (ie poor man's k-means)
	FFT of those --> matches?
	histogram correlations
	cluster correlations
	cluster time bins 
	do time-bin clusters match correlation time-bins?
    '''
    avg_over_runs = np.mean(data,axis=0)

    ## 1a: cut out region 
    # this region is the first burst of activity
    region_min = 125
    region_max = 325
    length = region_max-region_min
    sliced = avg_over_runs[:, region_min:region_max]
    #plt.imshow(sliced)
       
    ## 1b: measure correlation function of ith block to jth block
    num_neurons = sliced.shape[0]
    num_time = sliced.shape[1]
    max_tau = 20
    full_corr = np.zeros((num_neurons,num_neurons, max_tau))
    
    # things go a bit faster if we load saved data
    avg_corr_name = f"data/avg_corr_tau_{max_tau}.txt"
    full_corr_name =  f"data/outcome_{max_tau}.npy"
    if os.path.exists(avg_corr_name):
        avg_corr = np.loadtxt(avg_corr_name)
        full_corr= np.load(full_corr_name)
    else:
        for i,time_trace in enumerate(sliced):
            for j,compare in enumerate(sliced[:]):
                g2 = corr(time_trace, compare,num_time-max_tau,max_tau)
                full_corr[i,j,:] = g2
        avg_corr = np.mean(full_corr,axis=2)
        np.savetxt(avg_corr_name, avg_corr)
        np.save(full_corr_name, full_corr)
    
    #plt.imshow(avg_corr, origin='lower',vmin=0,vmax=5) ## chose color range based on hist of data
    #plt.title(f"full cut average, $\\tau={max_tau}$")
    
    ## 1c: rank based on proximity
    # How different are these nuerons? To see if there are clusters we'll use cross-correlation as the 'distance metric'. We've calcualted all the 'distances' above, now let's see if there are any clear cutoffs by looking at the histogram
    plt.hist(avg_corr.flatten(),bins=20, log=True)
    plt.title("cross correlation histogram")

    # as a reference, the self-correlation line is here:
    plt.figure()
    self_corr = np.diag(avg_corr)
    plt.plot( self_corr )

    
    ## 1d: look at the averages. Do they feel correlated?
         

    plt.show()
        
##END question1

 
def main():
    read_dict = io.loadmat("bint_fishmovie32_100.mat")
    data = read_dict['bint']
    # data is 297 x 160 x 953
    # corresponding to 297 iterations of 160 neurons for 953 time steps

    ## action 0: play with data
    #show_allTraces(data, True)
    #cf_rank(data)    
    regional_fft(data)

    ## action 1: answer questions (see writeup_questions.txt)
    #question1(data)
##END main
    

if __name__ == '__main__':
    foo = main()
