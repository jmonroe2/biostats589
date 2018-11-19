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
from scipy import signal
import scipy.io as io
from sklearn.cluster import SpectralClustering
import networkx as nx
np.random.seed(1000)

#plt.style.use('jtm_style')

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
       plt.xlabel("Fourier Mode")
       plt.ylabel("|Fourier amplitude|")
   
   ## blindly apply k-means: look at dendrogram
   show_dendrogram = False
   if show_dendrogram:
       plt.figure()
       linked = clust.hierarchy.linkage(average_iter, 'single')
       clust.hierarchy.dendrogram(linked, orientation='top', no_labels=True, distance_sort='descending')
       plt.xlabel("Cluster Index")
       plt.ylabel("Distance Threshold")
       
   ## Let's look at a single cluster
   if False:
       plt.figure()
       k = 5
       scaled = clust.vq.whiten(average_iter)
       clusters,distortion = clust.vq.kmeans(scaled, k)
       for i,c in enumerate(clusters):
           plt.plot(c+i*2)
       plt.xlabel("Time")
       plt.ylabel("Cluster Average Firing")
       
   ## find appropriate number of k      
   if True:
       plt.figure()
       for k in range(1,40):
           scaled = clust.vq.whiten(average_iter)
           clusters,distortion = clust.vq.kmeans(scaled, k)
           plt.plot(k,distortion,'ok')
       plt.xlabel("Num Clusters")
       plt.ylabel("$\Sigma (x_i - c_i)^2$") 
       ## not sure what distortion means
   return 0;
##END test_stuff
   

def show_allTraces(data,show=False):
   
    traceOut_iter = np.mean(data,axis=0) ## cells across time
    #print(traceOut_iter.shape)
    plt.figure()
    plt.imshow(traceOut_iter)
    plt.ylabel("Nuerons"); plt.xlabel("Time")
    
    traceOut_time = np.mean(data,axis=2) ## cell firing across time
    #print(traceOut_time.shape)
    plt.figure()
    plt.imshow(traceOut_time)
    plt.ylabel("Iteration"); plt.xlabel("Cell")
    
    traceOut_cells = np.mean(data,axis=1) ## movie activity 
    #print(traceOut_cells.shape)
    plt.figure()
    plt.imshow(traceOut_cells)
    plt.ylabel("Iteration"); plt.xlabel("Time")

    if show: plt.show()
##END show_allTraces()

def corr(x1, x2, num_avg, max_tau,offset=0,block=1,verbose=False):
    ## x1 needs to be at least num_avg + max_tau
    ## x2 needs to be at least num_avg
    out = np.zeros(max_tau)
    if verbose: plt.figure()
    for i in range(max_tau):
        moving = x1[offset+i*block:num_avg+i*block+offset]
        fixed  = x2[offset:num_avg+offset]
        pairwise_product = moving*fixed
        out[i] = sum(pairwise_product) 
        if verbose: plt.plot(moving+(i+1)*1.,'b')
    ##END loop in tau
    if verbose: 
        plt.plot(fixed,'r')
        plt.title("moving average for correlation")
    return out
##END corr
  

def cf_rank(data,show=False):
    avg_over_runs = np.mean(data,axis=0)
    #plt.imshow(avg_over_runs)

    def avg_first_n(xs):
        return np.mean(xs[710:730])

    sorted_avg = list(avg_over_runs)
    sorted_avg.sort(key=avg_first_n, reverse=True)
    sorted_avg= np.array(sorted_avg)

    ## show sorted list
    plt.figure()
    plt.imshow(sorted_avg)
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index (sorted)")

    ## measure some correlations
    plt.figure()
    num_neurons = sorted_avg.shape[0]
    corr_image = np.zeros((num_neurons,30))
    base = sorted_avg[0]
    for i,neuron in enumerate(sorted_avg):
        line = corr(base, neuron, num_avg=20,max_tau = 30, offset=0)
        corr_image[i] = line
   
    if show: 
        plt.imshow(corr_image)
        plt.show()

    return sorted_avg
## cf_rank


def regional_fft(data):
    avg_over_runs = np.mean(data,axis=0)

    ## cut out region 
    region_min = 125
    region_max = 325
    sliced = avg_over_runs[:, region_min:region_max]
    sliced_fft = np.real(np.fft.fft(sliced,axis=0))
    '''
    plt.figure()
    plt.imshow(sliced)
    plt.title("Time traces in active region #1")
    plt.figure()
    plt.imshow(sliced_fft,vmax=4)
    plt.title("FFT of time traces in active region #1")
    plt.show()  ; return
    #'''

    fig, ax_time = plt.subplots()
    fig, ax_freq = plt.subplots()
    for i,neuron in enumerate(sliced[:5]):
        fft = np.real(np.fft.fft(neuron))
        ax_time.plot( neuron +i )
        ax_freq.plot( fft +2*i )
    ax_freq.set_xlabel("FFT Mode")
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("Spike Amplitude [offset]")
    ax_freq.set_ylabel("Mode Amplitude [offset]")
    ax_time.set_title("Time domain measurements of first 5 neurons [arb order]")
    ax_freq.set_title("Frequency domain measurements of first 5 neurons [arb order]")
    
    plt.show()
##END regional_fft


def question1(data):
    '''
    V   find a block
    V   measure correlation funciton of blocks
	X   rank based on averaged correlation (ie poor man's k-means)
	X   FFT of those --> matches?
	V    histogram correlations
	d cluster correlations
	e cluster time bins 
	f do time-bin clusters match correlation time-bins?
    '''
    avg_over_runs = np.mean(data,axis=0)
    

    ## 1a: cut out region 
    # this region is the first burst of activity
    #region_min, region_max = 0,110
    #region_min, region_max = 125,325
    region_min, region_max = 710,730
    do_sort = True
    if do_sort:
        def avg_first_n(xs):
            return np.mean(xs[0:region_max])

        sorted_avg = list(avg_over_runs)
        sorted_avg.sort(key=avg_first_n, reverse=True)
        avg_over_runs = np.array(sorted_avg)
    ##END do_sort 
    sliced = avg_over_runs[:, region_min:region_max]
    plt.figure()
    plt.imshow(sliced,aspect='auto')
    plt.xlabel("Time")
    plt.ylabel("Neuron label")
    
   
    ## 1b: measure correlation function of ith block to jth block
    num_neurons = sliced.shape[0]
    num_time = sliced.shape[1]
    max_tau = 5
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
                g2 = corr(time_trace, compare,num_avg=num_time-max_tau,max_tau=max_tau)
                #g2 = corr(time_trace, compare,num_avg=20,max_tau=max_tau,block=num_time//max_tau)
                if max(g2):g2 /= max(g2)
                full_corr[i,j,:] = g2
                
        avg_corr = np.mean(full_corr,axis=2)
        np.savetxt(avg_corr_name, avg_corr)
        np.save(full_corr_name, full_corr)
    ##'''
    num = 5
    # plot time traces
    plt.figure()
    for i,time_trace in enumerate(avg_over_runs[:num, region_min:region_max]):
        plt.plot(time_trace+i)
    plt.xlabel("Time")
    plt.ylabel("Neuron Signal")
    plt.title("First 5 Neurons in Quiesscent Region")

    # plot correlation
    plt.figure()
    for i in range(num):
        ith_corr = full_corr[0,i,:]
        plt.plot( ith_corr+i)
    plt.xlabel("Correlation Delay Time")
    plt.ylabel("Correlation with $N_0$[norm'd to max]")
    #'''

    #'''
    plt.figure()
    plt.imshow(avg_corr, origin='lower')
    plt.title(f"Average $<N_i(t)N_j(t+\\tau)>$ for $\\tau$={max_tau}")
    plt.xlabel("Neuron index i")
    plt.ylabel("Neuron index j")
    #'''
   
 
    ## 1c: rank based on proximity
    # How different are these nuerons? To see if there are clusters we'll use cross-correlation as the 'distance metric'. We've calcualted all the 'distances' above, now let's see if there are any clear cutoffs by looking at the histogram
    plt.figure()
    #xs = plt.hist(avg_corr[:10],bins=20,log=True) ## hist for each of 160 bins
    plt.hist(avg_corr.flatten(),bins=20,log=True) 
    plt.title("Cross Correlation Histogram")
    plt.xlabel("Correlation")

    # as a reference, the self-correlation line is here:
    plt.figure()
    self_corr = np.diag(avg_corr)
    plt.plot( self_corr )
    plt.xlabel("Neuron Index")
    plt.ylabel(f"Average Autocorrelation (max $\\tau$={max_tau})")


    ## 1d: Let's cluster these with spectral clustering (i.e. k-means for graphs)
    sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
    sc.fit(avg_corr)
    colors = ['rbgcym'[i] for i in sc.labels_]

    plt.figure()
    G = nx.Graph(avg_corr)
    pos = nx.spring_layout(G,k=None, iterations=50)
    nx.draw_networkx_nodes(G,pos, nodesize=10,node_color=colors)
    nx.draw_networkx_edges(G,pos,alpha=0.2)
    nx.draw_networkx_labels(G,pos, font_color='w',font_size=12)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return sc 
        
##END question1


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def question4(data):
    ## question 4: can we see the periodic and not periodic groups?


    ## 4a does our correlation function make sense?
    test_data = np.array([[2, 2, 2, 1, 1, 1, 1, 1, 1,1,1,1],
              [1, 1, 1, 2, 2, 2, 1, 1, 1,1,1,1],
              [1, 1, 1, 1, 1, 1, 2, 2, 2,1,1,1]])


    g2 = corr(test_data[0], test_data[0], num_avg=6,max_tau=3,block=3)
    #print(g2)
    g2 = corr(test_data[1], test_data[0], num_avg=6,max_tau=3,block=3)
    #print(g2)

    ## 4b: do we see clusters in histogramed data?
    box_smooth = 10
    avg_over_runs = np.mean(data,axis=0)
    fig, ax1 = plt.subplots()
    fig, ax2 = plt.subplots()
    avg_g2 = np.zeros(len(avg_over_runs))
    for i,trace in enumerate(avg_over_runs):
        smoothed = smooth(trace,box_smooth)
        g2 = corr(trace, trace, num_avg=100, max_tau=25,offset=100, block=30)
        if i<10:
            ax1.plot(smoothed+i*0.5)
            ax2.plot(g2)
        avg_g2[i] = np.mean(g2)
    ax2.set_xlabel("Delay Time")
    ax2.set_ylabel("Cross Correlation")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Smoothed Neuron Signal [offset]")
    
    fig, hist_ax = plt.subplots()
    hist_ax.hist(avg_g2,bins=30,log=True)
    hist_ax.set_xlabel("Average Auto-correlation")
    
    ## 4c check that fake signal shows up in analysis
    length = 100
    fake_signal = 0.2*np.ones(length) 
    fake_trace = np.zeros(data.shape[2])
    start_indices = [100, 420, 670]
    for start in start_indices:
        fake_trace[start:start+length] = fake_signal + 0.25*np.random.randn(length)
    fake_trace = smooth(fake_trace,box_smooth)
    g2 = corr(fake_trace,fake_trace, num_avg=100, max_tau=25,offset=100, block=30,verbose=True)
    ax1.plot(fake_trace-1)
    ax2.plot(g2,'k--')  
    hist_ax.plot([np.mean(g2), np.mean(g2)],[0,100], 'k--')


    ## 4d: let's see what the FFT of those good groups is
    threshold = 1.0
    good_indices = np.where( avg_g2 > threshold )[0]
    fig, ax3 = plt.subplots()
    fig, ax4 = plt.subplots()
    fft_cutoff = 20
    for i, index in enumerate(good_indices):
        trace = avg_over_runs[index, :]
        ax3.plot(trace+i)
        fft = np.abs( np.fft.fft(trace))[fft_cutoff:-fft_cutoff]
        ax4.plot( fft + i*10)
    ax3.set_title("High Auto-correlation Neurons")
    ax4.set_title("FFT of High Auto-correlation Neurons")
    ax3.set_xlabel("Time")
    ax4.set_xlabel("Fourier Mode")
    ax3.set_ylabel("Neuron Signal [offset]")
    ax4.set_ylabel("Fourier Amplitude [offset]")

    ax3.plot(fake_trace-1,'k')
    ax4.plot(np.abs( np.fft.fft(fake_trace)[fft_cutoff:-fft_cutoff])-10 , 'k')
        
    plt.show() 
##END question4

def question4a():
    ## make fake FFT data
    if False:
        ts = np.linspace(0,1,2**10)
        f = 6
        xs = 0.5+0.5*scipy.signal.square(2*np.pi*6*ts,duty=0.1) + 0.1*np.random.randn(len(ts))
        ys = 0.01*np.random.randn(2**10)  

        plt.plot(xs)
        plt.plot(ys)
        plt.xlabel("Time")
        plt.ylabel("Fake Neuron Signals [offset]")
        plt.figure()
        plt.plot( np.abs(np.fft.fft(xs))[10:-10] )
        plt.plot( np.abs(np.fft.fft(ys))[10:-10] )
        plt.xlabel("Fourier Mode")
        plt.ylabel("Fourier Amplitude [offset]")

    ## fake data for correlation
    if True:
        plt.figure()
        #xs,ys = np.random.randn(2,100)
        xs,ys = np.zeros((2,100))
        xs[10:20] = 1
        ys[25:40] = 1
        plt.plot(xs)
        plt.plot(ys+1)
        plt.xlabel("Time")
        plt.ylabel("Smoothed Neuron Signal [offset]")

        plt.figure()
        g2 = corr(ys,xs,num_avg=20,max_tau=40,block=1,verbose=False)
        plt.plot(g2)
        plt.xlabel("Delay Time")
        plt.ylabel("Cross Correlation")
       
    plt.show() 

##END quiestion4a

 
def main():
    read_dict = io.loadmat("bint_fishmovie32_100.mat")
    data = read_dict['bint']
    # data is 297 x 160 x 953
    # corresponding to 297 iterations of 160 neurons for 953 time steps

    ## action 0: play with data
    #show_allTraces(data, True)
    #cf_rank(data)    
    #regional_fft(data)

    ## action 1: answer questions (see writeup_questions.txt)
    question1(data)
    #question4(data)
    #question4a()
##END main
    

if __name__ == '__main__':
    foo = main()
