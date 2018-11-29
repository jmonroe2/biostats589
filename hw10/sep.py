#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:26:40 2018

@author: jonathan

This script exists to 
"""
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)

#def main():
if True:
    num_replicates = 3
    measures_per_replicate = 1000
    species_prob = [1, 2, 8, 15, 30, 37, 52, 100, 150, 152 ] ## relative weights
    species_prob = np.array(species_prob[1:])/sum(species_prob)
    species_prob[-1] = 1.
    #species_prob = np.linspace(0.1,1,10)
    print(species_prob)
    num_species = len(species_prob) # == 10
    mu_a = -5
    mu_b = 5
    sig = 1
    second_mu = np.arange(-num_species//2, num_species//2) ## fractional splitting for species
    
    
     ## each replicate should sample which species to extract, then add then about their respective means. 
    rands = np.random.random(measures_per_replicate)
    data = np.zeros(measures_per_replicate)
    tmp = []
    for j,rand in enumerate(rands):
        ## get first index which has a higher probability than the random value
        index = np.where(species_prob>rand )[0][0]
        gp_id = index % 2
        mu = [mu_a, mu_b][gp_id]
        mu_mini = second_mu[index % (num_species//2)]
  
        data[j] = sig*np.random.randn(1) + mu
        data_y[j] = sig*np.random.randn(1)+ mu_mini
        tmp.append(index)
    ##END data loop
    plt.figure()
    plt.hist(data,bins=30)
    plt.title("generated data")
    
    plt.figure()
    plt.hist(tmp)
    plt.title("Indices")
    plt.show()
##END main
    
def main():
    pass
    
if __name__ == '__main__':
    main()

