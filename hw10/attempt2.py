#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 22:31:12 2018

@author: jonathan

This script exists to 
"""
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2018)

def make_data(show=False):
    ## create a dataset where the first half is big, and the second half is small
    data = np.zeros((6,10))
    sig = 0.1
    big = 2
    small= 0.05
    for set_num in range(2):
        for rep_num in range(3):
            if set_num: 
                first = small; second = big
            else:
                first = big; second = small
            first_half  =  first + sig*np.random.randn(5)
            second_half = second + sig*np.random.randn(5)
            abundance = np.append(first_half,second_half)
                            
            data[set_num*3 + rep_num] = abundance
    if show:
        plt.figure()
        plt.imshow(data)
        plt.xlabel("Species number")
        plt.ylabel("Replicate number")
        plt.title("Abundance of species")
        plt.show()
    return data
##END make_data


def main():
    data = make_data()
    
    ## coarse_graining version 1
    ## average first 5 population in each
    plt.figure()
    for abundance in data:
        x = np.mean(abundance[:5])
        y = np.mean(abundance[5:])
        plt.plot(x,y, 'ok')
        
    ## coarse_graining version 2
    ## distance from the first replicate
    plt.figure()
    base = data[0]
    for i,abundance in enumerate(data):
        dist = sum(np.sqrt((base-abundance)**2))
        plt.plot(i,dist, 'ok')
        
    ## bad coarse_graining
    ## using average expression
    avg = data.mean(axis=1)
    plt.figure()
    plt.plot(range(len(avg)), avg, 'or')
    plt.xlabel("Replicate index")
    plt.ylabel("Bad classifier")
    
    
##END main
    
if __name__ == '__main__':
    main()

