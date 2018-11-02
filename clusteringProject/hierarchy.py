#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 23:50:14 2018

@author: jmonroe

This script exists to 
"""
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage  
np.random.seed(2109)

def get_easy_data():
    ## generate data
    xs = 0.2*np.random.randn(5,2) + [1,3]
    xs = np.append(xs[:,0], xs[:,1])
    xs[-1] += 0.2
    ys = 0.2*np.random.randn(5,2) + [2,3]
    ys = np.append(ys[:,0], ys[:,1])
    plt.scatter(xs,ys)
    
    ## show it
    ## make plot with annotation
    labels = range(1,len(xs)+1)
    for label, x, y in zip(labels, xs,ys):  
        plt.annotate(
            label,
            xy=(x, y), xytext=(-3, 3),
            textcoords='offset points', ha='right', va='bottom')
        
    return xs, ys
##END get_easy_data


def get_fun_data():
    
    ## circular arrangement of [n_gauss] Gaussians [rad] away
    n_outer = 5
    n_iner = 4
    rad1, rad2 = 3, 0.35
    n_pointsPer = 50
    sig = 0.05
    
    xs = np.array([])
    ys = np.array([])
    
    for theta1 in np.linspace(0,2*np.pi,n_outer):
        outer_x = rad1*np.cos(theta1)
        outer_y = rad1*np.sin(theta1)
        for theta2 in np.linspace(0,2*np.pi,n_iner):
            mu_x = rad2*np.cos(theta2)  + outer_x
            mu_y = rad2*np.sin(theta2) + outer_y
            new_xs = sig* np.random.randn(n_pointsPer) + mu_x 
            new_ys = sig* np.random.randn(n_pointsPer) + mu_y 
            xs = np.append(xs,new_xs)
            ys = np.append(ys,new_ys)
    plt.scatter(xs,ys,color='k',s=2)
    return xs, ys
##END get_fun_data
    
    
def main():
    xs,ys = get_fun_data()
    return 0;
    
    ##use scipy dendrogram:
    labels = range(1,len(xs)+1)
    data = [(xs[i],ys[i]) for i in range(len(xs))]
    linked = linkage(data, 'single')
    plt.figure()
    dendrogram(linked,  
                orientation='top',
                no_labels=True,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()
##END main

if __name__ == '__main__':
    main()
