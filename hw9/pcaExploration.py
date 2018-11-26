#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:00:56 2018

@author: jmonroe

This script exists to help understand the capabilities of PCA via completing 
biostats 589 homework 9
"""

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2018)
from mpl_toolkits.mplot3d import Axes3D

def main():
    show_plots = True
    
    ### part a: generate feature vectors
    num_dim = 100
    num_cells = 500
    
    v1,v2,v3 = np.random.randn(3,num_dim)
    norm = np.sqrt(np.dot(v1,v1))
    v1 /= norm
    norm = np.sqrt(np.dot(v2,v2))
    v2 /= norm
    norm = np.sqrt(np.dot(v3,v3))
    v3 /= norm

    
    ### part b: check out their scalar products:
    if show_plots:
        print("\nPart B: Scalar products:")
        print("v_1 . v_2  = {0}".format(np.dot(v1,v2)))
        print("v_1 . v_3  = {0}".format(np.dot(v1,v3)))    
        print("v_2 . v_3  = {0}".format(np.dot(v3,v2)))  
    
    
    ### part c: create a dataset using 3 secret variables (zero-mean, given variance)
    a_vect = np.sqrt(20)*np.random.randn(num_cells) + 0
    b_vect = np.sqrt(5)*np.random.randn(num_cells) + 0
    c_vect = np.sqrt(0.5)*np.random.randn(num_cells) + 0
    
    data = np.zeros((num_cells,num_dim))
    for i in range(num_cells):
        x_i = 0 + a_vect[i]*v1 + b_vect[i]*v2 +c_vect[i]*v3
        data[i] = x_i
    if num_dim ==2 :
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.title("Part C: Data in 3D (before noise)")
        ax.scatter(data[:,0], data[:,1], alpha=0.3, s=1)
    
    ### part d: add zero-mean noise
    sig = 1
    noise = sig*np.random.randn(num_cells,num_dim) + 0
    data += noise
    
    ### part e: do we see a correlation b/t the first and second features?
    feat1 = data[:, 0]
    feat2 = data[:, 1]
    if show_plots:
        plt.figure()
        plt.title(f"Part E: First vs second vectors \nwidth should match $\sigma$={sig}")
        plt.scatter(feat1,feat2,s=1)

    ### part f: math
    '''
    For a vector x, projected onto a unit vector u, the length of 
    the projection of x onto u is
    x'  =  u |x|cos(theta) = u x.u
    The length of this vector is simply |x.u| which lies somewhere between 0 and |x|
    '''
    
    ### part g: project onto new vectors
    # try to choose a random direction (ie combo of features) 
    #   and see if this vector is representative of the data
    #   ie does each of num_cells vectors 'match' with this?
    unit1, unit2 = np.random.randn(2,num_dim)
    unit1 /= np.sqrt( np.dot(unit1,unit1) )
    unit2 /= np.sqrt( np.dot(unit2,unit2) )
    
    proj1 = np.dot(data, unit1.T)
    proj2 = np.dot(data, unit2.T)
    if show_plots:
        plt.figure()
        plt.title("Part G: Random projections")
        plt.scatter(proj1, proj2,s=1)
        print("G: std dev, {0}, {1}".format(np.var(proj1), np.var(proj2)))
   
 
    ### part h: calculate eigenvectors of covariance matrix
    cov = np.dot(data.T,data) /num_cells
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # we need to find the highest 3 vectors. 
    sort_indices = eigenvalues.argsort()[::-1] ## max should be first
    ev = eigenvalues
    e_vect_1 = eigenvectors[:, sort_indices[0]]
    e_vect_2 = eigenvectors[:, sort_indices[1]]
    e_vect_3 = eigenvectors[:, sort_indices[2]]    
    
    
    ### part i: projection onto primary vectors
    # note, eigenvectors are already normalized
    proj1 = np.dot(data, e_vect_1)
    #proj1 = np.dot(data, e_vect_1)
    proj2 = np.dot(data, e_vect_2)
    proj3 = np.dot(data, e_vect_3)
    var1 = np.var(proj1)
    var2 = np.var(proj2)    
    var3 = np.var(proj3)    
    print(f"\nPart I: var of (1,2,3) projection is {var1,var2,var3}")
    if show_plots:
        plt.figure()
        plt.title("Part I: Eigenvector projection")
        plt.scatter(proj1,proj2)
    plt.show(); return 0; show()
    
    
    ### part j: check the output
    print("\nPart J: Variable correspondence:")
    print("v1.eigen1={0}".format(np.dot(e_vect_1,v1)))
    print("v1.eigen2={0}".format(np.dot(e_vect_2,v1)))
    print("v1.eigen3={0}".format(np.dot(e_vect_3,v1)))    
    
    '''
    If these scalar products are +1 then our data vectors are exactly parallel
    to the "template vectors" implying accurate prediction by the templates.
    -1 indicates anti-alignment which is a trivial consequence of 
    
    While the first two components have been recovered, the third seems to have been lost.
    This is due to the fact that the variance, $\lambda_c$ is less than the noise
    variance, $\sigma=1$. Ie it is below our "noise floor" as we show next.
    '''
    
    ### part k: compare to Marchenko-Pastur distribution from random matrix theory
    #r = num_cells/num_dim
    def p(r):
        lam_plus = sig**2 *(1+np.sqrt(r))**2
        lam_minus = sig**2 *(1-np.sqrt(r))**2
        x = np.linspace(lam_minus,lam_plus,100)
        prob = 1.0/(2*np.pi*sig) *np.sqrt((lam_plus-x)*(x-lam_minus))/r/x    
        return x, prob
    
    xs_first, ps_first = p(num_cells/num_dim)
    xs_scaled, ps_scaled = p(2*num_cells/num_dim)
    if False:
        plt.figure()
        plt.title("\nPart K: Eigenvalue distribution")
        plt.plot(xs_first,ps_first, 'k-')
        plt.plot(xs_scaled,ps_scaled, 'r--')
    
    # get eigenspectrum
    plt.hist(eigenvalues,bins=30,color='b',density=True)
    
    ##END 
    
    plt.show()
    
##END main
    
if __name__ == '__main__':
    main()
