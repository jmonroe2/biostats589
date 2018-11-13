# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 22:59:35 2018

@author: jonathan
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

def main():
    loaded_mat= scipy.io.loadmat("bint_fishmovie32_100.mat")

    data = loaded_mat[loaded_mat.keys()[0]]
    print(type(data))        
    print data.shape
    
    avg = np.mean(data,axis=0)
    print avg.shape
    #plt.imshow(avg)
    
    ft = np.fft.fft(avg, axis=1)
    plt.imshow(np.abs(ft[:,300:-300]))
##END main
    
if __name__ == '__main__':
    main()