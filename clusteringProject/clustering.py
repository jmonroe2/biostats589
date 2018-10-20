# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:41:13 2018

@author: J. Monroe
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1337)

def get_easy_data(show=False):
    '''
    Creates two well-separated 2-d clusters 
    '''
    mu1 = [3,9]
    mu2 = [6,4]
    sig = 0.4
    numPoints = 30
    cluster1 = sig*np.random.randn(numPoints ,2) + mu1
    cluster2 = sig*np.random.randn(numPoints ,2) + mu2
    
    if show:
        plt.scatter(cluster1[:,0], cluster1[:,1], color='cyan')
        plt.scatter(cluster2[:,0], cluster2[:,1], color='orange')
        plt.show()
        
    return cluster1, cluster2
##END get_easy_data
    
    
def k_means(data, k):
    '''
    data:   (n, dim) array 
    k:      number of clusters
    '''
    #TODO: add dimensions
    data = np.array(data)
    if len(data.shape)>1:
        dim = data.shape[1]
    else:
        dim = 1
    numPoints = data.shape[0]
    #initial_indices = np.random.randint(0,k,size=numPoints)
    initial_indices= np.zeros(numPoints,dtype='int')
    initial_indices[numPoints//2:] += 1
    
    
    centers = np.zeros(k)
    for i,index in enumerate(initial_indices):
        centers[index] += data[i] ## does this work element-wise for a d-dim vector?
    centers /= numPoints
    print(centers)
    
    plt.plot(data)
    #plt.hist(data,bins=40)
    plt.show()
    return 0;
    
    num_iter = 4
    for i in range(num_iter):
        ## define clusters
        for x in data:
            ## calculate distance to each cluster
            pass
        ## 
    return 0
##END k_means


def main():
    ## get data
    c1, c2 = get_easy_data()
    
    ## shuffle data
    xs = list(c1[:, 0]) + list(c2[:,0])
    ys = list(c1[:, 1]) + list(c2[:,1])
    
    ## find clusters
    clusters = k_means(xs, k=2)
##END main()


if __name__ == '__main__':
    main()