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
    numPoints = 300
    cluster1 = sig*np.random.randn(numPoints ,2) + mu1
    cluster2 = sig*np.random.randn(numPoints ,2) + mu2
    
    if show:
        plt.scatter(cluster1[:,0], cluster1[:,1], color='cyan')
        plt.scatter(cluster2[:,0], cluster2[:,1], color='orange')
        plt.show()
        
    return cluster1, cluster2
##END get_easy_data
    

def calc_distance(x,y):
    ## make a separate function for extensibility
    return np.sqrt(x**2 + y**2)
##END calc_distance
    
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
    
    
    num_iter = 4
    centers = np.zeros(k)
    cluster_ids_perIter = np.zeros((num_iter, numPoints) )
    
    cluster_indices = np.random.randint(0,k,size=numPoints)
    cluster_ids_perIter[0] = cluster_indices
    for i in range(num_iter):
        ## define clusters
        for j,index in enumerate(cluster_indices):
            centers[index] += data[j]
        centers /= numPoints/k  # numPoints includes all clusters

        tot_distance = 0
        ## reassign each point to nearest cluster 
        for j,x in enumerate(data):
            distances = calc_distance(x,centers)
            new_cluster_index = np.argmin(distances)
            cluster_indices[j] = new_cluster_index
            tot_distance += min(distances)
        ## track progress
        cluster_ids_perIter[i] = cluster_indices
        print(tot_distance)
    ##END iterations
        
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