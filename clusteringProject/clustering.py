# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:41:13 2018

@author: J. Monroe
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2047)

def get_easy_data(show=False):
    '''
    Creates two well-separated 2-d clusters 
    '''
    mu1 = [0.2,2.5]
    mu2 = [2,2]
    mu3 = [1,1]
    sig = 0.15
    numPoints = 200 ## magic at 1,2,12, and 16 (checked to 20); 2016 seed
    cluster1 = sig*np.random.randn(numPoints ,2) + mu1
    cluster2 = sig*np.random.randn(numPoints ,2) + mu2
    cluster3 = sig*np.random.randn(numPoints ,2) + mu3
    
    if show:
        plt.scatter(cluster1[:,0], cluster1[:,1], color='cyan')
        plt.scatter(cluster2[:,0], cluster2[:,1], color='orange')
        plt.scatter(cluster3[:,0], cluster3[:,1], color='purple')
        plt.show()
        
    return cluster1, cluster2, cluster3
##END get_easy_data
    
    
def get_hard_data(show=False):
    '''
    Creates two well-separated 2-d clusters 
    '''
    mu1 = [0.2,2.5]
    mu2 = [2,2]
    mu3 = [1,1]
    sig1,sig2, sig3 = 0.15, 0.15, 0.4
    numPoints = 80 ## magic at 1,2,12, and 16 (checked to 20)
    cluster1 = sig1*np.random.randn(numPoints,2) + mu1
    cluster2 = sig2*np.random.randn(numPoints,2) + mu2
    cluster3 = sig3*np.random.randn(5*numPoints,2) + mu3
    
    
    if show:
        plt.scatter(cluster1[:,0], cluster1[:,1], color='cyan')
        plt.scatter(cluster2[:,0], cluster2[:,1], color='orange')
        plt.scatter(cluster3[:,0], cluster3[:,1], color='purple')
        plt.show()
        
    return cluster1, cluster2, cluster3
##END get_hard_data    

def calc_distance(xy_tuple,centers_list):
    ## make a separate function for extensibility
    dx = abs(centers_list[:, 0] - xy_tuple[0])
    dy = abs(centers_list[:, 1] - xy_tuple[1])    
    return np.sqrt( dx**2 + dy**2 )
    return dx+dy    
##END calc_distance
    

def k_means(data, k):
    '''
    data:   (n, dim) array 
    k:      number of clusters
    '''
    k = min(k,4) ## this is how many colors I've chosen...
    data = np.array(data)
    if len(data.shape)>1:
        dim = data.shape[1]
    else:
        dim = 1
    numPoints = data.shape[0]
    color_list = ["cyan","orange","purple","green"]
    
    num_iter = 4
    centers = np.zeros((k,dim))
    cluster_counts = np.zeros(k)
    cluster_ids_fullList = np.zeros((num_iter+1, numPoints) ,dtype="int")    
    distance_fullList = np.zeros(num_iter+1)
    
    cluster_indices = np.random.randint(0,k,size=numPoints)
    cluster_ids_fullList[0] = cluster_indices

    ## Initial calculations
    # centers
    for j,index in enumerate(cluster_indices):
        centers[index] += data[j]
        cluster_counts[index] += 1
    for k_index in range(k):
        centers[k_index] /= cluster_counts[k_index]
    # figure
    fig = plt.figure()
    plt.title("Initial Assignment")
    tot_dist = 0
    for i,(x,y) in enumerate(data):
        plt.scatter(x,y,color=color_list[cluster_indices[i]])
        tot_dist += min(calc_distance((0,0), centers))
    plt.scatter(centers[:, 0], centers[:, 1], marker='x',s=20,color='k')
    distance_fullList[0] = tot_dist
    
    ## k-means assignment
    for i in range(1,num_iter+1):
        ## reassign each point to nearest cluster 
        tot_distance = 0        
        #print(i, centers[0], centers[1])
        for j,(x,y) in enumerate(data):
            distances = calc_distance((x,y), centers)
            new_cluster_index = np.argmin(distances)
            cluster_indices[j] = new_cluster_index
            tot_distance += min(distances)
        ##END data loop
            
        ## define clusters
        cluster_list = [ [] for j in range(k)]
        for j,index in enumerate(cluster_indices):
            cluster_list[index].append(data[j])
        for j in range(k):
            if len(cluster_list[j]): centers[j] = np.mean(cluster_list[j],axis=0)
        #print str(i)+ "\t", centers
        #print cluster_list[1]
        ## track progress
        distance_fullList[i] = tot_distance
        cluster_ids_fullList[i] = cluster_indices
        plt.show()
    ##END iterations
    
    ## iteration-wise plots
    for i in range(1,num_iter+1):
        plt.figure()
        plt.title(str(i)+"th iteration")
        for j,(x,y) in enumerate(data):
            plt.scatter(x,y,color=color_list[cluster_ids_fullList[i][j]])
        plt.scatter(centers[:,0], centers[:,1], marker='x',s=20,color='k')
    
    plt.show()
    return cluster_ids_fullList, distance_fullList;
##END k_means


def main():
    ## get data
    c1, c2, c3 = get_easy_data(True)
    #c1,c2,c3 = get_hard_data(True)
    
    ## shuffle data
    xs = list(c1[:, 0]) + list(c2[:,0]) + list(c3[:,0])
    ys = list(c1[:, 1]) + list(c2[:,1]) + list(c3[:,1])
    zipped = [(xs[i],ys[i]) for i in range(len(xs))]
    
    ## find clusters
    return  k_means(zipped, k=3)
##END main()


if __name__ == '__main__':
    foo = main()
    
'''
    mu1 = [3,9]
    mu2 = [6,4]
    mu3 = [2,2]
    sig = 0.4
    numPoints = 200 ## magic at 1,2,12, and 16 (checked to 20); 2016 seed
'''