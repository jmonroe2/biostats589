## SO answer: https://stackoverflow.com/questions/46258657/
## see also here: http://ptrckprry.com/course/ssd/lecture/community.html

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import matplotlib.pyplot as plt
np.random.seed(1)

# Get your mentioned graph
G = nx.karate_club_graph()

# Get ground-truth: club-labels -> transform to 0/1 np-array
#     (possible overcomplicated networkx usage here)
gt_dict = nx.get_node_attributes(G, 'club')
gt = [gt_dict[i] for i in G.nodes()]
gt = np.array([0 if i == 'Mr. Hi' else 1 for i in gt])

# Get adjacency-matrix as numpy-array
adj_mat = nx.to_numpy_matrix(G)
my_mat = np.array(adj_mat)

#plt.imshow(my_mat,origin='lower')
#plt.show()

print('ground truth')
print(gt)

# Cluster
sc = SpectralClustering(2, affinity='precomputed', n_init=100,assign_labels='discretize')
sc.fit(adj_mat)

# Compare ground-truth and clustering-results
print('spectral clustering')
print(sc.labels_)

print('just for better-visualization: invert clusters (permutation)')
print(np.abs(sc.labels_ - 1))

# Calculate some clustering metrics
print(metrics.adjusted_rand_score(gt, sc.labels_))
print(metrics.adjusted_mutual_info_score(gt, sc.labels_))
#'''



################## EXAMPLE 2 #####################
if True:
    print("\n\nEXAMPLE 2")
    np.random.seed(0)
    
    adj_mat = [[3,2,2,0,0,0,0,0,100],
               [2,3,2,0,0,0,0,0,0],
               [2,2,3,1,0,0,0,0,0],
               [0,0,1,3,3,3,0,0,0],
               [0,0,0,3,3,3,0,0,0],
               [0,0,0,3,3,3,1,0,0],
               [0,0,0,0,0,1,3,1,1],
               [0,0,0,0,0,0,1,3,1],
               [100,0,0,0,0,0,1,1,3]]
    
    adj_mat = np.array(adj_mat)
    
    sc = SpectralClustering(3, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    
    print('spectral clustering')
    print(sc.labels_)