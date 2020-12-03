"""
LexRank re-implementation
Based on: https://github.com/crabcamp/lexrank/tree/dev
"""

import numpy as np
from scipy.sparse.csgraph import connected_components

def degree_centrality_scores(mat):
    rows_sum = mat.sum(axis=1, keepdims=True)
    markov = mat / rows_sum
    return stationary_dist(markov)

def stationary_dist(mat):
    dist = np.zeros(mat.shape[0])
    temp, lab = connected_components(mat)
    groups = []
    for i in np.unique(lab):
        groups.append(np.where(lab == i)[0])
    
    for group in groups:
        transition = mat[np.ix_(group, group)]
        eigenvector = power_method(transition)
        dist[group] = eigenvector
    return dist

def power_method(mat):
    eigen = np.ones(len(mat))
    if len(eigen) == 1:
        return eigen
    t = mat.T
    c=0
    while True:
        next_eigen = np.dot(t, eigen)
        if np.allclose(next_eigen, eigen) or c > 10:
            return next_eigen
        eigen = next_eigen
        t = np.dot(t, t)
        c+=1
