"""Kernel k-means clustering"""

# The MIT License (MIT)
# Copyright (c) 2019 Guillaume Rongier
#
# Author: Guillaume Rongier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


import numpy as np
from numba import jit, prange

from sklearn.utils import check_random_state


################################################################################
# Kernels


@jit(nopython=True, nogil=True, parallel=True)
def compute_kernel_matrix(samples):
    '''
    Computes the Gaussian kernel of the matrix of pairwise Euclidean distance
    between the samples
    '''
    kernel_matrix = np.zeros((samples.shape[0], samples.shape[0]))
    for j in prange(samples.shape[0]):
        for i in range(j + 1, samples.shape[0]):
            for v in range(samples.shape[1]):
                kernel_matrix[j, i] += (samples[j, v] - samples[i, v])**2
            kernel_matrix[i, j] = kernel_matrix[j, i]

    sigma = 0.2*np.max(kernel_matrix)

    return np.exp(-kernel_matrix/(2*sigma**2))

################################################################################
# Kernel k-means


class KernelKMeans:
    '''
    Dual k-means clustering specified by a Gaussian kernel
    
    Parameters
    ----------
        
    n_clusters : int (default 2)
        Number of clusters to form.
        
    max_iter : int (default 100)
        Maximum number of iterations of the k-means algorithm.
        
    verbose : bool (default True)
        If True, print information about the successive steps, if False, nothing
        is printed.
        
    random_state : int or RandomState instance, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        'np.random'.
        
    Attributes
    ----------
    
    labels_ : array, shape (n_samples)
        Store the label of the cluster associated to each sample.
    
    References
    ----------
    
    Based on the following MATLAB implementation:
    https://github.com/xtan1/comparingGSalgorithms/blob/f02c20f3f163e87b35ecdb90d3f42e4a7e7fa10b/dualkmeansFast.m
    
    '''
    def __init__(self,
                 n_clusters=2,
                 max_iter=100,
                 verbose=False,
                 random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X):
        
        random_state = check_random_state(self.random_state)
        
        kernel_matrix = compute_kernel_matrix(X)

        n = kernel_matrix.shape[0]
        cluster_matrix = np.zeros((n, self.n_clusters))
        self.labels_ = random_state.randint(0, self.n_clusters, n)
        cluster_matrix[np.arange(n), self.labels_] = 1

        change = True
        it = 0
        while change == True and it < self.max_iter:
            if self.verbose:
                print('Iteration ', it + 1, '/', self.max_iter, sep='')
                
            change = False

            nb_cluster_samples = np.sum(cluster_matrix, axis=0)
            # This is required to reproduce MATLAB's output, otherwise everything 
            # becomes NaNs and one cluster gets all the samples
            nb_cluster_samples[nb_cluster_samples != 0.] = 1./nb_cluster_samples[nb_cluster_samples != 0.]
            E = cluster_matrix@np.diag(nb_cluster_samples)
            cluster_distances = np.ones((n, 1))@np.diag(E.T@kernel_matrix@E).T[np.newaxis] - 2*kernel_matrix@E
            new_labels = np.argmin(cluster_distances, axis=1)
#             distances = cluster_distances[np.arange(cluster_distances.shape[0]), new_labels]

            for i in range(n):
                if self.labels_[i] != new_labels[i]:
                    cluster_matrix[i, new_labels[i]] = 1
                    cluster_matrix[i, self.labels_[i]] = 0
                    change = True

            self.labels_ = new_labels
            it += 1

        return self
