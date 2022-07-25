"""Kernel k-means clustering"""

# The MIT License (MIT)
# Copyright (c) 2019-2022 Guillaume Rongier
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
import numba as nb
from sklearn.utils import check_random_state


################################################################################
# Utils

@nb.jit(nopython=True, nogil=True, parallel=True)
def square_euclidean_distance(samples_1, samples_2):
    """
    Computes the squared Euclidean distance between two sets of samples.
    """
    distance_matrix = np.zeros((samples_1.shape[0], samples_2.shape[0]))
    for j in nb.prange(samples_1.shape[0]):
        for i in range(j + 1, samples_2.shape[0]):
            for v in range(samples_1.shape[1]):
                distance_matrix[j, i] += (samples_1[j, v] - samples_2[i, v])**2
            if j < samples_2.shape[0] and i < samples_1.shape[0]:
                distance_matrix[i, j] = distance_matrix[j, i]

    return distance_matrix


################################################################################
# Kernels

class GaussianKernel:
    """
    Computes the Gaussian kernel of the matrix of pairwise Euclidean distance
    between the samples.

    Parameters
    ----------
    gamma : float, optional
        Coefficient of the Gaussian kernel.
    """
    def __init__(self, gamma=None):
        
        self.gamma = gamma
        
    def __call__(self, samples):

        kernel_matrix = square_euclidean_distance(samples, samples)

        gamma = self.gamma
        if gamma is None:
            sigma = 0.2*np.max(kernel_matrix)
            gamma = 1/(2.*sigma**2)

        return np.exp(-gamma*kernel_matrix)


################################################################################
# Kernel k-means

class KernelKMeans:
    """
    Kernel k-means clustering.
    
    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to form.
    init : {'random', 'k-means++'}, optional
        Method to initialize the labels, either 'random' or 'k-means++'.
        k-means++ is a method to improve convergence. It is not the default
        option to stay closer to the original ANODI implementation.
    n_init : int, optional
        Number of times the clustering is applied with different seeds, the
        final result being the one with the lowest inertia.
    max_iter : int, optional
        Maximum number of iterations of the k-means algorithm.
    kernel : kernel object or function, ndarray, or None, optional
        Kernel to compute the kernel matrix from the data. If a kernel object,
        it must have a __call__ function with a single argument (the data) that
        returns a ndarray (the kernel matrix). A kernel function must fit the
        same input and output. An ndarray must be a kernel matrix. None means
        that a normal k-means is used.
    verbose : bool, optional
        If True, print information about the successive steps, if False, nothing
        is printed.
    random_state : int or RandomState instance, optional
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
    Shawe-Taylor, J. & Cristianini, N. (2011).
        Kernel Methods for Pattern Analysis.
        Cambridge University Press, https://doi.org/10.1017/CBO9780511809682
    Arthur, D. & Vassilvitskii, S. (2007).
        k-means++: the advantages of careful seeding.
        Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms, 1027–1035
    """
    def __init__(self,
                 n_clusters=2,
                 init='random',
                 n_init=10,
                 max_iter=300,
                 kernel=GaussianKernel(),
                 verbose=False,
                 random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.kernel = kernel
        self.verbose = verbose
        self.random_state = random_state
        
    def _initialize(self, matrix, random_state):
        """
        Initializes the labels at random or using the k-means++ method
        """
        if random_state is None:
            random_state = check_random_state(self.random_state)
        else:
            random_state = check_random_state(random_state)
        
        n = matrix.shape[0]
        if self.init == 'k-means++':
            centroids = [random_state.randint(n)]
            while len(centroids) < self.n_clusters:
                centroid_distances = matrix[centroids]
                if self.kernel is None:
                    centroid_distances = square_euclidean_distance(centroid_distances,
                                                                   matrix)
                distances = np.min(centroid_distances, axis=0)**2
                distances /= np.sum(distances)
                centroid = random_state.choice(n, p=distances)
                centroids.append(centroid)
            centroid_distances = matrix[centroids]
            if self.kernel is None:
                centroid_distances = square_euclidean_distance(centroid_distances,
                                                               matrix)
            return np.argmin(centroid_distances, axis=0)
        else:
            return random_state.randint(self.n_clusters, size=n)
    
    def _fit_single(self, kernel_matrix, init_labels):
        """
        Computes the kernel k-means clustering for a single initialization
        """
        labels = init_labels
        n = kernel_matrix.shape[0]
        cluster_matrix = np.zeros((n, self.n_clusters))
        cluster_matrix[np.arange(n), labels] = 1

        change = True
        it = 0
        while change == True and it < self.max_iter:
            if self.verbose:
                print('Iteration ', it + 1, '/', self.max_iter, sep='')
                
            change = False

            nb_cluster_samples = np.sum(cluster_matrix, axis=0)
            nb_cluster_samples[nb_cluster_samples != 0.] = 1./nb_cluster_samples[nb_cluster_samples != 0.]
            E = cluster_matrix@np.diag(nb_cluster_samples)
            KE = kernel_matrix@E
            cluster_distances = -2*KE + np.diag(E.T@KE)
            new_labels = np.argmin(cluster_distances, axis=1)

            for i in range(n):
                if labels[i] != new_labels[i]:
                    cluster_matrix[i, new_labels[i]] = 1
                    cluster_matrix[i, labels[i]] = 0
                    change = True

            labels = new_labels
            it += 1
            
        distances = cluster_distances[np.arange(cluster_distances.shape[0]),
                                      labels]
        inertia = np.sum(distances)

        return labels, inertia, it
    
    def fit(self, X):
        """
        Computes the kernel k-means clustering
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training instances to cluster.
        """
        random_state = check_random_state(self.random_state)
        
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
        if self.kernel is not None:
            if isinstance(self.kernel, np.ndarray):
                kernel_matrix = self.kernel
            else:
                kernel_matrix = self.kernel(X)
        else:
            kernel_matrix = X@X.T
            
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self.n_init)
        for seed in seeds:
            if self.kernel is not None:
                labels = self._initialize(kernel_matrix, seed)
            else:
                labels = self._initialize(X, seed)
            labels, inertia, it = self._fit_single(kernel_matrix, labels)
            
            if self.inertia_ is None or inertia < self.inertia_:
                self.labels_ = labels.copy()
                self.inertia_ = inertia
                self.n_iter_ = it
            
        return self
