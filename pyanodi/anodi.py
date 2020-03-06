"""Analysis of distance (ANODI)"""

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
from numba import jit, prange, vectorize, float64

import skimage as ski
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial import distance as sdistance

from .kernel_k_means import *


################################################################################
# Utils


@jit(nopython=True)
def get_patterns(array, halfwindow, step=(1, 1)):
    '''
    Gets the patterns from the training image
    '''
    patterns = np.empty((int(np.ceil((array.shape[0] - 2*halfwindow[0])/step[0])
                             *np.ceil((array.shape[1] - 2*halfwindow[1])/step[1])),
                         2*halfwindow[0] + 1,
                         2*halfwindow[1] + 1))
    p = 0
    for j in range(halfwindow[0], array.shape[0] - halfwindow[0], step[0]):
        for i in range(halfwindow[1], array.shape[1] - halfwindow[1], step[1]):
            patterns[p] = array[j - halfwindow[0]:j + halfwindow[0] + 1,
                                i - halfwindow[1]:i + halfwindow[1] + 1]
            p += 1
    
    return patterns


@jit(nopython=True)
def compute_differential_entropy(samples, bins):
    '''
    Computes the differential entropy for a set of samples
    '''
    hist, bin_edges = np.histogram(samples, bins=bins)
    hist = hist/np.sum(hist)
    
    return -np.sum(hist[hist != 0]*np.log2(hist[hist != 0]/(bin_edges[1:][hist != 0]
                                                            - bin_edges[:-1][hist != 0])))


@jit(nopython=True, nogil=True, parallel=True)
def compute_mean_differential_entropy_patterns(array, halfwindows, step=(1, 1)):
    '''
    Computes the mean differential entropy of all the patterns from the training
    image for several pattern sizes
    '''
    mean_entropies = np.zeros(halfwindows.shape[0])
    for k in range(halfwindows.shape[0]):
        bins = np.max(halfwindows[k])
        for j in range(halfwindows[k, 0], array.shape[0] - halfwindows[k, 0], step[0]):
            for i in range(halfwindows[k, 1], array.shape[1] - halfwindows[k, 1], step[1]):
                pattern = array[j - halfwindows[k, 0]:j + halfwindows[k, 0] + 1,
                                i - halfwindows[k, 1]:i + halfwindows[k, 1] + 1]
                mean_entropies[k] += compute_differential_entropy(pattern.ravel(),
                                                                  bins)
        mean_entropies[k] /= (array.shape[0] - 2*halfwindows[k, 0])*(array.shape[1] - 2*halfwindows[k, 1])
        
    return mean_entropies


def compute_forward_second_difference(samples):
    '''
    Computes the forward second difference for a set of samples
    '''
    return samples[2:] - 2*samples[1:-1] + samples[:-2]


def compute_log_gaussian(samples, mean, var):
    '''
    Computes the log of a gaussian distribution for a set of samples
    '''
    return len(samples)*np.log(1/np.sqrt(2*np.pi*var))\
           - np.sum(((samples - mean)**2)/(2*var))


def compute_profile_log_likelihood(samples, common_scale=False):
    '''
    Computes the profile log-likelihood of a set of samples
    '''
    start = 2
    stop = len(samples) - 1
    if common_scale == True:
        start = 1
        stop = len(samples)
    
    log_likelihoods = np.full(len(samples), np.nan)
    for q in range(start, stop):

        mean_1 = np.mean(samples[:q])
        mean_2 = np.mean(samples[q:])
        var_1 = np.std(samples[:q])**2
        var_2 = np.std(samples[q:])**2
        if common_scale == True:
            sigma = ((q - 1)*var_1 + (len(samples) - q - 1)*var_2)/(len(samples) - 2)
            var_1 = sigma
            var_2 = sigma

        lq_1 = compute_log_gaussian(samples[:q], mean_1, var_1)
        lq_2 = compute_log_gaussian(samples[q:], mean_2, var_2)
        
        log_likelihoods[q - 1] = lq_1 + lq_2
        
    return log_likelihoods


def select_template_shape(array, max_halfwindow=None, step=1, return_all=False):
    '''
    Selects the template shape using the elbow plot so that the template size
    records the pattern variations from the training image
    '''
    if max_halfwindow is None:
        # TODO: Need to find a way to deal with rectangular halfwindows
        max_halfwindow = np.min(array.shape)
        max_halfwindow = (int(0.2*max_halfwindow), int(0.2*max_halfwindow))
    if isinstance(step, int):
        step = (step, step)

    halfwindows = np.array((np.arange(1, max_halfwindow[0] + 1),
                            np.arange(1, max_halfwindow[1] + 1))).T
    mean_entropies = compute_mean_differential_entropy_patterns(array,
                                                                halfwindows,
                                                                step=step)
    diff_entropies = compute_forward_second_difference(mean_entropies)
    log_likelihoods = compute_profile_log_likelihood(diff_entropies)
    best_halfwindow = halfwindows[np.nanargmax(log_likelihoods)]
        
    if return_all == False:
        return best_halfwindow
    return best_halfwindow, variabilities, diff_variabilities, log_likelihoods


def find_reducing_dimensions(samples,
                             return_reduced_samples=False,
                             random_state=None):
    '''
    Finds the number of components for dimensionality reduction using the elbow
    plot so that as much information as possible is preserved
    '''
    reducing = PCA(random_state=random_state)
    samples_reduced = reducing.fit_transform(samples)
    
    profile_likelihood = compute_profile_log_likelihood(reducing.explained_variance_ratio_)
    nb_components = np.nanargmax(profile_likelihood) + 1
    
    if return_reduced_samples == False:
        return nb_components
    return nb_components, samples_reduced[:, :nb_components]


@jit(nopython=True)
def find_medoid(samples):
    '''
    Finds the medoid pattern from a set of patterns
    '''
    distances = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        for j in range(samples.shape[0]):
            distance = 0.
            for v in range(samples.shape[1]):
                for u in range(samples.shape[2]):
                    distance += (samples[j, v, u] - samples[i, v, u])**2
            distances[i] += np.sqrt(distance)
        
    return np.argmin(distances)


@jit(nopython=True)
def compute_mean_pattern(patterns):
    '''
    Computes the mean pattern from a set of patterns
    '''
    mean_pattern = np.zeros(patterns.shape[1:])
    for k in range(patterns.shape[0]):
        for j in range(patterns.shape[1]):
            for i in range(patterns.shape[2]):
                mean_pattern[j, i] += patterns[k, j, i]
            
    return mean_pattern/patterns.shape[0]


@jit(nopython=True, nogil=True, parallel=True)
def compute_cluster_prototypes(patterns, clusters, method='mean'):
    '''
    Computes the prototype of the patterns from each cluster
    '''
    if method != 'mean' and method != 'medoid':
        raise ValueError('''Method must be 'mean' or 'medoid' ''')
    
    id_clusters = np.unique(clusters)
    prototypes = np.empty((id_clusters.shape[0],) + patterns.shape[1:])
    for i in prange(id_clusters.shape[0]):
        if method == 'mean':
            prototypes[i] = compute_mean_pattern(patterns[clusters == id_clusters[i]])
        elif method == 'medoid':
            id_medoid = find_medoid(patterns[clusters == id_clusters[i]])
            prototypes[i] = patterns[clusters == id_clusters[i]][id_medoid]
        
    return prototypes


@jit(nopython=True)
def assign_clusters(array, prototypes):
    '''
    Assigns the patterns of a realization to a cluster based on the Euclidean
    distance to the pattern prototypes computed from the training image
    '''
    halfwindow = (int(prototypes.shape[1]/2), int(prototypes.shape[2]/2))
    n = (array.shape[0] - 2*halfwindow[0])*(array.shape[1] - 2*halfwindow[1])
    if n == 0:
        raise ValueError("Template larger than resized realization")
    
    clusters = np.zeros(n)
    distances = np.full(n, np.inf)
    for j in range(halfwindow[0], array.shape[0] - halfwindow[0]):
        for i in range(halfwindow[1], array.shape[1] - halfwindow[1]):
            p = (j - halfwindow[0])*(array.shape[1] - 2*halfwindow[1]) + i - halfwindow[1]
            pattern = array[j - halfwindow[0]:j + halfwindow[0] + 1,
                            i - halfwindow[1]:i + halfwindow[1] + 1]
            for k in range(prototypes.shape[0]):
                distance = 0.
                for v in range(pattern.shape[0]):
                    for u in range(pattern.shape[1]):
                        distance += (pattern[v, u] - prototypes[k, v, u])**2
                distance = np.sqrt(distance)
                if distance < distances[p]:
                    clusters[p] = k
                    distances[p] = distance
        
    return clusters


@vectorize([float64(float64, float64)])
def rel_entr(x, y):
    '''
    Computes logarithm operations for the Jensen-Shannon distance
    '''
    if np.isnan(x) or np.isnan(y):
        return np.nan
    elif x > 0 and y > 0:
        return x*np.log(x / y)
    elif x == 0 and y >= 0:
        return 0
    else:
        return np.inf


@jit(nopython=True)
def jensen_shannon(p, q, base=None):
    '''
    Computes the Jensen-Shannon distance, based on SciPy function
    distance.jensenshannon, see:

    https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html
    '''
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/np.sum(p)
    q = q/np.sum(q)
    m = (p + q)/2.0
    left = rel_entr(p, m)
    right = rel_entr(q, m)
    js = np.sum(left) + np.sum(right)
    if base is not None:
        js /= np.log(base)

    return np.sqrt(js/2.0)


@jit(nopython=True, nogil=True, parallel=True)
def compute_distances(distances, distribution_ti, distributions_rez, g, nb_methods, nb_rez, verbose):
    '''
    Fills a matrix of distances based on the Jensen-Shannon distance between
    distributions
    '''
    if verbose:
        print('\nComputing distances\n... Within')
    for j in range(nb_methods*nb_rez):
        v = int(j/nb_rez)
        u = int(j%nb_rez)
        distances[j + 1, 0, g] = jensen_shannon(distributions_rez[v, u],
                                                distribution_ti)
        distances[0, j + 1, g] = distances[j + 1, 0, g]
    if verbose:
        print('... Between')
    for j in prange(nb_methods*nb_rez):
        v = int(j/nb_rez)
        u = int(j%nb_rez)
        for i in range(j + 1, nb_methods*nb_rez):
            z = int(i/nb_rez)
            w = int(i%nb_rez)
            distances[j + 1, i + 1, g] = jensen_shannon(distributions_rez[v, u],
                                                        distributions_rez[z, w])
            distances[i + 1, j + 1, g] = distances[j + 1, i + 1, g]

################################################################################
# ANODI


class ANODI:
    '''
    Analysis of distance (ANODI)
    
    Parameters
    ----------
    
    pyramid : array-like (default (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) )
        Scaling coefficients for multiscale analysis.
        
    halfwindow : int or array-like, optional (default None)
        Half-size of the window to extract the patterns from the training image
        and the realizations.
        
    max_halfwindow : int or array-like, optional (default None)
        Maximum possible half-size of the window. Only used when halfwindow is
        None, and set to 20% of the training image size by default.
        
    n_clusters : int (default 48)
        Number of clusters to group patterns and build the histograms of patterns.
        
    step : int or array-like (default 1)
        Step between two patterns in the training image, to limit the number of
        patterns during clustering and reduce the computational burden.
        
    use_mds : bool (default False)
        If False, the dimensionality reduction before clustering the patterns is
        done using a PCA, if True, it is done using the MDS method SMACOF. The
        original code uses the classical MDS method with the Euclidean distance
        to process continuous variables, which is equivalent to a PCA but far
        less efficient computationaly.
        
    method : str (default 'mean')
        Method to compute the pattern prototype for each cluster, either 'mean'
        or 'medoid'.
        
    verbose : bool (default True)
        If True, print information about the successive steps, if False, nothing
        is printed.
        
    random_state : int or RandomState instance, optional (default None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        'np.random'.
        
    n_jobs : None (default True)
        Number of jobs to use for the computation.
        
    Attributes
    ----------
    
    prototypes_ : array, shape (n_clusters, 2*halfwindow[0] + 1, 2*halfwindow[1] + 1)
        Store the pattern prototypes for each cluster.
    
    distances_ : array, shape (1 + n_realizations, 1 + n_realizations, n_pyramid_levels)
        Store the distances between realizations and training image and
        in-between realizations.
    
    References
    ----------
    
    Tan, X., Tahmasebi, P. & Caers, J. (2014).
    Comparing Training-Image Based Algorithms Using an Analysis of Distance.
    Mathematical Geosciences, 46(2), 149-169, https://doi.org/10.1007/s11004-013-9482-1
    
    '''
    def __init__(self,
                 pyramid=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 halfwindow=None,
                 max_halfwindow=None,
                 n_clusters=48,
                 step=1,
                 use_mds=True,
                 method='mean',
                 verbose=True,
                 random_state=None,
                 n_jobs=None):
        self.pyramid = pyramid
        self.halfwindow = halfwindow
        self.max_halfwindow = max_halfwindow
        self.n_clusters = n_clusters
        self.step = step
        self.use_mds = use_mds
        self.method = method
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _cluster_training_image(self, ti_patterns):
        '''
        Computes the clusters from the patterns of the training image
        '''
        if self.verbose:
            print('... ... Finding number of dimensions using PCA')
            print('... ... ... Initial dimensions:', ti_patterns.shape[1]*ti_patterns.shape[2])
        nb_components, ti_patterns_reduced = find_reducing_dimensions(ti_patterns.reshape((ti_patterns.shape[0], -1)),
                                                                      return_reduced_samples=True,
                                                                      random_state=self.random_state)
        if self.verbose:
            print('... ... ... Reduced dimensions:', nb_components)

        if self.use_mds == True:
            if self.verbose:
                print('... ... Reducing dimensions using MDS')
            reducing = MDS(n_components=nb_components,
                           n_jobs=n_jobs,
                           random_state=random_state,
                           dissimilarity='euclidean')
            ti_patterns_reduced = reducing.fit_transform(ti_patterns.reshape((ti_patterns.shape[0], -1)))

        if self.verbose:
            print('... ... Clustering patterns using kernel K-means')
        clustering = KernelKMeans(n_clusters=self.n_clusters,
                                  random_state=self.random_state)
        clustering.fit(ti_patterns_reduced)

        return clustering.labels_

    def _fit(self, training_image):
        '''
        Extracts the patterns from the training image, computes the clusters
        from the patterns, and computes the pattern prototypes for each cluster
        '''
        if isinstance(self.step, int):
            self.step = (self.step, self.step)
        if self.halfwindow is None:
            self.halfwindow = select_template_shape(training_image,
                                                    max_halfwindow=self.max_halfwindow,
                                                    step=self.step)
        elif isinstance(self.halfwindow, int):
            self.halfwindow = (self.halfwindow, self.halfwindow)
        if self.verbose:
            print('Template: ', 2*self.halfwindow[0] + 1, '*', 
                                2*self.halfwindow[1] + 1, sep='')
            print('Processing training image\n... Computing patterns')
        ti_patterns = get_patterns(training_image,
                                   self.halfwindow,
                                   step=self.step)
        if self.verbose:
            print('... ... Number of patterns:', ti_patterns.shape[0])
            print('... Computing clusters')
        ti_clusters = self._cluster_training_image(ti_patterns)
        if self.verbose:
            print('''... Computing clusters' prototypes''')
        self.prototypes_ = compute_cluster_prototypes(ti_patterns,
                                                      ti_clusters,
                                                      method=self.method)
        
        return np.histogram(ti_clusters, bins=self.n_clusters)[0]

    def fit_transform(self,
                      training_image,
                      realizations):
        '''
        Computes the histograms of patterns for the training image and the
        realizations, and the Jensen-Shannon distance between all those
        histograms, for each pyramid level
        
        Parameters
        ----------

        training_image : array, shape (n_cells_ti_y, n_cells_ti_x)
            Training image used to build the clusters and pattern prototypes.

        realizations : array, shape (n_methods, n_realizations_per_method, n_cells_y, n_cells_x)
            Realizations from different methods and parameter values to be
            compared with the training image.
        '''
        self.nb_methods = realizations.shape[0]
        self.nb_rez = realizations.shape[1]
        self.nb_grids = len(self.pyramid)
        
        self.distances_ = np.zeros((1 + self.nb_methods*self.nb_rez,
                                    1 + self.nb_methods*self.nb_rez,
                                    self.nb_grids))
        
        for g in self.pyramid:
            
            if self.nb_grids > 1 and self.verbose:
                print('Processing multiresolution grid ' + str(g), sep='')
            
            training_image_g = training_image
            if g != 1:
                out_shape = (int(training_image.shape[0]/g),
                             int(training_image.shape[1]/g))
                training_image_g = ski.transform.resize(training_image,
                                                        out_shape,
                                                        order=3,
                                                        anti_aliasing=False)
        
            distribution_ti = self._fit(training_image_g)
            distributions_rez = np.empty((self.nb_methods,
                                          self.nb_rez,
                                          self.n_clusters))

            if self.verbose:
                print('Processing realizations')
            for j in range(self.nb_methods):
                for i in range(self.nb_rez):
                    if self.verbose:
                        print('... method ' + str(j + 1) + '/' + str(self.nb_methods) \
                              + ', realization ' + str(i + 1) + '/' + str(self.nb_rez),
                              end='\r')
                    realization_g = realizations[j, i]
                    if g != 1:
                        out_shape = (int(realization_g.shape[0]/g),
                                     int(realization_g.shape[1]/g))
                        realization_g = ski.transform.resize(realizations[j, i],
                                                             out_shape,
                                                             order=3,
                                                             anti_aliasing=False)
                    clusters = assign_clusters(realization_g, self.prototypes_)
                    distributions_rez[j, i] = np.histogram(clusters,
                                                           bins=self.n_clusters)[0]

            compute_distances(self.distances_,
                              distribution_ti,
                              distributions_rez, 
                              g - self.pyramid[0],
                              self.nb_methods,
                              self.nb_rez,
                              self.verbose)
            if self.verbose:
                print('\n', end='')

        return self
    
    def score(self):
        '''
        Returns the ranking of the different methods that produced the realizations
        
        Returns
        -------

        rankings : dict
            Ranking for the between-realization variability (space of uncertainty),
            ranking for the within-realization variability, (reproduction the target statistics,
            i.e., the training image), and total ranking
        '''
        distance_between = np.empty((self.nb_methods, self.nb_grids))
        for j in range(self.nb_methods):
            for i in range(self.nb_grids):
                distance_between[j, i] = np.mean(self.distances_[1 + self.nb_rez*j:1 + self.nb_rez*(j + 1), 1:, i])
        distance_within = np.empty((self.nb_methods, self.nb_grids))
        for j in range(self.nb_methods):
            for i in range(self.nb_grids):
                distance_within[j, i] = np.mean(self.distances_[0, 1 + self.nb_rez*j:1 + self.nb_rez*(j + 1), i])
 
        weights = np.array([1/2**i for i in range(self.nb_grids)])
        ranking_between = np.empty((self.nb_methods, self.nb_methods))
        for j in range(self.nb_methods):
            for i in range(self.nb_methods):
                ranking_between[j, i] = np.sum(weights*distance_between[j]/distance_between[i])
        ranking_within = np.empty((self.nb_methods, self.nb_methods))
        for j in range(self.nb_methods):
            for i in range(self.nb_methods):
                ranking_within[j, i] = np.sum(weights*distance_within[j]/distance_within[i])
        ranking_total = ranking_between/ranking_within

        return dict(between=ranking_between,
                    within=ranking_within,
                    total=ranking_total)
