"""Analysis of distance (ANODI)"""

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
from scipy.stats import differential_entropy
from skimage.transform import resize as imresize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

from pyanodi.kernel_k_means import *


################################################################################
# Utils

def _get_patterns(x, window, step):
    """
    Gets the patterns an image.
    """
    slices = tuple(slice(0, None, s) for s in step)
    
    return np.lib.stride_tricks.sliding_window_view(x, window)[slices].reshape(-1, *window)


def _compute_mean_differential_entropy(training_image, windows, step):
    """
    Computes the mean differential entropy of all the patterns from the training
    image for several pattern sizes.
    """
    mean_entropies = np.empty(len(windows))
    for i, window in enumerate(windows):
        patterns = _get_patterns(training_image, window, step)
        mean_entropies[i] = np.mean(differential_entropy(patterns.reshape(patterns.shape[0], -1), base=2, axis=1))
        
    return mean_entropies


def _compute_forward_second_difference(samples):
    """
    Computes the forward second difference for a set of samples.
    """
    return samples[2:] - 2*samples[1:-1] + samples[:-2]


def _compute_log_gaussian(samples, mean, var):
    """
    Computes the log of a gaussian distribution for a set of samples.
    """
    return len(samples)*np.log(1/np.sqrt(2*np.pi*var))\
           - np.sum(((samples - mean)**2)/(2*var))


def _compute_profile_log_likelihood(samples, common_scale=False):
    """
    Computes the profile log-likelihood of a set of samples.
    """
    start = 1
    stop = len(samples) - 2
    if common_scale == True:
        start = 0
        stop = len(samples) - 1
    
    log_likelihoods = np.full(len(samples), np.nan)
    for q in range(start, stop):

        mean_1 = np.mean(samples[:q + 1])
        mean_2 = np.mean(samples[q + 1:])
        var_1 = np.var(samples[:q + 1])
        var_2 = np.var(samples[q + 1:])
        if common_scale == True:
            sigma = ((q - 1)*var_1 + (len(samples) - q - 1)*var_2)/(len(samples) - 2)
            var_1 = sigma
            var_2 = sigma

        lq_1 = _compute_log_gaussian(samples[:q + 1], mean_1, var_1)
        lq_2 = _compute_log_gaussian(samples[q + 1:], mean_2, var_2)
        log_likelihoods[q] = lq_1 + lq_2
        
    return log_likelihoods


def _find_elbow(x, use_second_diff=True):
    """
    Finds the elbow on a scree plot using the maximum of a log-likelihood function.
    """
    _x = _compute_forward_second_difference(x) if use_second_diff == True else x
    log_likelihoods = _compute_profile_log_likelihood(_x)
    
    return np.nanargmax(log_likelihoods)


def _select_template_shape(training_image, max_window=None, step=1):
    """
    Selects the template shape using the elbow plot so that the template size
    records the pattern variations from the training image.
    """
    if max_window is None:
        # TODO: Need to find a way to deal with rectangular halfwindows
        max_window = np.min(training_image.shape)
        max_window = tuple(int(0.4*max_window) for i in range(training_image.ndim))
    if isinstance(step, int):
        step = tuple(step for i in range(training_image.ndim))

    windows = np.array([np.arange(3, max_window[i] + 1) for i in range(training_image.ndim)]).T
    mean_entropies = _compute_mean_differential_entropy(training_image, windows, step)
    best_index = _find_elbow(mean_entropies, use_second_diff=True)
    
    return windows[best_index]


def _find_reducing_dimensions(samples,
                              return_reduced_samples=False,
                              random_state=None):
    """
    Finds the number of components for dimensionality reduction using the elbow
    plot so that as much information as possible is preserved.
    """
    reducing = PCA(random_state=random_state)
    samples_reduced = reducing.fit_transform(samples)
    
    nb_components = _find_elbow(reducing.explained_variance_ratio_,
                                use_second_diff=False)
    
    if return_reduced_samples == False:
        return nb_components
    return nb_components, samples_reduced[:, :nb_components]


def _find_medoid(samples):
    """
    Finds the medoid pattern from a set of patterns.
    """
    return np.argmin(np.sum(euclidean_distances(samples.reshape(samples.shape[0], -1)), axis=0))


def _compute_cluster_prototypes(patterns, clusters, method='mean'):
    """
    Computes the prototype of the patterns from each cluster.
    """
    if method != 'mean' and method != 'medoid':
        raise ValueError("""Method must be 'mean' or 'medoid' """)
    
    id_clusters = np.unique(clusters)
    prototypes = np.empty((id_clusters.shape[0],) + patterns.shape[1:])
    for i in range(id_clusters.shape[0]):
        if method == 'mean':
            prototypes[i] = np.mean(patterns[clusters == id_clusters[i]], axis=0)
        elif method == 'medoid':
            id_medoid = _find_medoid(patterns[clusters == id_clusters[i]])
            prototypes[i] = patterns[clusters == id_clusters[i]][id_medoid]
        
    return prototypes


def _assign_clusters(realization, prototypes, step):
    """
    Assigns the patterns of a realization to a cluster based on the Euclidean
    distance to the pattern prototypes computed from the training image.
    """
    window = prototypes.shape[1:]
    n = np.prod(np.subtract(realization.shape, window))
    if n <= 0:
        raise ValueError("Template larger than resized realization")
        
    axis = tuple(range(2, prototypes.ndim + 1))
    patterns = _get_patterns(realization, window, step)
    
    return np.argmin(np.sum((patterns[np.newaxis] - prototypes[:, np.newaxis])**2, axis=axis), axis=0)


@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def _rel_entr(x, y):
    """
    Computes logarithm operations for the Jensen-Shannon distance.
    """
    if np.isnan(x) or np.isnan(y):
        return np.nan
    elif x > 0. and y > 0.:
        return x*np.log(x/y)
    elif x == 0. and y >= 0.:
        return 0.
    else:
        return np.inf


@nb.jit(nopython=True)
def _jensen_shannon(p, q, base=None):
    """
    Computes the Jensen-Shannon distance, based on SciPy function
    distance.jensenshannon, see:

    https://scipy.github.io/devdocs/generated/scipy.spatial.distance.jensenshannon.html
    """
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/np.sum(p)
    q = q/np.sum(q)
    m = (p + q)/2.
    left = _rel_entr(p, m)
    right = _rel_entr(q, m)
    js = np.sum(left) + np.sum(right)
    if base is not None:
        js /= np.log(base)

    return np.sqrt(js/2.)


@nb.jit(nopython=True, nogil=True, parallel=True)
def _compute_distances(distances, distribution_ti, distributions_rez, g, n_methods, n_rez, verbose):
    """
    Fills a matrix of distances based on the Jensen-Shannon distance between
    distributions.
    """
    if verbose:
        print('\nComputing distances\n... Within')
    for j in range(n_methods*n_rez):
        v = int(j/n_rez)
        u = int(j%n_rez)
        distances[j + 1, 0, g] = _jensen_shannon(distributions_rez[v, u],
                                                 distribution_ti)
        distances[0, j + 1, g] = distances[j + 1, 0, g]
    if verbose:
        print('... Between')
    for j in nb.prange(n_methods*n_rez):
        v = int(j/n_rez)
        u = int(j%n_rez)
        for i in range(j + 1, n_methods*n_rez):
            z = int(i/n_rez)
            w = int(i%n_rez)
            distances[j + 1, i + 1, g] = _jensen_shannon(distributions_rez[v, u],
                                                         distributions_rez[z, w])
            distances[i + 1, j + 1, g] = distances[j + 1, i + 1, g]

            
################################################################################
# ANODI

class ANODI:
    """
    Analysis of distance (ANODI).
    
    Parameters
    ----------
    pyramid : int or array-like of shape (n_levels), default=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        Scaling coefficients for multiscale analysis.
    window : int or array-like, optional
        Size of the window to extract the patterns from the training image
        and the realizations.
    max_window : int or array-like, optional
        Maximum possible size of the window. Only used when `window` is
        None, and set to 40% of the training image size by default.
    n_clusters : int, optional
        Number of clusters to group patterns and build the histograms of patterns.
    step : int or array-like, optional
        Step between two patterns in the training image, to limit the number of
        patterns during clustering and reduce the computational burden.
    use_mds : bool, optional
        If False, the dimensionality reduction before clustering the patterns is
        done using a PCA, otherwise it is done using the MDS method SMACOF. The
        original code uses the classical MDS method with the Euclidean distance
        to process continuous variables, which is equivalent to a PCA but far
        less computationally efficient.
    kmeans_params : dict, optional
        Parameters to pass to the kernel k-means used to compute the pattern
        prototypes.
    method : {'mean', 'medoid'}, optional
        Method to compute the pattern prototype for each cluster, either 'mean'
        or 'medoid'.
    verbose : bool, optional
        If True, print information about the successive steps, if False, nothing
        is printed.
    random_state : int or RandomState instance, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        'np.random'.
    n_jobs : int, optional
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
    """
    def __init__(self,
                 pyramid=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 window=None,
                 max_window=None,
                 n_clusters=48,
                 step=1,
                 use_mds=False,
                 kmeans_params=None,
                 method='mean',
                 verbose=True,
                 random_state=None,
                 n_jobs=None):
        self.pyramid = pyramid
        if isinstance(pyramid, int):
            self.pyramid = [pyramid]
        self.window = window
        self.max_window = max_window
        self.n_clusters = n_clusters
        self.step = step
        self.use_mds = use_mds
        self.kmeans_params = {} if kmeans_params is None else kmeans_params
        self.method = method
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def _cluster_training_image(self, ti_patterns):
        """
        Computes the clusters from the patterns of the training image.
        """
        if self.verbose:
            print('... ... Finding number of dimensions using PCA')
            print('... ... ... Initial dimensions:', ti_patterns.shape[1]*ti_patterns.shape[2])
        nb_components, ti_patterns_reduced = _find_reducing_dimensions(ti_patterns.reshape((ti_patterns.shape[0], -1)),
                                                                       return_reduced_samples=True,
                                                                       random_state=self.random_state)
        if self.verbose:
            print('... ... ... Reduced dimensions:', nb_components)

        if self.use_mds == True:
            if self.verbose:
                print('... ... Reducing dimensions using MDS')
            reducing = MDS(n_components=nb_components,
                           n_jobs=self.n_jobs,
                           random_state=self.random_state,
                           dissimilarity='euclidean')
            ti_patterns_reduced = reducing.fit_transform(ti_patterns.reshape((ti_patterns.shape[0], -1)))

        if self.verbose:
            print('... ... Clustering patterns using kernel K-means')
        clustering = KernelKMeans(n_clusters=self.n_clusters,
                                  **self.kmeans_params,
                                  random_state=self.random_state)
        clustering.fit(ti_patterns_reduced)

        return clustering.labels_

    def _fit(self, training_image):
        """
        Extracts the patterns from the training image, computes the clusters
        from the patterns, and computes the pattern prototypes for each cluster.
        """
        if isinstance(self.step, int):
            self.step = tuple(self.step for i in range(training_image.ndim))
        if self.window is None:
            self.window = _select_template_shape(training_image,
                                                 max_window=self.max_window,
                                                 step=self.step)
        elif isinstance(self.window, int):
            self.window = (self.window for i in range(training_image.ndim))
        if self.verbose:
            print('Template:', '*'.join(str(i) for i in self.window))
            print('Processing training image\n... Computing patterns')
        ti_patterns = _get_patterns(training_image, self.window, step=self.step)
        if self.verbose:
            print('... ... Number of patterns:', ti_patterns.shape[0])
            print('... Computing clusters')
        ti_clusters = self._cluster_training_image(ti_patterns)
        if self.verbose:
            print("""... Computing clusters' prototypes""")
        prototypes = _compute_cluster_prototypes(ti_patterns,
                                                 ti_clusters,
                                                 method=self.method)
        distribution_ti = np.histogram(ti_clusters, bins=self.n_clusters)[0]
        
        return distribution_ti, prototypes

    def fit_transform(self,
                      training_image,
                      realizations):
        """
        Computes the histograms of patterns for the training image and the
        realizations, and the Jensen-Shannon distance between all those
        histograms, for each pyramid level.
        
        Parameters
        ----------
        training_image : array, shape (n_cells_ti_y, n_cells_ti_x)
            Training image used to build the clusters and pattern prototypes.
        realizations : array, shape (n_methods, n_rez_per_method, n_cells_y, n_cells_x)
            Realizations from different methods and parameter values to be
            compared with the training image.
        """
        self.n_methods = realizations.shape[0]
        self.n_rez = realizations.shape[1]
        self.nb_grids = len(self.pyramid)
        
        self.prototypes_ = {}
        self.distances_ = np.zeros((1 + self.n_methods*self.n_rez,
                                    1 + self.n_methods*self.n_rez,
                                    self.nb_grids))
        
        for g in self.pyramid:
            
            if self.nb_grids > 1 and self.verbose:
                print('Processing multiresolution grid ' + str(g), sep='')
            
            training_image_g = training_image
            if g != 1:
                out_shape = tuple(int(i/g) for i in training_image.shape)
                training_image_g = imresize(training_image,
                                            out_shape,
                                            order=3,
                                            anti_aliasing=False)
        
            distribution_ti, self.prototypes_[g] = self._fit(training_image_g)
            distributions_rez = np.empty((self.n_methods,
                                          self.n_rez,
                                          self.n_clusters))

            if self.verbose:
                print('Processing realizations')
            for j in range(self.n_methods):
                for i in range(self.n_rez):
                    if self.verbose:
                        print('... method ' + str(j + 1) + '/' + str(self.n_methods) \
                              + ', realization ' + str(i + 1) + '/' + str(self.n_rez) \
                              + ' '*len(str(self.n_methods) + str(self.n_rez)),
                              end='\r')
                    realization_g = realizations[j, i]
                    if g != 1:
                        out_shape = tuple(int(i/g) for i in training_image.shape)
                        realization_g = imresize(realizations[j, i],
                                                 out_shape,
                                                 order=3,
                                                 anti_aliasing=False)
                    clusters = _assign_clusters(realization_g, self.prototypes_[g], self.step)
                    distributions_rez[j, i] = np.histogram(clusters,
                                                           bins=self.n_clusters)[0]

            _compute_distances(self.distances_,
                               distribution_ti,
                               distributions_rez, 
                               g - self.pyramid[0],
                               self.n_methods,
                               self.n_rez,
                               self.verbose)
            if self.verbose:
                print('\n', end='')

        return self
    
    def score(self):
        """
        Returns the ranking of the different methods that produced the realizations.
        
        Returns
        -------
        rankings : dict
            Ranking for the between-realization variability (space of uncertainty),
            ranking for the within-realization variability, (reproduction the target
            statistics, i.e., the training image), and total ranking.
        """
        distance_between = np.empty((self.n_methods, self.nb_grids))
        for j in range(self.n_methods):
            for i in range(self.nb_grids):
                distance_between[j, i] = np.mean(self.distances_[1 + self.n_rez*j:1 + self.n_rez*(j + 1), 1:, i])
        distance_within = np.empty((self.n_methods, self.nb_grids))
        for j in range(self.n_methods):
            for i in range(self.nb_grids):
                distance_within[j, i] = np.mean(self.distances_[0, 1 + self.n_rez*j:1 + self.n_rez*(j + 1), i])
 
        weights = np.array([1/2**i for i in range(1, self.nb_grids + 1)])
        ranking_between = np.empty((self.n_methods, self.n_methods))
        for j in range(self.n_methods):
            for i in range(self.n_methods):
                ranking_between[j, i] = np.sum(weights*distance_between[j]/distance_between[i])
        ranking_within = np.empty((self.n_methods, self.n_methods))
        for j in range(self.n_methods):
            for i in range(self.n_methods):
                ranking_within[j, i] = np.sum(weights*distance_within[j]/distance_within[i])
        ranking_total = ranking_between/ranking_within

        return dict(between=ranking_between,
                    within=ranking_within,
                    total=ranking_total)
