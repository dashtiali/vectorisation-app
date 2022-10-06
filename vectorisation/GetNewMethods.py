# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# Modification of Gudhi script vectorization_methods.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.


import numpy as np
from sklearn.base          import BaseEstimator, TransformerMixin
from sklearn.exceptions    import NotFittedError
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.neighbors     import DistanceMetric
from sklearn.metrics       import pairwise

from gudhi.representations.preprocessing import DiagramScaler, BirthPersistenceTransform

__all__ = ["GetNewMethods"]


def _automatic_sample_range(sample_range, X, y):
        """
        Compute and returns sample range from the persistence diagrams if one of the sample_range values is numpy.nan.

        Parameters:
            sample_range (a numpy array of 2 float): minimum and maximum of all piecewise-linear function domains, of
                the form [x_min, x_max].
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        nan_in_range = np.isnan(sample_range)
        if nan_in_range.any():
            try:
                pre = DiagramScaler(use=True, scalers=[([0], MinMaxScaler()), ([1], MinMaxScaler())]).fit(X,y)
                [mx,my] = [pre.scalers[0][1].data_min_[0], pre.scalers[1][1].data_min_[0]]
                [Mx,My] = [pre.scalers[0][1].data_max_[0], pre.scalers[1][1].data_max_[0]]
                return np.where(nan_in_range, np.array([mx, My]), sample_range)
            except ValueError:
                # Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
                pass
        return sample_range





class Entropy(BaseEstimator, TransformerMixin):
    """
    This is a class for computing persistence entropy. Persistence entropy is a statistic for persistence diagrams inspired from Shannon entropy. This statistic can also be used to compute a feature vector, called the entropy summary function. See https://arxiv.org/pdf/1803.08304.pdf for more details. Note that a previous implementation was contributed by Manuel Soriano-Trigueros.
    """
    def __init__(self, mode="scalar", normalized=True, resolution=100, sample_range=[np.nan, np.nan]):
        """
        Constructor for the Entropy class.

        Parameters:
            mode (string): what entropy to compute: either "scalar" for computing the entropy statistics, or "vector" for computing the entropy summary functions (default "scalar").
            normalized (bool): whether to normalize the entropy summary function (default True). Used only if **mode** = "vector". 
            resolution (int): number of sample for the entropy summary function (default 100). Used only if **mode** = "vector".
            sample_range ([double, double]): minimum and maximum of the entropy summary function domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method. Used only if **mode** = "vector".
        """
        self.mode, self.normalized, self.resolution, self.sample_range = mode, normalized, resolution, sample_range

    def fit(self, X, y=None):
        """
        Fit the Entropy class on a list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        return self

    def transform(self, X):
        """
        Compute the entropy for each persistence diagram individually and concatenate the results.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
    
        Returns:
            numpy array with shape (number of diagrams) x (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]
        new_X = BirthPersistenceTransform().fit_transform(X)  

        for i in range(num_diag):
            orig_diagram, diagram, num_pts_in_diag = X[i], new_X[i], X[i].shape[0]
            try:
                #new_diagram = DiagramScaler(use=True, scalers=[([1], MaxAbsScaler())]).fit_transform([diagram])[0]
                new_diagram = DiagramScaler().fit_transform([diagram])[0]
            except ValueError:
                # Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
                assert len(diagram) == 0
                new_diagram = np.empty(shape = [0, 2])   
            p = new_diagram[:,1]
            #we need this condition to avoid dividing by zero
            if (p!=0).any():
                p = p/np.sum(p)
            #This function is necessary to guarantee that 0*log(0)=0 later
            log0 = lambda x: 0 if x == 0 else np.log(x)
            log0 = np.frompyfunc(log0, 1, 1)
            if self.mode == "scalar":
                ent = -np.dot(p, log0(p))
                Xfit.append(np.array([[ent]]))
            else:
                ent = np.zeros(self.resolution)
                for j in range(num_pts_in_diag):
                    [px,py] = orig_diagram[j,:2]
                    if px!=py:
                        min_idx = np.clip(np.ceil((px - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                        max_idx = np.clip(np.ceil((py - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                        ent[min_idx:max_idx]-=p[j]*log0(p[j])
                if self.normalized:
                    ent = ent / np.linalg.norm(ent, ord=1)
                Xfit.append(np.reshape(ent,[1,-1]))

        Xfit = np.concatenate(Xfit, axis=0)
        return Xfit

    def __call__(self, diag):
        """
        Apply Entropy on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            numpy array with shape (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        return self.fit_transform([diag])[0,:]
    
class Lifespan(BaseEstimator, TransformerMixin):
    """
    This is a class to calculate the normalized lifespan curve as defined in persistence curves (doi: 10.1007/s10444-021-09893-4).
    """
    def __init__(self, resolution=100, sample_range=[np.nan, np.nan]):
        """
        Constructor for the Lifespan class.

        Parameters:
            resolution (int): number of sample for the lifespan persistence curve (default 100).
            sample_range ([double, double]): minimum and maximum of the lifespan persistence curve domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method.
        """
        self.resolution, self.sample_range = resolution, sample_range

    def fit(self, X, y=None):
        """
        Fit the Lifespan class on a list of persistence diagrams.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
        return self

    def transform(self, X):
        """
        Compute the lifespan for each persistence diagram individually and concatenate the results.

        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
    
        Returns:
            numpy array with shape (number of diagrams) x (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        num_diag, Xfit = len(X), []
        x_values = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        step_x = x_values[1] - x_values[0]
        new_X = BirthPersistenceTransform().fit_transform(X)        

        for i in range(num_diag):
            orig_diagram, diagram, num_pts_in_diag = X[i], new_X[i], X[i].shape[0]
            try:
                new_diagram = DiagramScaler().fit_transform([diagram])[0]
            except ValueError:
                # Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
                assert len(diagram) == 0
                new_diagram = np.empty(shape = [0, 2])
                
            p = new_diagram[:,1]
            lsp = np.zeros(self.resolution)
            for j in range(num_pts_in_diag):
                [px,py] = orig_diagram[j,:2]
                if px!=py:
                    min_idx = np.clip(np.ceil((px - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                    max_idx = np.clip(np.ceil((py - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                    lsp[min_idx:max_idx]+=p[j]
                
            Xfit.append(np.reshape(lsp,[1,-1]))

        Xfit = np.concatenate(Xfit, axis=0)
        return Xfit

    def __call__(self, diag):
        """
        Apply Lifespan on a single persistence diagram and outputs the result.

        Parameters:
            diag (n x 2 numpy array): input persistence diagram.

        Returns:
            numpy array with shape (1 if **mode** = "scalar" else **resolution**): output entropy.
        """
        return self.fit_transform([diag])[0,:]

class TopologicalVector(BaseEstimator, TransformerMixin):
    """
    This is a class for computing topological vectors from a list of persistence diagrams. The topological vector associated to a persistence diagram is the sorted vector of a slight modification of the pairwise distances between the persistence diagram points. See https://diglib.eg.org/handle/10.1111/cgf12692 for more details.
    """
    def __init__(self, threshold=10):
        """
        Constructor for the TopologicalVector class.
        Parameters:
            threshold (int): number of distances to keep (default 10). This is the dimension of the topological vector. If -1, this threshold is computed from the list of persistence diagrams by considering the one with the largest number of points and using the dimension of its corresponding topological vector as threshold. 
        """
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fit the TopologicalVector class on a list of persistence diagrams (this function actually does nothing but is useful when TopologicalVector is included in a scikit-learn Pipeline).
        Parameters:
            X (list of n x 2 or n x 1 numpy arrays): input persistence diagrams.
            y (n x 1 array): persistence diagram labels (unused).
        """
        return self

    def transform(self, X):
        """
        Compute the topological vector for each persistence diagram individually and concatenate the results.
        Parameters:
            X (list of n x 2 numpy arrays): input persistence diagrams.
    
        Returns:
            numpy array with shape (number of diagrams) x (**threshold**): output topological vectors.
        """
        if self.threshold == -1:
            thresh = np.array([X[i].shape[0] for i in range(len(X))]).max()
        else:
            thresh = self.threshold

        num_diag = len(X)
        Xfit = np.zeros([num_diag, thresh])

        for i in range(num_diag):

            diagram, num_pts_in_diag = X[i], X[i].shape[0]
            pers = 0.5 * (diagram[:,1]-diagram[:,0])
            pers1 = np.zeros([num_pts_in_diag, num_pts_in_diag])
            pers2 = np.zeros([num_pts_in_diag, num_pts_in_diag])
            for k in range(num_pts_in_diag):
                pers1[:,k]=pers
                pers2[k,:]=pers
            min_pers = np.minimum(pers1,pers2)
            # Works fine with sklearn 1.0, but an ValueError exception is thrown on past versions
            try:
                distances = DistanceMetric.get_metric("chebyshev").pairwise(diagram)
            except ValueError:
                # Empty persistence diagram case - https://github.com/GUDHI/gudhi-devel/issues/507
                assert len(diagram) == 0
                distances = np.empty(shape = [0, 0])
            vect = np.flip(np.sort(np.triu(np.minimum(distances, min_pers)), axis=None), 0)
            
            dim = min(len(vect), thresh)
            Xfit[i, :dim] = vect[:dim]

        return Xfit

    def __call__(self, diag):
        """
        Apply TopologicalVector on a single persistence diagram and outputs the result.
        Parameters:
            diag (n x 2 numpy array): input persistence diagram.
        Returns:
            numpy array with shape (**threshold**): output topological vector.
        """
        return self.fit_transform([diag])[0,:]