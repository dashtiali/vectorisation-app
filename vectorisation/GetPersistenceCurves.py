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

__all__ = ["GetPersistenceCurves"]


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


class BettiCurve(BaseEstimator, TransformerMixin):
    """
    Compute Betti curves from persistence diagrams. There are several modes of operation: with a given resolution (with or without a sample_range), with a predefined grid, and with none of the previous. With a predefined grid, the class computes the Betti numbers at those grid points. Without a predefined grid, if the resolution is set to None, it can be fit to a list of persistence diagrams and produce a grid that consists of (at least) the filtration values at which at least one of those persistence diagrams changes Betti numbers, and then compute the Betti numbers at those grid points. In the latter mode, the exact Betti curve is computed for the entire real line. Otherwise, if the resolution is given, the Betti curve is obtained by sampling evenly using either the given sample_range or based on the persistence diagrams.
    """

    def __init__(self, resolution=100, sample_range=[np.nan, np.nan], predefined_grid=None):
        """
        Constructor for the BettiCurve class.

        Parameters:
            resolution (int): number of sample for the piecewise-constant function (default 100).
            sample_range ([double, double]): minimum and maximum of the piecewise-constant function domain, of the form [x_min, x_max] (default [numpy.nan, numpy.nan]). It is the interval on which samples will be drawn evenly. If one of the values is numpy.nan, it can be computed from the persistence diagrams with the fit() method.
            predefined_grid (1d array or None, default=None): Predefined filtration grid points at which to compute the Betti curves. Must be strictly ordered. Infinities are ok. If None (default), and resolution is given, the grid will be uniform from x_min to x_max in 'resolution' steps, otherwise a grid will be computed that captures all changes in Betti numbers in the provided data.

        Attributes:
            grid_ (1d array): The grid on which the Betti numbers are computed. If predefined_grid was specified, `grid_` will always be that grid, independently of data. If not, the grid is fitted to capture all filtration values at which the Betti numbers change.

        Examples
        --------
        If pd is a persistence diagram and xs is a nonempty grid of finite values such that xs[0] >= pd.min(), then the results of:

        >>> bc = BettiCurve(predefined_grid=xs) # doctest: +SKIP
        >>> result = bc(pd) # doctest: +SKIP

        and

        >>> from scipy.interpolate import interp1d # doctest: +SKIP
        >>> bc = BettiCurve(resolution=None, predefined_grid=None) # doctest: +SKIP
        >>> bettis = bc.fit_transform([pd]) # doctest: +SKIP
        >>> interp = interp1d(bc.grid_, bettis[0, :], kind="previous", fill_value="extrapolate") # doctest: +SKIP
        >>> result = np.array(interp(xs), dtype=int) # doctest: +SKIP

        are the same.
        """

        if (predefined_grid is not None) and (not isinstance(predefined_grid, np.ndarray)):
            raise ValueError("Expected predefined_grid as array or None.")

        self.predefined_grid = predefined_grid
        self.resolution = resolution
        self.sample_range = sample_range

    def is_fitted(self):
        return hasattr(self, "grid_")

    def fit(self, X, y = None):
        """
        Fit the BettiCurve class on a list of persistence diagrams: if any of the values in **sample_range** is numpy.nan, replace it with the corresponding value computed on the given list of persistence diagrams. When no predefined grid is provided and resolution set to None, compute a filtration grid that captures all changes in Betti numbers for all the given persistence diagrams.

        Parameters:
            X (list of 2d arrays): Persistence diagrams.
            y (None): Ignored.
        """

        if self.predefined_grid is None:
            if self.resolution is None: # Flexible/exact version
                events = np.unique(np.concatenate([pd.flatten() for pd in X] + [[-np.inf]], axis=0))
                self.grid_ = np.array(events)
            else:
                self.sample_range = _automatic_sample_range(np.array(self.sample_range), X, y)
                self.grid_ = np.linspace(self.sample_range[0], self.sample_range[1], self.resolution)
        else:
            self.grid_ = self.predefined_grid # Get the predefined grid from user

        return self

    def transform(self, X):
        """
        Compute Betti curves.

        Parameters:
            X (list of 2d arrays): Persistence diagrams.

        Returns:
            `len(X).len(self.grid_)` array of ints: Betti numbers of the given persistence diagrams at the grid points given in `self.grid_`
        """

        if not self.is_fitted():
            raise NotFittedError("Not fitted.")

        if not X:
            X = [np.zeros((0, 2))]
        
        N = len(X)

        events = np.concatenate([pd.flatten(order="F") for pd in X], axis=0)
        sorting = np.argsort(events)
        offsets = np.zeros(1 + N, dtype=int)
        for i in range(0, N):
            offsets[i+1] = offsets[i] + 2*X[i].shape[0]
        starts = offsets[0:N]
        ends = offsets[1:N + 1] - 1

        bettis = [[0] for i in range(0, N)]

        i = 0
        for x in self.grid_:
            while i < len(sorting) and events[sorting[i]] <= x:
                j = np.searchsorted(ends, sorting[i])
                delta = 1 if sorting[i] - starts[j] < len(X[j]) else -1
                bettis[j][-1] += delta
                i += 1
            for k in range(0, N):
                bettis[k].append(bettis[k][-1])

        return np.array(bettis, dtype=int)[:, 0:-1]

    def fit_transform(self, X):
        """
        The result is the same as fit(X) followed by transform(X), but potentially faster.
        """

        if self.predefined_grid is None and self.resolution is None:
            if not X:
                X = [np.zeros((0, 2))]

            N = len(X)

            events = np.concatenate([pd.flatten(order="F") for pd in X], axis=0)
            sorting = np.argsort(events)
            offsets = np.zeros(1 + N, dtype=int)
            for i in range(0, N):
                offsets[i+1] = offsets[i] + 2*X[i].shape[0]
            starts = offsets[0:N]
            ends = offsets[1:N + 1] - 1

            xs = [-np.inf]
            bettis = [[0] for i in range(0, N)]

            for i in sorting:
                j = np.searchsorted(ends, i)
                delta = 1 if i - starts[j] < len(X[j]) else -1
                if events[i] == xs[-1]:
                    bettis[j][-1] += delta
                else:
                    xs.append(events[i])
                    for k in range(0, j):
                        bettis[k].append(bettis[k][-1])
                    bettis[j].append(bettis[j][-1] + delta)
                    for k in range(j+1, N):
                        bettis[k].append(bettis[k][-1])

            self.grid_ = np.array(xs)
            return np.array(bettis, dtype=int)

        else:
            return self.fit(X).transform(X)

    def __call__(self, diag):
        """
        Shorthand for transform on a single persistence diagram.
        """
        return self.fit_transform([diag])[0, :]



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
            p = p/np.sum(p)
            if self.mode == "scalar":
                ent = -np.dot(p, np.log(p))
                Xfit.append(np.array([[ent]]))
           
            else:
                ent = np.zeros(self.resolution)
                for j in range(num_pts_in_diag):
                    [px,py] = orig_diagram[j,:2]
                    min_idx = np.clip(np.ceil((px - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                    max_idx = np.clip(np.ceil((py - self.sample_range[0]) / step_x).astype(int), 0, self.resolution)
                    ent[min_idx:max_idx]-=p[j]*np.log(p[j])
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
            p = p/np.sum(p)

            lsp = np.zeros(self.resolution)
            for j in range(num_pts_in_diag):
                [px,py] = orig_diagram[j,:2]
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