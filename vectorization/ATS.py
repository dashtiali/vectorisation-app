'''
This code is a collection of functions obtained from
    https://github.com/barnesd8/machine_learning_for_persistence/blob/master/mlforpers/persistence_methods/ATS.py
Under the following license:
   Copyright 2019 Luis G. Polanco Contreras
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import numpy as np
# import multidim
import itertools
import os
#import hdbscan
import sys
import pandas as pd

from scipy import interpolate
from copy import deepcopy
from matplotlib.patches import Ellipse
from ripser import ripser
from persim import plot_diagrams
from numba import jit, njit, prange
from sklearn import mixture
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeClassifier


import matplotlib.pyplot as plt



def limits_box(list_dgms):
	'''
	This function computes the min and max of a collection of pers. dgms. in the birth-lifespan space.
	
	:param list_dgms: List of persistent diagrmas
	:type list_dgms: list
	:return: list -- mins = [min_birth, min_lifespan] and maxs = [max_birth, max_lifespan]
	'''

	list_dgms_temp = deepcopy(list_dgms)
	mins = np.inf*np.ones(2)
	maxs = -np.inf*np.ones(2)


	for dgm in list_dgms_temp:
		dgm[:,1] = dgm[:,1]-dgm[:,0] # Turns the birth-death into birth-lifespan
		mins = np.minimum(np.amin(dgm, axis=0), mins)
		maxs = np.maximum(np.amax(dgm, axis=0), maxs)

	return mins-0.05, maxs+0.05

	# return np.array([0,0]), np.array([1,1])

def box_centers(list_dgms, d, padding):
	'''
	This function computes the collection of centers use to define tent functions, as well as the size the tent.
	:param list_dgms: List of persistent diagrmas
	:type list_dgms: list
	:param d: number of a bins in each axis.
	:type d: int
	:param padding: this increases the size of the box aournd the collection of persistent diagrmas
	:type padding: float
	:return: numpy array, float -- d x 2 array of centers, size of the tent domain
	'''

	minimums, maximums = limits_box(list_dgms)
	birth_min = minimums[0] - padding
	lifespan_min = minimums[1] - padding
	birth_max = maximums[0] + padding
	lifespan_max = maximums[1] + padding

	birth_step = (birth_max - birth_min)/(d+1)
	lifespan_step = (lifespan_max - lifespan_min)/(d+1)

	birth_coord = []
	lifespan_coord = []
	for i in range(1,d):
		birth_coord.append(birth_min + birth_step*i)
		lifespan_coord.append(lifespan_min + lifespan_step*i)

	x, y = np.meshgrid(birth_coord, lifespan_coord)

	x = x.flatten() # center of the box, birth coordinate
	y = y.flatten() # center of the box, lifespan coordiante

	return np.column_stack((x,y)), max(birth_step, lifespan_step)

def f_box(x, center, delta):
	'''
	Computes the function
	.. math::
		g_{(a,b), \delta}(x) = \max \\left\{ 0,1 - \\frac{1}{\delta} \max\\left\{ | x-a | , | y-b | \\right\} \\right\}
	:param x: point to evaluate the function :math:`g_{(a,b), \delta}`.
	:type x: numpy array
	:param center: Center of the tenf function.
	:type center: numpy array
	:param delta: size fo the tent function domain.
	:type delta: float
	:return: float -- tent function :math:`g_{(a,b), \delta}(x)` evaluated at `x`.
	'''
	x = deepcopy(x)
	
	x[:,1] = x[:,1] - x[:,0] # Change death to lifespan.
	
	temp = np.maximum(0, 1 - (1/delta)*np.maximum(np.abs(x[:,0] - center[0]), np.abs(x[:,1] - center[1])))
	
	return temp


def f_ellipse (x, center=np.array([0,0]), axis=np.array([1,1]), rotation=np.array([[1,0],[0,1]])):
	'''
	Computes a bump function centered with an ellipsoidal domain centered ac `c`, rotaded by 'rotation' and with axis given by 'axis'. The bump function is computed using the gollowing formula 
	.. math::
		f_{A,c} (x) = \max \\left\{ 0, 1 - (x - c)^T A (x - c)\\right\}
	:param x: point to avelatuate the function :math:`f_{A,c}`
	:type z: Numpy array
	:param center: center of the ellipse
	:type center: Numpy array
	:param axis: Size f themjor an minor axis of the ellipse
	:type axis: Numpy array
	:param rotation: Rotation matrix for the ellipse
	:type rotation: Numpy array
	:return: float -- value of :math:`f_{A,c} (x)`.
	'''
	sigma = np.diag(np.power(axis, -2))
	x_centered = np.subtract(x, center)
	temp = x_centered@rotation@sigma@np.transpose(rotation)@np.transpose(x_centered)
	temp = np.diag(temp)

	return np.maximum(0, 1-temp)


#@jit(parallel=True)
def f_dgm(dgm, function, **keyargs):
	'''
	Given a persistend diagram :math:`D = (S,\mu)` and a compactly supported function in :math:`\mathbb{R}^2`, this function computes
	.. math::
		\\nu_{D}(f) = \sum_{x\in S} f(x)\mu(x)
	:param dgm: persistent diagram, array of points in :math:`\mathbb{R}^2`.
	:type dgm: Numpy array
	:param function: Compactly supported function in :math:`\mathbb{R}^2`.
	:type function: function
	:param keyargs: Additional arguments required by `funciton`
	:type keyargs: Dicctionary
	:return: float -- value of :math:`\\nu_{D}(f)`.
	'''

	temp = function(dgm, **keyargs)

	return sum(temp)

#@jit(parallel=True)
def feature(list_dgms, function, **keyargs):
	'''
	Given a collection of persistent diagrams and a compactly supported function in :math:`\mathbb{R}^2`, computes :math:`\\nu_{D}(f)` for each diagram :math:`D` in the collection.
	:param list_dgms: list of persistent diagrams
	:type list_dgms: list
	:param function: Compactly supported function in :math:`\mathbb{R}^2`.
	:type function: function
	:param keyargs: Additional arguments required by `funciton`
	:type keyargs: Dicctionary
	:return: Numpy array -- Array of values :math:`\\nu_{D}(f)` for each diagram :math:`D` in the collection `list_dgms`.
	'''
	num_diagrams = len(list_dgms)

	feat = np.zeros(num_diagrams)
	for i in range(num_diagrams):
		feat[i] = f_dgm(list_dgms[i], function, **keyargs)

	return feat

def get_all_features(list_dgms, list_ellipses, function):
	'''
	This function computes all the features for a colelction of persistent diagrams given a list ot ellipses.
	:param list_dgms: list of persistent diagrams
	:type list_dgms: list
	:param list_ellipses: List of dicctionaries. Each dicctionary represents a ellipse. It must have the following keys: `mean`, `std` and `rotation`.
	:type list_ellipses: list
	:param function: Compactly supported function in :math:`\mathbb{R}^2`.
	:type function: function
	:return: Numpy array -- 
	'''
	features = np.zeros((len(list_dgms), len(list_ellipses)))
	for i in range(len(list_ellipses)):
		args = {key:list_ellipses[i][key] for key in ['mean', 'std', 'rotation']}
		args['center'] = args.pop('mean')
		args['axis'] = args.pop('std')

		features[:,i] = feature(list_dgms, function, **args)

	return features

def get_all_features_boxes(list_dgms, centers, delta):

	features = np.zeros((len(list_dgms), len(centers)))
	for i in range(len(centers)):
		args = {'center':centers[i], 'delta':delta}

		features[:,i] = feature(list_dgms, f_box, **args)

	return features


def tent_features(X_train, X_test, d=5, padding=.05):
    centers, delta = box_centers(X_train, d, padding) 

    X_train_features = get_all_features_boxes(X_train, centers, delta)

    X_test_features = get_all_features_boxes(X_test, centers, delta)

    return X_train_features, X_test_features

def adaptive_features(X_train, X_test, model, y_train, d=25):
    if model == "gmm":

        print('Begin GMM...')

        X_train_temp = np.vstack(X_train)
        gmm_f_train=[]
        for i in range(len(X_train)):
            gmm_f_train.append(y_train[i]*np.ones(len(X_train[i])))
        gmm_f_train = np.concatenate(gmm_f_train)

        gmm = mixture.BayesianGaussianMixture(n_components=d, covariance_type='full', max_iter=int(10e4)).fit(X_train_temp, gmm_f_train)

        ellipses = []
        for i in range(len(gmm.means_)):
            L, v = np.linalg.eig(gmm.covariances_[i])
            temp = {'mean':gmm.means_[i], 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':gmm.weights_[i]}
            temp['std'] = 3*temp['std']
            ellipses.append(temp)

        X_train_features = get_all_features(X_train, ellipses, f_ellipse)

        X_test_features = get_all_features(X_test, ellipses, f_ellipse)

        
    # elif model == "hdb":
    #     print('Begin HDBSCAN...')
    #     X_train_temp = np.vstack(X_train)

    #     clusterer = hdbscan.HDBSCAN()

    #     clusterer.fit(X_train_temp)

    #     num_clusters = clusterer.labels_.max()

    #     ellipses = []
    #     for i in range(num_clusters):
    #         cluster_i = X_train_temp[clusterer.labels_ == i]

    #         en = np.mean(clusterer.probabilities_[clusterer.labels_ == i])

    #         mean = np.mean(cluster_i, axis=0)
    #         cov_matrix = np.cov(cluster_i.transpose())

    #         L,v = np.linalg.eig(cov_matrix)

    #         temp = {'mean':mean, 'std':np.sqrt(L), 'rotation':v.transpose(), 'radius':max(np.sqrt(L)), 'entropy':en}
    #         temp['std'] = 2*temp['std']
    #         ellipses.append(temp)

    #     X_train_features = get_all_features(X_train, ellipses, f_ellipse)

    #     X_test_features = get_all_features(X_test, ellipses, f_ellipse)
        
        
    else:
        print("Not a valid model type")
    return X_train_features, X_test_features
