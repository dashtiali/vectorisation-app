import numpy as np
from vectorisation import ATS
from gudhi import representations
from sklearn.cluster import KMeans

from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetAdaptativeSystemFeature"]

def GetAdaptativeSystemFeature(barcodes_train, barcodes_test, y_train, 
                             model='gmm', d=25):
    
    barcodes_train = list(map(bar_cleaner, barcodes_train))
    barcodes_test = list(map(bar_cleaner, barcodes_test))
    
    features_train, features_test = ATS.adaptive_features(X_train=barcodes_train, 
                                                          X_test=barcodes_test, 
                                                          model=model, 
                                                          y_train=y_train, 
                                                          d=d)
    
    feature_vector = np.vstack([features_train, features_test])
    return feature_vector