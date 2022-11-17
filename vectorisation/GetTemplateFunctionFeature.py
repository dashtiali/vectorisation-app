import numpy as np
from vectorisation import ATS
from gudhi import representations
from sklearn.cluster import KMeans

from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetTemplateFunctionFeature"]

def GetTemplateFunctionFeature(barcodes_train, barcodes_test, d=5, padding=.05):
    
    barcodes_train = list(map(bar_cleaner, barcodes_train))
    barcodes_test = list(map(bar_cleaner, barcodes_test))
    
    features_train, features_test = ATS.tent_features(barcodes_train, 
                                                      barcodes_test, 
                                                      d, padding)
    
    feature_vector = np.vstack([features_train, features_test])
    return feature_vector