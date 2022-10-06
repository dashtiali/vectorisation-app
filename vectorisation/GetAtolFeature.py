import numpy as np
from gudhi import representations
from sklearn.cluster import KMeans

from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetAtolFeature"]

def GetAtolFeature(barcode_list, k=2):
    qt = KMeans(n_clusters=k, random_state=1)
    
    barcode_list = list(map(bar_cleaner, barcode_list))
    
    atol = representations.vector_methods.Atol(quantiser=qt)
    #each row is the vector corresponding to each barcode
    feature_vector = atol.fit_transform(barcode_list)
    return feature_vector