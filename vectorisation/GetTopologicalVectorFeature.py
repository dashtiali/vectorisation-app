import numpy as np
from gudhi import representations

__all__ = ["GetTopologicalVectorFeature"]

def GetTopologicalVectorFeature(barcode, thres = 10):

    if(np.size(barcode) > 0):
        topologicalVector = representations.vector_methods.TopologicalVector(threshold = thres)
        feature_vector = topologicalVector.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(thres)
        
    return feature_vector
