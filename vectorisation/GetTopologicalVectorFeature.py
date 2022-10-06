import numpy as np
from vectorisation import GetNewMethods
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetTopologicalVectorFeature"]

def GetTopologicalVectorFeature(barcode, thres = 10):
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        topologicalVector = GetNewMethods.TopologicalVector(threshold = thres)
        feature_vector = topologicalVector.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(thres)
        
    return feature_vector
