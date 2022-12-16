import numpy as np
import teaspoon.ML.feature_functions as Ff
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetCarlssonCoordinatesFeature"]

def GetCarlssonCoordinatesFeature(barcode,*p):
    feature_vector = np.zeros(5)
    barcode = bar_cleaner(barcode)
    
    if(np.size(barcode) > 0):
        featureMatrix, _, _ = Ff.F_CCoordinates([barcode], 5)
        feature_vector = np.concatenate(
            [mat.flatten() for mat in featureMatrix[0:5]])

    return feature_vector