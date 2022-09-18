import numpy as np
import teaspoon.ML.feature_functions as Ff

__all__ = ["GetCarlssonCoordinatesFeature"]

def GetCarlssonCoordinatesFeature(barcode, FN=5):
    feature_vector = np.zeros(FN)

    if(np.size(barcode) > 0):
        featureMatrix, _, _ = Ff.F_CCoordinates([barcode], FN)
        feature_vector = np.concatenate(
            [mat.flatten() for mat in featureMatrix[0:FN]])

    return feature_vector