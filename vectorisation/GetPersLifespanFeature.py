import numpy as np
from vectorisation import GetPersistenceCurves

__all__ = ["GetPersLifespanFeature"]

def GetPersLifespanFeature(barcode, res=100):
    feature_vector = []

    if(np.size(barcode) > 0):
        lfsp = GetPersistenceCurves.Lifespan(resolution = res)
        feature_vector = lfsp.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
    
    return feature_vector