import numpy as np
from vectorisation import GetPersistenceCurves

__all__ = ["GetPersEntropyFeature"]

def GetPersEntropyFeature(barcode, res=100):

    if (barcode.shape[0]) > 1:
        ent = GetPersistenceCurves.Entropy(mode='vector', resolution = res)
        feature_vector = ent.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
        
    return feature_vector