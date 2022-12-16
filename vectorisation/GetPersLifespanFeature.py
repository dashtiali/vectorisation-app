import numpy as np
from vectorisation import GetNewMethods
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetPersLifespanFeature"]

def GetPersLifespanFeature(barcode, res=100):
    feature_vector = []
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        lfsp = GetNewMethods.Lifespan(resolution = res)
        feature_vector = lfsp.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
    
    return feature_vector