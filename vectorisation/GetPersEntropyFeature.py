import numpy as np
from vectorisation import GetNewMethods
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetPersEntropyFeature"]

def GetPersEntropyFeature(barcode, res=100):
    barcode = bar_cleaner(barcode)
    if (barcode.shape[0]) > 1:
        ent = GetNewMethods.Entropy(mode='vector', resolution = res, 
                                           normalized = False)
        feature_vector = ent.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
        
    return feature_vector