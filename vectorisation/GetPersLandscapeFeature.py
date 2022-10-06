import numpy as np
from gudhi import representations
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetPersLandscapeFeature"]

def GetPersLandscapeFeature(barcode, res=100, num=5):
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        perLand = representations.Landscape(resolution=res,num_landscapes=num)
        feature_vector = perLand.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(num*res)
        
    return feature_vector
