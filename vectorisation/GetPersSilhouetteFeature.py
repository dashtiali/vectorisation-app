import numpy as np
from gudhi import representations
from vectorisation.bar_cleaner import bar_cleaner
__all__ = ["GetPersSilhouetteFeature"]

def GetPersSilhouetteFeature(barcode, res=100, w=1):
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        persSilhouette = representations.vector_methods.Silhouette(resolution=res, 
                                                                   weight=lambda x : (x[1]-x[0])**w)
        feature_vector = persSilhouette.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(res)

    return feature_vector
