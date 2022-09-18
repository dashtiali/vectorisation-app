import numpy as np
from copy import deepcopy

__all__ = ["GetPersTropicalCoordinatesFeature"]

def GetPersTropicalCoordinatesFeature(barcode, r=28):
    feature_vector = np.zeros(7)
    if(np.size(barcode) > 0):
        #change the deaths by the lifetime
        new_barcode = deepcopy(barcode)
        new_barcode[:,1] = new_barcode[:,1]-new_barcode[:,0]
        #sort them so the bars with the longest lifetime appears first
        new_barcode = new_barcode[np.argsort(-new_barcode[:,1])]
        #Write the output of the selected polynomials
        feature_vector[0] = new_barcode[0,1]
        if barcode.shape[0] > 1:
            feature_vector[1] = new_barcode[0,1] + new_barcode[1,1]
            if barcode.shape[0] > 2:
                feature_vector[2] = new_barcode[0,1] + new_barcode[1,1] + new_barcode[2,1]
                if barcode.shape[0] > 3:
                    feature_vector[3] = new_barcode[0,1] + new_barcode[1,1] + new_barcode[2,1] + new_barcode[3,1]
        feature_vector[4] = sum(new_barcode[:,1])
        #In each row, take the minimum between the birth time and r*lifetime
        aux_array = np.array(list(map(lambda x : min(r*x[1], x[0]), new_barcode)))
        feature_vector[5] = sum(aux_array)
        M = max(aux_array + new_barcode[:,1])
        feature_vector[6] = sum(M - (aux_array + new_barcode[:,1]))
            
    return feature_vector