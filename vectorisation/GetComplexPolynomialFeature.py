import numpy as np
from gudhi import representations
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetComplexPolynomialFeature"]

def GetComplexPolynomialFeature(barcode, thres = 10, pol_type='R'):
    #We pick the first tresh largest cofficient from the polynomial.
    #There are different pol_type, 'R' is the most common but unstable,
    #'S' and 'T' sends points close to the diagonal to points close to zero.
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        complexPolynomial = representations.vector_methods.ComplexPolynomial(threshold = thres, 
                                                                             polynomial_type = pol_type)
        feature_vector = complexPolynomial.fit_transform([barcode]).flatten()
        feature_vector = np.concatenate([np.array([np.real(i), np.imag(i)]) 
                                         for i in feature_vector])
    else:
    	feature_vector = np.zeros(2*thres)
        
    return feature_vector