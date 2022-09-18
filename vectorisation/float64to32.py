import numpy as np

__all__ = ["float64to32"]

def e_index(x):
    if ((len(str(x))>4)and(str(x)[-4] =='e')):
        y = int(str(x)[-3:])  
    else:
        y = len(str(x))
    return y

def float64to32(complex_coeff):
    n = max([max([e_index(c) for c in cp]) for cp in complex_coeff] +
            [max([e_index(c) for c in cp]) for cp in complex_coeff])
    n = max(0, n-38+1)
    complex_coeff = [[np.float32(x/10**n) for x in cp] for cp in complex_coeff]
    return complex_coeff
