__all__ = ["bar_cleaner"]
import numpy as np

def bar_cleaner(barcode):
    if (np.size(barcode) > 0):
        return barcode[barcode[:,0]!=barcode[:,1]]
    else:
        return []