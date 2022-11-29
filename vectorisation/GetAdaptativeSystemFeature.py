from vectorisation import ATSnew
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetAdaptativeSystemFeature"]

def GetAdaptativeSystemFeature(barcodes_train, barcodes_test, y_train, 
                             model='gmm', d=25):
    
    barcodes_train = list(map(bar_cleaner, barcodes_train))
    barcodes_test = list(map(bar_cleaner, barcodes_test))
    
    _, feature_vector = ATSnew.adaptive_features(X_train=barcodes_train, 
                                                          X_test=barcodes_test, 
                                                          model=model, 
                                                          y_train=y_train, 
                                                          d=d)
    
    return feature_vector