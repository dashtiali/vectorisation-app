from vectorisation import ATSnew
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetTemplateFunctionFeature"]

def GetTemplateFunctionFeature(barcodes, d=5, padding=.05):
   
    barcodes = [bar_cleaner(barcodes)] 
   
    features = ATSnew.tent_features(barcodes, d, padding)
   
    return features