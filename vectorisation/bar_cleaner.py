__all__ = ["bar_cleaner"]

def bar_cleaner(barcode):
    return barcode[barcode[:,0]!=barcode[:,1]]