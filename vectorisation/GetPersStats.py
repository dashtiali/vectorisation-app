import numpy as np
from vectorisation import GetNewMethods
from vectorisation.bar_cleaner import bar_cleaner

__all__ = ["GetPersStats"]

def GetPersStats(barcode):
    barcode = bar_cleaner(barcode)
    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        diff_barcode = np.subtract([i[1] for i in barcode], [
                                   i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Number of Bars
        bc_count = len(diff_barcode)
        # Persitent Entropy
        ent = GetNewMethods.Entropy()
        bc_ent = ent.fit_transform([barcode])

        bar_stats = np.array([bc_av0, bc_av1, bc_std0, bc_std1, bc_med0, bc_med1,
                              bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_count,  # ])
                              bc_ent[0][0]])
    else:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    bar_stats[~np.isfinite(bar_stats)] = 0

    return bar_stats
