import numpy as np
from gudhi import CubicalComplex

__all__ = ["GetCubicalComplexPDs"]

def GetCubicalComplexPDs(img, img_dim):
    cub_filtration = CubicalComplex(
        dimensions=img_dim, top_dimensional_cells=img)
    cub_filtration.persistence()
    pds = [cub_filtration.persistence_intervals_in_dimension(0),
           cub_filtration.persistence_intervals_in_dimension(1)]
    for j in range(pds[0].shape[0]):
        if pds[0][j,1]==np.inf:
            pds[0][j,1]=256
    for j in range(pds[1].shape[0]):
        if pds[1][j,1]==np.inf:
            pds[1][j,1]=256

    return pds
