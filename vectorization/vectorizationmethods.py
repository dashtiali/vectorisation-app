import numpy as np
from gudhi import representations, CubicalComplex
from sklearn.cluster import KMeans
from vectorization.bar_cleaner import bar_cleaner
from vectorization import GetNewMethods, ATS
import teaspoon.ML.feature_functions as Ff
from copy import deepcopy



__all__ = ["GetAdaptativeSystemFeature",
           "GetAlgebraicFunctions",
           "GetAtolFeature",
           "GetBettiCurveFeature",
           "GetComplexPolynomialFeature",
           "GetCubicalComplexPDs",
           "GetEntropySummary",
           "GetPersImageFeature",
           "GetPersLandscapeFeature",
           "GetPersLifespanFeature",
           "GetPersSilhouetteFeature",
           "GetPersStats",
           "GetPersTropicalCoordinatesFeature",
           "GetTemplateFunctionFeature"
           ]

def GetAdaptativeSystemFeature(barcodes_train, barcodes_test, model='gmm', d=25):
    
    barcodes_train = list(map(bar_cleaner, barcodes_train))
    barcodes_test = list(map(bar_cleaner, barcodes_test))
    
    # y_train is given as an input but never used, so we give a dummy input.
    features_train, features_test = ATS.adaptive_features(X_train=barcodes_train, 
                                                          X_test=barcodes_test, 
                                                          model=model, 
                                                          y_train=np.ones(len(barcodes_train)), 
                                                          d=d)
    
    feature_vector = np.vstack([features_train, features_test])
    return feature_vector

def GetAlgebraicFunctions(barcode,*p):
    feature_vector = np.zeros(5)
    barcode = bar_cleaner(barcode)
    
    if(np.size(barcode) > 0):
        featureMatrix, _, _ = Ff.F_CCoordinates([barcode], 5)
        feature_vector = np.concatenate(
            [mat.flatten() for mat in featureMatrix[0:5]])

    return feature_vector

def GetAtolFeature(barcode_list, k=2):
    qt = KMeans(n_clusters=k, random_state=1)
    
    barcode_list = list(map(bar_cleaner, barcode_list))
    
    atol = representations.vector_methods.Atol(quantiser=qt)
    #each row is the vector corresponding to each barcode
    feature_vector = atol.fit_transform(barcode_list)
    return feature_vector

def GetBettiCurveFeature(barcode, res=100):
    barcode = bar_cleaner(barcode)
    
    if(np.size(barcode) > 0):
        bettiCurve = representations.vector_methods.BettiCurve(resolution=res)
        feature_vector = bettiCurve.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(res)
        
    return feature_vector

def GetComplexPolynomialFeature(barcode, thres = 10, pol_type='R'):
    #We pick the first tresh largest cofficient from the polynomial.
    #There are different pol_type, 'R' is the most common but unstable,
    #'S' and 'T' sends points close to the diagonal to points close to zero.
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        complexPolynomial = representations.vector_methods.ComplexPolynomial(threshold = thres, 
                                                                             polynomial_type = pol_type)
        feature_vector = complexPolynomial.fit_transform([barcode]).flatten()
        feature_vector = np.stack((feature_vector.real,feature_vector.imag),-1)

    else:
    	feature_vector = np.zeros((thres,2))
        
    return feature_vector


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

def GetEntropySummary(barcode, res=100):
    barcode = bar_cleaner(barcode)
    if (barcode.shape[0]) > 1:
        ent = GetNewMethods.Entropy(mode='vector', resolution = res, 
                                           normalized = False)
        feature_vector = ent.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
        
    return feature_vector

def GetPersImageFeature(barcode, bw=0.2, r=20):
    barcode = bar_cleaner(barcode)
    res=[r,r]
    if(np.size(barcode) > 0):
        perImg = representations.PersistenceImage(bandwidth=bw, resolution=res)
        feature_vector = perImg.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(res[0]**2)

    return feature_vector

def GetPersLandscapeFeature(barcode, res=100, num=5):
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        perLand = representations.Landscape(resolution=res,num_landscapes=num)
        feature_vector = perLand.fit_transform([barcode])[0]
    else:
        feature_vector = np.zeros(num*res)
        
    return feature_vector

def GetPersLifespanFeature(barcode, res=100):
    feature_vector = []
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        lfsp = GetNewMethods.Lifespan(resolution = res)
        feature_vector = lfsp.fit_transform([barcode]).flatten()
    else:
        feature_vector = np.zeros(res)
    
    return feature_vector

def GetPersSilhouetteFeature(barcode, res=100, w=1):
    barcode = bar_cleaner(barcode)
    if(np.size(barcode) > 0):
        persSilhouette = representations.vector_methods.Silhouette(resolution=res, 
                                                                   weight=lambda x : (x[1]-x[0])**w)
        feature_vector = persSilhouette.fit_transform([barcode])[0]
    else:
    	feature_vector = np.zeros(res)

    return feature_vector

def GetPersStats(barcode,*p):
    barcode = bar_cleaner(barcode)
    if (np.size(barcode) > 0):
        # Average of Birth and Death of the barcode
        bc_av0, bc_av1 = np.mean(barcode, axis=0)
        # STDev of Birth and Death of the barcode
        bc_std0, bc_std1 = np.std(barcode, axis=0)
        # Median of Birth and Death of the barcode
        bc_med0, bc_med1 = np.median(barcode, axis=0)
        # Intercuartil range of births and death
        bc_iqr0, bc_iqr1 = np.subtract(*np.percentile(barcode, [75, 25], axis=0)) 
        # Range of births and deaths
        bc_r0, bc_r1=np.max(barcode, axis=0) - np.min(barcode, axis=0)
        # Percentiles of births and deaths
        bc_p10_0,bc_p10_1=np.percentile(barcode, 10, axis=0)
        bc_p25_0,bc_p25_1=np.percentile(barcode,25, axis=0)
        bc_p75_0,bc_p75_1=np.percentile(barcode, 75, axis=0)
        bc_p90_0,bc_p90_1=np.percentile(barcode, 90, axis=0)
        
        
        avg_barcodes = (barcode[:,1] + barcode[:,0])/2
        # Average of midpoints of the barcode
        bc_av_av = np.mean(avg_barcodes)
        # STDev of midpoints of the barcode
        bc_std_av = np.std(avg_barcodes)
        # Median of midpoints of the barcode
        bc_med_av = np.median(avg_barcodes)
        # Intercuartil range of midpoints
        bc_iqr_av = np.subtract(*np.percentile(avg_barcodes, [75, 25])) 
        # Range of midpoints
        bc_r_av = np.max(avg_barcodes) - np.min(avg_barcodes)
        # Percentiles of midpoints
        bc_p10_av = np.percentile(barcode, 10)
        bc_p25_av=np.percentile(barcode,25)
        bc_p75_av=np.percentile(barcode, 75)
        bc_p90_av=np.percentile(barcode, 90)
        
        diff_barcode = np.subtract([i[1] for i in barcode], [
                                   i[0] for i in barcode])
        diff_barcode = np.absolute(diff_barcode)
        # Average of the length of Bars
        bc_lengthAverage = np.mean(diff_barcode)
        # STD of length of Bars
        bc_lengthSTD = np.std(diff_barcode)
        # Median of length of Bars
        bc_lengthMedian = np.median(diff_barcode)
        # Intercuartil range of length of the bars
        bc_lengthIQR= np.subtract(*np.percentile(diff_barcode, [75, 25]))
        # Range of length of the bars
        bc_lengthR=np.max(diff_barcode) - np.min(diff_barcode)
        # Percentiles of lengths of the bars
        bc_lengthp10=np.percentile(diff_barcode, 10)
        bc_lengthp25=np.percentile(diff_barcode, 25)
        bc_lengthp75=np.percentile(diff_barcode, 75)
        bc_lengthp90=np.percentile(diff_barcode, 90)
        
        # Number of Bars
        bc_count = len(diff_barcode)
        # Persitent Entropy
        ent = GetNewMethods.Entropy()
        bc_ent = ent.fit_transform([barcode])
        
        bar_stats = np.array([bc_av0, bc_std0, bc_med0, bc_iqr0, bc_r0, bc_p10_0, bc_p25_0, bc_p75_0, bc_p90_0,
                              bc_av1, bc_std1, bc_med1, bc_iqr1, bc_r1, bc_p10_1, bc_p25_1, bc_p75_1, bc_p90_1,
                              bc_av_av, bc_std_av, bc_med_av, bc_iqr_av, bc_r_av, bc_p10_av, bc_p25_av, bc_p75_av, bc_p90_av,
                              bc_lengthAverage, bc_lengthSTD, bc_lengthMedian, bc_lengthIQR, bc_lengthR, bc_lengthp10, bc_lengthp25, bc_lengthp75, bc_lengthp90,
                              bc_count, bc_ent[0][0]])
    else:
        bar_stats = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0,0, 0])

    bar_stats[~np.isfinite(bar_stats)] = 0

    return bar_stats

def GetPersTropicalCoordinatesFeature(barcode, r=28):
    barcode = bar_cleaner(barcode)
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

def GetTemplateFunctionFeature(barcodes_train, barcodes_test, d=5, padding=.05):
    
    barcodes_train = list(map(bar_cleaner, barcodes_train))
    barcodes_test = list(map(bar_cleaner, barcodes_test))
    
    features_train, features_test = ATS.tent_features(barcodes_train, 
                                                      barcodes_test, 
                                                      d, padding)
    
    feature_vector = np.vstack([features_train, features_test])
    return feature_vector