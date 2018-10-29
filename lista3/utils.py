import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

def kdn(k, X, y):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    kdn = np.zeros(len(indices))
    for i, neighbors in enumerate(indices):
        label = y[neighbors[0]]
        knn = y[neighbors[1:]]
        kdn[i] = len(knn[knn != label])/k

    return kdn, nbrs

def gmean(y_true, y_pred):    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    return (specificity*sensitivity)**(1/2)
