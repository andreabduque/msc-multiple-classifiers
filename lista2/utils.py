import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

def kdn(k, X, y):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, indices = nbrs.kneighbors(X)
    kdn = np.zeros(len(indices))
    for i, neighbors in enumerate(indices):
        label = y[neighbors[0]]
        knn = y[neighbors[1:]]
        kdn[i] = len(knn[knn != label])/k

    return kdn   

def disagreement_measure(c1_pred, c2_pred, ground_truth):
    negated_xor = lambda x,y: np.logical_not(np.bitwise_xor(x,y))

    c1_correct = negated_xor(c1_pred,ground_truth)
    c2_correct =  negated_xor(c2_pred,ground_truth)

    if(np.array_equal(c1_correct, c2_correct)):
        return 0

    n_00, n_01, n_10, n11 =  confusion_matrix(c1_correct, c2_correct).ravel()

    return (n_01+n_10)/(n_00+n_01+n_10+n11)

def average_pairwise_diversity(values, L):
    pass

def gmean(y_true, y_pred):    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    return (specificity*sensitivity)**(1/2)
