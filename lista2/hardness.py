import numpy as np
from sklearn.metrics import confusion_matrix

def kdn():
    pass

def disagreement_measure(c1_pred, c2_pred, ground_truth):
    negated_xor = lambda x,y: np.logical_not(np.bitwise_xor(x,y))

    c1_correct = negated_xor(c1_pred,ground_truth)
    c2_correct =  negated_xor(c2_pred,ground_truth)

    if(np.array_equal(c1_correct, c2_correct)):
        return 0

    n_00, n_01, n_10, n11 =  confusion_matrix(c1_correct, c2_correct).ravel()

    return (n_01+n_10)/(n_00+n_01+n_10+n11)
