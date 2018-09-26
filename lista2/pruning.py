import itertools
from sklearn.metrics import cohen_kappa_score
import numpy as np

from mlxtend.classifier import EnsembleVoteClassifier


#Receives input from validation set
#Returns pruned model
def kappa_pruning(M, input_data, model):
    combs = list(itertools.combinations(model.estimators_, 2))
    naive = []
    for cl1, cl2 in combs:
        kappa_score = cohen_kappa_score(cl1.predict(input_data), cl2.predict(input_data))
        if(not np.isnan(kappa_score)):            
            naive.append((kappa_score, cl1, cl2))

 
    naive_sorted = sorted(naive, key=lambda tup: tup[0])
    aux = naive_sorted[0:M]
    aux = []
    for i in range(0,M):
        aux.append(naive_sorted[i][1])
        aux.append(naive_sorted[i][2])
        
    new_estimators = list(set(aux))
    print('Pool reduzido de ', str(len(model.estimators_)), 'para ', str(len(new_estimators)))
    return new_estimators
