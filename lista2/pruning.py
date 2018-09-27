import itertools
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

from mlxtend.classifier import EnsembleVoteClassifier

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

#Receives input from validation set
#Returns pruned model
def kappa_pruning(M, input_data, output_data, model):
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
    eclf = EnsembleVoteClassifier(clfs=new_estimators)   
    return eclf.fit(input_data, output_data), len(new_estimators)

def best_first(input_data, output_data, model):
    estimatorsByFscore = []
    for i, estimator in enumerate(model.estimators_):
        score = f1_score(output_data, estimator.predict(input_data))
        if(score > 0):
            estimatorsByFscore.append((score, estimator))
    
    estimatorsByFscore = sorted(estimatorsByFscore, key=lambda tup: tup[0], reverse=True)
    estimators = [tup[1] for tup in estimatorsByFscore]

    #score, ensemble object, pool size
    best_ensemble = (-1, None, 0)
    for i in range(1, len(estimators) + 1):
        new_pool = estimators[0:i]
        eclf = EnsembleVoteClassifier(clfs=new_pool)   
        ensemble = eclf.fit(input_data, output_data)
        score = f1_score(output_data, ensemble.predict(input_data)) 

        if(score > best_ensemble[0]):
            best_ensemble = (score, ensemble, i)

    return best_ensemble[1], best_ensemble[2]





    