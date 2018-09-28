import itertools
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np

from mlxtend.classifier import EnsembleVoteClassifier

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

#Receives input from validation set
#Returns pruned model
def kappa_pruning(input_data, output_data, model, validation_input, validation_output, M=50):   
    combs = list(itertools.combinations(model.estimators_, 2))
    naive = []
    for cl1, cl2 in combs:
        kappa_score = cohen_kappa_score(cl1.predict(validation_input), cl2.predict(validation_input))
        if(not np.isnan(kappa_score)):            
            naive.append((kappa_score, cl1, cl2))

    naive_sorted = sorted(naive, key=lambda tup: tup[0], reverse=True)   
    new_estimators = set()
    i = 0
    while(len(new_estimators) < M and len(naive_sorted)):
        pair = naive_sorted.pop()
        new_estimators.add(pair[1])
        new_estimators.add(pair[2])    
        i += 1
        
    eclf = EnsembleVoteClassifier(clfs=list(new_estimators), refit=False)   
    return eclf.fit(input_data, output_data), len(new_estimators)

#Modificar para nao dar fit no conjunto todo
#modificar pra pegar kdn do conjunto de teste!!!!!!
def best_first(input_data, output_data, model, validation_input, validation_output, M=None):
    estimatorsByFscore = []
    for i, estimator in enumerate(model.estimators_):
        score = f1_score(validation_output, estimator.predict(validation_input))
        #if(score > 0):
         #   estimatorsByFscore.append((score, estimator))

   #if(not len(estimatorsByFscore)):
    #   return model, len(model.estimators_) 
    
    estimatorsByFscore = sorted(estimatorsByFscore, key=lambda tup: tup[0], reverse=True)
    estimators = [tup[1] for tup in estimatorsByFscore]

    #score, ensemble object, pool size
    best_ensemble = (-1, None, 0)
    
    for i in range(1, len(estimators) + 1):
        new_pool = estimators[0:i]
        eclf = EnsembleVoteClassifier(clfs=new_pool, refit=False)   
        ensemble = eclf.fit(input_data, output_data)
        score = f1_score(validation_output, ensemble.predict(validation_input)) 
        if(score > best_ensemble[0]):
            best_ensemble = (score, ensemble, i)


    return best_ensemble[1], best_ensemble[2]





    