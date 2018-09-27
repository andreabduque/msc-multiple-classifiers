from arff2pandas import a2p
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score
from pruning import kappa_pruning, best_first
from mlxtend.classifier import EnsembleVoteClassifier

from utils import kdn 

import time

np.seterr(divide='ignore', invalid='ignore')

#return instances according to instance hardness
def get_validation_set():
    pass

def get_data(file_path):
    with open(file_path) as f:
        df = a2p.load(f)
        df = df.interpolate()
        input_features = df.drop(["defects@{false,true}"], axis=1)
        output_class = np.where(df["defects@{false,true}"] == 'true', 1, 0)
        return np.array(input_features), np.array(output_class)

    return

X, y = get_data('../cm1.arff')
skf = StratifiedKFold(n_splits=10, random_state=42)
es = BaggingClassifier(base_estimator= Perceptron(max_iter=1000, class_weight = 'balanced'), 
                    n_estimators=100, 
                    max_samples=1.0, 
                    max_features=1.0, 
                    bootstrap=True,
                    bootstrap_features=False, 
                    n_jobs=4)

kdn(7, X, y)

# mean_fscore_model = []
# mean_fscore_pruned = []
# initial = time.time()
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     model = es.fit(X_train, y_train)
#     mean_fscore_model.append(f1_score(y_test, model.predict(X_test)))

#     #KAPPA PRUNING

#     """ estimators = kappa_pruning(500, X_train, model)
#     eclf = EnsembleVoteClassifier(clfs=estimators)   
#     model_pruned = eclf.fit(X_train, y_train) 
#     mean_fscore_pruned.append(f1_score(y_test, model_pruned.predict(X_test))) """

#     score, model_pruned, size_pool = best_first(X_train, y_train, model)
#     print('tamanho do pruned pool')
#     print(size_pool)
#     mean_fscore_pruned.append(f1_score(y_test, model_pruned.predict(X_test)))

# print('tempo ',str(time.time() - initial) )
# print('F-measure do modelo sem poda')
# print(np.mean(mean_fscore_model))
# print('F-measure kappa pruning')
# print(np.mean(mean_fscore_pruned)) 

 