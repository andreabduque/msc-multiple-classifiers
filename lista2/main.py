from arff2pandas import a2p
import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


from mlxtend.classifier import EnsembleVoteClassifier

from pruning import kappa_pruning, best_first
from utils import kdn, gmean

np.seterr(divide='ignore', invalid='ignore')

def print_results(results):
    for key in results.keys():
        print(key, end='\t')
    
    print()            
    for val in results.values():
        print("{}+-({})".format(str(round(np.mean(val), 2)), str(round(np.std(val), 2))), end='\t')

def test_prunning(kdn, pruning_function, pruning_name='kappa', M=400):
    results = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    results_pruned = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = es.fit(X_train, y_train)

        if(pruning_name == 'kappa'):
            model_pruned, size_pool = pruning_function(M, X_train, y_train, model)
        else:
            model_pruned, size_pool = pruning_function(X_train, y_train, model)

        y_pred = model.predict(X_test)
        y_pred_pruned = model_pruned.predict(X_test)

        for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
            results[name].append(metric(y_test, y_pred))
            results_pruned[name].append(metric(y_test, y_pred_pruned))
        
        results['pool_size'].append(100)
        results_pruned['pool_size'].append(size_pool)
        print('fold ', str(fold))
        fold += 1

    print('sem poda')
    print_results(results)
    print('com poda')
    print_results(results_pruned)

    

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

test_prunning(kdn(7, X, y), kappa_pruning, 'kappa')

#

# mean_fscore_model = []
# mean_fscore_pruned = []
# initial = time.time()
# for train_index, test_index in skf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     model = es.fit(X_train, y_train)
#     mean_fscore_model.append(f1_score(y_test, model.predict(X_test)))

#     #KAPPA PRUNING

#     estimators = kappa_pruning(500, X_train, model)
#     eclf = EnsembleVoteClassifier(clfs=estimators)   
#     model_pruned = eclf.fit(X_train, y_train) 
#     mean_fscore_pruned.append(f1_score(y_test, model_pruned.predict(X_test)))

#     # score, model_pruned, size_pool = best_first(X_train, y_train, model)
#     # print('tamanho do pruned pool')
#     # print(size_pool)
#     # mean_fscore_pruned.append(f1_score(y_test, model_pruned.predict(X_test)))
#     print('fold')

# print('tempo ',str(time.time() - initial) )
# print('F-measure do modelo sem poda')
# print(np.mean(mean_fscore_model))
# print('F-measure kappa pruning')
# print(np.mean(mean_fscore_pruned)) 

 