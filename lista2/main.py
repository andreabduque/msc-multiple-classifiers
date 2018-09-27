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
    print()

def test_prunning(kdn, pruning_function, pruning_name='kappa', M=50):
    results = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    results_pruned_all = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    results_pruned_hard = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    results_pruned_easy = {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[]}
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        kdn_train, kdn_test = kdn[train_index], kdn[test_index]

        model = es.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
            results[name].append(metric(y_test, y_pred))


        kdn_hard_indices = np.argwhere(kdn_train > 0.5).flatten()
        kdn_easy_indices = np.argwhere(kdn_train < 0.5).flatten()

        model_pruned_all, size_pool_all = pruning_function(X_train, y_train, model,  True, M=M)
        model_pruned_hard, size_pool_hard = pruning_function(X_train, y_train, model,  False, kdn_hard_indices,  M)
        model_pruned_easy, size_pool_easy = pruning_function(X_train, y_train, model,  False, kdn_easy_indices,  M)
        
        y_pred_pruned_all = model_pruned_all.predict(X_test)
        y_pred_pruned_hard = model_pruned_hard.predict(X_test)
        print(model_pruned_easy)
        y_pred_pruned_easy = model_pruned_easy.predict(X_test)

        for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
            results_pruned_all[name].append(metric(y_test, y_pred_pruned_all))
            results_pruned_hard[name].append(metric(y_test, y_pred_pruned_hard))
            results_pruned_easy[name].append(metric(y_test, y_pred_pruned_easy))

        results_pruned_all['pool_size'].append(size_pool_all)
        results_pruned_hard['pool_size'].append(size_pool_hard)       
        results_pruned_easy['pool_size'].append(size_pool_easy)         
        results['pool_size'].append(100)

        #adicionar métricas de diversidade
       
        print('fold ', str(fold))
        fold += 1

    print('sem poda')
    print_results(results)
    print('com poda')
    print('all')
    print_results(results_pruned_all)
    print('hard')
    print_results(results_pruned_hard)
    print('easy')
    print_results(results_pruned_easy)
  

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

#test_prunning(kdn(7, X, y), kappa_pruning, 'kappa')
test_prunning(kdn(7, X, y), best_first, 'best')

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

 