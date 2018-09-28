from arff2pandas import a2p
import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


from mlxtend.classifier import EnsembleVoteClassifier

from pruning import kappa_pruning, best_first
from utils import kdn, gmean, average_kappa, average_disagreement

np.seterr(divide='ignore', invalid='ignore')

def print_results(results):
    #remove nan from kappa and disagreement
    d = np.array(results['disagreement'])
    d = d[~np.isnan(d)]
    results['disagreement'] = d
    k = np.array(results['kappa'])
    k = k[~np.isnan(k)]
    results['kappa'] = k

    for key in results.keys():
        print(key, end='\t')
    print()
    for val in results.values():
        print("{}+-({})".format(str(round(np.mean(val), 2)), str(round(np.std(val), 2))), end='\t')
    print()

def test_prunning(kdn, M=25):
    new_result = lambda : {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[], 'pool_size':[], 'disagreement':[], 'kappa':[]}
    results = new_result()    
    results_pruned_all = [new_result(), new_result()]
    results_pruned_hard = [new_result(), new_result()]
    results_pruned_easy = [new_result(), new_result()]
    
    fold = 1
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        kdn_train = kdn[train_index]

        model = es.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
            results[name].append(metric(y_test, y_pred))

        results['pool_size'].append(100)
        results['kappa'].append(average_kappa(model.estimators_, X_test))
        results['disagreement'].append(average_disagreement(model.estimators_, X_test, y_test))

        hard_indices = np.argwhere(kdn_train > 0.5).flatten()
        easy_indices = np.argwhere(kdn_train < 0.5).flatten()
        # print('tam validacao hard')
        # len(len(hard_indices))
        # print('tam validacao easy')
        # len(len(easy_indices))
        # print('tam validacao all')
        # print(len(X_train))s
       
        for i, pruning_function in enumerate([kappa_pruning, best_first]):        
            model_pruned_all, size_pool_all = pruning_function(X_train, y_train, model, X_train , y_train, M)
            model_pruned_hard, size_pool_hard = pruning_function(X_train, y_train, model,  X_train[hard_indices], y_train[hard_indices],  M)
            model_pruned_easy, size_pool_easy = pruning_function(X_train, y_train, model,  X_train[easy_indices], y_train[easy_indices],  M)
            
            y_pred_pruned_all = model_pruned_all.predict(X_test)
            y_pred_pruned_hard = model_pruned_hard.predict(X_test)
            y_pred_pruned_easy = model_pruned_easy.predict(X_test)

            for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
                results_pruned_all[i][name].append(metric(y_test, y_pred_pruned_all))
                results_pruned_hard[i][name].append(metric(y_test, y_pred_pruned_hard))
                results_pruned_easy[i][name].append(metric(y_test, y_pred_pruned_easy))

            results_pruned_all[i]['pool_size'].append(size_pool_all)
            results_pruned_hard[i]['pool_size'].append(size_pool_hard)       
            results_pruned_easy[i]['pool_size'].append(size_pool_easy)         
            

            #adicionar métricas de diversidade
            results_pruned_all[i]['kappa'].append(average_kappa(model_pruned_all.clfs, X_test))
            results_pruned_hard[i]['kappa'].append(average_kappa(model_pruned_hard.clfs, X_test))       
            results_pruned_easy[i]['kappa'].append(average_kappa(model_pruned_easy.clfs, X_test))         
            

            results_pruned_all[i]['disagreement'].append(average_disagreement(model_pruned_all.clfs, X_test, y_test))
            results_pruned_hard[i]['disagreement'].append(average_disagreement(model_pruned_hard.clfs, X_test, y_test))       
            results_pruned_easy[i]['disagreement'].append(average_disagreement(model_pruned_easy.clfs, X_test, y_test))                     
            
        print('fold ', str(fold))
        fold += 1
    
    
    print('sem poda')
    print_results(results)

    print('com poda ', 'kappa')
    print('all')
    print_results(results_pruned_all[0])
    print('hard')
    print_results(results_pruned_hard[0])
    print('easy')
    print_results(results_pruned_easy[0])

    print('com poda ', 'best')
    print('all')
    print_results(results_pruned_all[1])
    print('hard')
    print_results(results_pruned_hard[1])
    print('easy')
    print_results(results_pruned_easy[1])
  

def get_data(file_path):
    with open(file_path) as f:
        df = a2p.load(f)
        df = df.interpolate()
        input_features = df.drop(["defects@{false,true}"], axis=1)
        output_class = np.where(df["defects@{false,true}"] == 'true', 1, 0)
        return np.array(input_features), np.array(output_class)

    return

# print('---------cm1----------')
# X, y = get_data('../cm1.arff')
# skf = StratifiedKFold(n_splits=10, random_state=42)
# es = BaggingClassifier(base_estimator= Perceptron(max_iter=1000, class_weight = 'balanced'), 
#                     n_estimators=100, 
#                     max_samples=1.0, 
#                     max_features=1.0, 
#                     bootstrap=True,
#                     bootstrap_features=False, 
#                     n_jobs=4)

# test_prunning(kdn(7, X, y))

# print('---------kc2----------')

# X, y = get_data('../kc2.arff')
# skf = StratifiedKFold(n_splits=10, random_state=42)
# es = BaggingClassifier(base_estimator= Perceptron(max_iter=1000, class_weight = 'balanced'), 
#                     n_estimators=100, 
#                     max_samples=1.0, 
#                     max_features=1.0, 
#                     bootstrap=True,
#                     bootstrap_features=False, 
#                     n_jobs=4)

# test_prunning(kdn(7, X, y))




print('---------jm1----------')
X, y = get_data('../jm1.arff')
skf = StratifiedKFold(n_splits=10, random_state=42)
es = BaggingClassifier(base_estimator= Perceptron(max_iter=1000, class_weight = 'balanced'), 
                    n_estimators=100, 
                    max_samples=1.0, 
                    max_features=1.0, 
                    bootstrap=True,
                    bootstrap_features=False, 
                    n_jobs=4)

test_prunning(kdn(7, X, y))
 