
from arff2pandas import a2p
import pandas as pd
import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from deslib.des.knora_e import KNORAE
from utils import kdn, gmean
import math

#select between knn or des
def select_classifier(threshold, k_neigh, X_dsel, y_dsel, test_instance):
    use_des = False
    kdn_dsel, nbrs = kdn(k_neigh, X_dsel, y_dsel)
    distances, neighbors = nbrs.kneighbors([test_instance])
    kdn_neigh = kdn_dsel[neighbors[0]]

    ih_instance = np.mean(kdn_neigh)

    num = 0
    denom = 0
    very_near = False
    for i, val in enumerate(kdn_neigh):
        if(distances[0][i] < 1e-7):
            ih_instance = val
            very_near = True
            break
        else:
            weight = 1/(distances[0][i]**2)
            num += weight*val
            denom += weight
    
    if(not very_near):
        ih_instance = num/denom

    if(ih_instance > 0.4):
        use_des = True

    return use_des
        

def print_results(results):
    for key in results.keys():
        print(key, end='\t')
    print()
    for val in results.values():
        print("{}+-({})".format(str(round(np.mean(val), 2)), str(round(np.std(val), 2))), end='\t')
    print()

def get_data(file_path):
    with open(file_path) as f:
        df = a2p.load(f)
        df = df.interpolate()
        input_features = df.drop(["defects@{false,true}"], axis=1)
        output_class = np.where(df["defects@{false,true}"] == 'true', 1, 0)
        return np.array(input_features), np.array(output_class)

    return   


X, y = get_data('../cm1.arff')
es = BaggingClassifier(base_estimator= Perceptron(max_iter=1000, class_weight = 'balanced'), 
                    n_estimators=100, 
                    max_samples=1.0, 
                    max_features=1.0, 
                    bootstrap=True,
                    bootstrap_features=False, 
                    n_jobs=4)

new_result = lambda : {'accuracy':[], 'roc_auc': [], 'gmean': [], 'f1':[]}
results = new_result()
k_neigh = 7
threshold = 0.4

#20 repetitions
for rep in range(1,6):
    skf = StratifiedKFold(n_splits=4)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        neigh = KNeighborsClassifier(n_neighbors=k_neigh)
        neigh.fit(X_train, y_train) 

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.66)
        pool_classifiers = es.fit(X_train, y_train).estimators_
        knorau = KNORAE(pool_classifiers)
        knorau.fit(X_dsel, y_dsel)        

        y_pred = []
        for instance in X_test:
            #use_des = select_classifier(threshold, k_neigh, X_dsel, y_dsel, instance)
            use_des = True
            if(use_des):
                result = knorau.predict([instance])
                y_pred.append(result[0])
            else:
                result = neigh.predict([instance])
                y_pred.append(result[0])



        for name, metric in zip(['accuracy','roc_auc','gmean','f1'], [accuracy_score, roc_auc_score, gmean, f1_score]): 
            results[name].append(metric(y_test, y_pred))    

print_results(results)