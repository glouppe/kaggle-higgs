import pandas as pd
import numpy as np
import itertools
from functools import partial

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed

from utils import load_train, load_test
from utils import find_threshold
from utils import rescale, rebalance

# Load training data
X, y, w, _ = load_train()

# Look for the best model
print "Optimize parameters in 5-CV..."


# from sklearn.ensemble import GradientBoostingClassifier
#Classifier = GradientBoostingClassifier
#grid = ParameterGrid({"n_estimators": [500],
#                      "learning_rate": [0.1],
#                      "max_depth": [6],
#                      "max_features": [None],
#                      "min_samples_leaf": [1]})

# from xg import XGBoostClassifier
#Classifier = XGBoostClassifier
#grid = ParameterGrid({"n_estimators": [490],
#                      "eta": [0.1],
#                      "subsample": [1.0],
#                      "max_depth": [6]})

prefix = "bagging-xgb"
from sklearn.ensemble import BaggingClassifier
from xg import XGBoostClassifier
Classifier = partial(BaggingClassifier, base_estimator=XGBoostClassifier(n_estimators=500, eta=0.1, max_depth=6, n_jobs=24))
grid = ParameterGrid({"n_estimators": [20], "n_jobs": [1], "bootstrap": [False], "max_features": [27]})

# from sklearn.ensemble import ExtraTreesClassifier
#Classifier = ExtraTreesClassifier
#grid = ParameterGrid({"n_estimators": [500],
#                      "max_features": [15, 20],
#                      "n_jobs": [12]})


n_jobs = 1

def _parallel_eval(Classifier, params, X, y, w, n_repeat=5, verbose=1):
    if verbose > 0:
        print "[Start]", params

    thresholds, scores, decisions = [], [], []

    for i in range(n_repeat):
        if verbose > 0:
            print "Fold", i

        X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(X, y, w, train_size=0.5, random_state=i)
        X_train = np.asfortranarray(X_train, dtype=np.float32)

        w_train = rescale(w_train)
        w_train = rebalance(y_train, w_train)

        clf = Classifier(**params)
        try:
            clf = clf.fit(X_train, y_train, sample_weight=w_train)
        except:
            clf = clf.fit(X_train, y_train)

        threshold, score, d = find_threshold(clf, X_valid, y_valid, w_valid)

        thresholds.append(threshold)
        scores.append(score)
        decisions.append(d)

    if verbose > 0:
        print "[End]", params, np.mean(thresholds), np.mean(scores)

    return (np.mean(scores), np.mean(thresholds), params, thresholds, scores, decisions)


all_results = Parallel(n_jobs=n_jobs, verbose=3)(
    delayed(_parallel_eval)(
        Classifier,
        p,
        X, y, w)
    for p in grid)

best = max(all_results)
print best

threshold = best[1]
params = best[2]

print "Best average score =", best[0]
print "Average threshold =", threshold
print "Best params =", params

print "Save fold predictions for stacking..."

decisions = best[5]
for i, d in enumerate(decisions):
    np.save("stack/%s-fold%d.npy" % (prefix, i), decisions[i])


# Retrain on the training set
print "Retrain on the full training set..."

clf = Classifier(**params)
w = rescale(w)
w = rebalance(y, w)

try:
    clf.fit(X, y, sample_weight=w)
except:
    clf.fit(X, y)

print "Save test predictions for stacking..."

X_test, _, _, ids = load_test()

try:
    d = -clf.decision_function(X_test)[:, 0]
except:
    d = clf.predict_proba(X_test)[:, 0]

np.save("stack/%s-test.npy" % prefix, d)
