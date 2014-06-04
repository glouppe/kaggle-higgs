import pandas as pd
import numpy as np
import itertools

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.grid_search import ParameterGrid

from utils import load_train
from utils import find_threshold
from utils import rescale, rebalance
from xg import XGBoostClassifier

# Load training data
X, y, w, _ = load_train()

# Look for the best model
#Classifier = GradientBoostingClassifier
#grid = ParameterGrid({"n_estimators": [500],
#                      "learning_rate": [0.1],
#                      "max_depth": [6],
#                      "max_features": [None],
#                      "min_samples_leaf": [1]})

#Classifier = XGBoostClassifier
#grid = ParameterGrid({"n_estimators": [490],
#                      "eta": [0.1],
#                      "subsample": [1.0],
#                      "max_depth": [6]})

Classifier = ExtraTreesClassifier
grid = ParameterGrid({"n_estimators": [500],
                      "max_features": [15, 20]})


from sklearn.externals.joblib import Parallel, delayed
n_jobs = 24

def _parallel_eval(Classifier, params, X, y, w, n_repeat=5, verbose=1):
    if verbose > 0:
        print "[Start]", params

    thresholds, scores = [], []

    for i in range(n_repeat):
        if verbose > 0:
            print i

        X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(X, y, w, train_size=0.5, random_state=i)
        X_train = np.asfortranarray(X_train, dtype=np.float32)

        w_train = rescale(w_train)
        w_train = rebalance(y_train, w_train)

        clf = Classifier(**params)
        clf = clf.fit(X_train, y_train, sample_weight=w_train)
        threshold, score = find_threshold(clf, X_valid, y_valid, w_valid)

        thresholds.append(threshold)
        scores.append(score)

    if verbose > 0:
        print "[End]", params, np.mean(thresholds), np.mean(scores)

    return (np.mean(scores), np.mean(thresholds), params, scores, thresholds)

all_results = Parallel(n_jobs=n_jobs, verbose=3)(
    delayed(_parallel_eval)(
        Classifier,
        p,
        X, y, w)
    for p in grid)

best = max(all_results)
print best

print "Best average score =", best[0]
print "Average threshold =", best[1]
print "Best params =", best[2]
