import pandas as pd
import numpy as np
import itertools
import glob
from functools import partial

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.externals.joblib import Parallel, delayed

from utils import load_train, load_test
from utils import find_threshold
from utils import rescale, rebalance
from utils import make_submission

def load_predictions(pattern):
    return np.column_stack([np.load(f) for f in sorted(glob.glob(pattern))])

# Load training data
X, y, w, _ = load_train()

# Tune stacker
print "Optimize parameters in 5-CV..."

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
#Classifier = partial(BaggingClassifier, base_estimator=GradientBoostingClassifier(n_estimators=500, learning_rate=0.025, max_depth=4, max_features=None, min_samples_leaf=250))
#grid = ParameterGrid({"n_estimators": [24], "max_features": [1.0, 0.9, 0.8], "n_jobs": [24]})

Classifier = GradientBoostingClassifier
grid = ParameterGrid({"n_estimators": [500], "max_features": [None, 0.95, 0.9], "learning_rate": [0.0225, 0.025, 0.0275], "max_depth": [4], "min_samples_leaf": [250]})

n_jobs = 24

def _parallel_eval(Classifier, params, X, y, w, n_repeat=5, verbose=1):
    if verbose > 0:
        print "[Start]", params

    thresholds, scores = [], []

    for i in range(n_repeat):
        if verbose > 0:
            print "Fold", i

        _, X_fold, _, y_fold, _, w_fold = train_test_split(X, y, w, train_size=0.5, random_state=i)
        X_pred = load_predictions("stack/*-fold%d.npy" % i)
        X_fold = np.hstack((X_fold, X_pred))

        X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(X_fold, y_fold, w_fold, train_size=0.33, random_state=i)
        X_train = np.asfortranarray(X_train, dtype=np.float32)

        w_train = rescale(w_train)
        w_train = rebalance(y_train, w_train)

        clf = Classifier(**params)
        try:
            clf = clf.fit(X_train, y_train, sample_weight=w_train)
        except:
            clf = clf.fit(X_train, y_train)

        threshold, score, _ = find_threshold(clf, X_valid, y_valid, w_valid)

        thresholds.append(threshold)
        scores.append(score)

    if verbose > 0:
        print "[End]", params, np.mean(thresholds), np.mean(scores)

    return (np.mean(scores), np.mean(thresholds), params, thresholds, scores)

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

# Retrain on the training set
print "Retrain on the full training set..."

all_X, all_y, all_w = [], [], []
for i in range(5):
    _, X_fold, _, y_fold, _, w_fold = train_test_split(X, y, w, train_size=0.5, random_state=i)
    X_pred = load_predictions("stack/*-fold%d.npy" % i)
    X_fold = np.hstack((X_fold, X_pred))

    all_X.append(X_fold)
    all_y.append(y_fold)
    all_w.append(w_fold)

X = np.vstack(all_X)
y = np.concatenate(all_y)
w = np.concatenate(all_w)

clf = Classifier(**params)
w = rescale(w)
w = rebalance(y, w)

try:
    clf.fit(X, y, sample_weight=w)
except:
    clf.fit(X, y)


# And make a submussion
print "Making submission..."
X_test, _, _, _ = load_test()
X_pred = load_predictions("stack/*-test.npy")
X_test = np.hstack((X_test, X_pred))

make_submission(clf, threshold, "output-stacking.csv", X_test=X_test)

import IPython; IPython.embed()

