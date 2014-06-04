import gc
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from utils import make_submission
from utils import rescale, rebalance
from xg import XGBoostClassifier

from sklearn.externals.joblib import Parallel, delayed
n_jobs = 24

# Load training data
X, y, w, _ = load_train()

# Best params
#Classifier = GradientBoostingClassifier
#params = {"n_estimators": 200,
#          "learning_rate": 0.1,
#          "max_depth": 6,
#          "subsample": 0.9,
#          "max_features": 20,
#          "min_samples_leaf": 44}

Classifier = XGBoostClassifier
params = {"n_estimators": 490,
          "eta": 0.1,
          "max_depth": 6,
          "scale_pos_weight": 1.0,
          "subsample": 1.0}

# Train on the whole training set
def _parallel_train(Classifier, params, X, y, w, i, verbose=1):
    if verbose > 0:
        print "[Start]", i

    w = rescale(w)
    w = rebalance(y, w)

    clf = Classifier(**params)

    try:
        clf.set_params(random_state=i)
    except:
        pass

    clf.fit(X, y, sample_weight=w)

    if verbose > 0:
        print "[End]", i

    return clf

all_clf = []
n_models = 20 # Average several models to reduce variance

for i in range(n_models):
    all_clf.append(_parallel_train(Classifier, params, X, y, w, i))

#all_clf = Parallel(n_jobs=n_jobs, verbose=3)(
#    delayed(_parallel_train)(
#        Classifier,
#        params,
#        X,
#        y,
#        w,
#        i)
#    for i in range(n_models))

X = None
y = None
gc.collect()

# Make submission
#make_submission(all_clf, -2.825, "output-2.825.csv")

import IPython; IPython.embed()
