import gc
import pandas as pd
import numpy as np

#from sklearn.ensemble import GradientBoostingClassifier

from utils import load_train, load_blend_train
from utils import make_submission
from utils import rescale, rebalance
from xg import XGBoostClassifier

from gbm_sw import GradientBoostingClassifier

# Load training data
X, y, w, _ = load_blend_train()

Classifier = GradientBoostingClassifier
#Classifier = partial(BaggingClassifier, base_estimator=GradientBoostingClassifier(n_estimators = 100, max_depth = 3, verbose = 2))
params = {"n_estimators": 200,
                       "learning_rate": 0.05,
                       "max_depth": 3,
                       "max_features": 20,
                       "min_samples_leaf": 50,
                       "verbose": 2}

# Best params
#Classifier = GradientBoostingClassifier
#params = {"n_estimators": 200,
#          "learning_rate": 0.1,
#          "max_depth": 6,
#          "subsample": 0.9,
#          "max_features": 20,
#          "min_samples_leaf": 44}

#Classifier = XGBoostClassifier
#params = {"n_estimators": 490,
#          "eta": 0.1,
#          "max_depth": 6,
#          "scale_pos_weight": 1.0,
#          "subsample": 1.0}

# from functools import partial
# from sklearn.ensemble import BaggingClassifier
# Classifier = partial(BaggingClassifier, base_estimator=XGBoostClassifier(n_estimators=490, eta=0.1, max_depth=6, n_jobs=24))
# params = {"n_estimators": 10, "n_jobs": 1, "bootstrap": False, "max_features": 28}


# Train on the whole training set
def train(Classifier, params, X, y, w, verbose=1):
    if verbose > 0:
        print "[Start]"

    w = rescale(w)
    w = rebalance(y, w)

    clf = Classifier(**params)
    clf.fit(X, y, sample_weight=w)

    if verbose > 0:
        print "[End]"

    return clf

clf = train(Classifier, params, X, y, w)

# Make submission
threshold = -2.86253711576
make_submission(clf, threshold, "output-gbm-2.86253711576.csv")

import IPython; IPython.embed()
