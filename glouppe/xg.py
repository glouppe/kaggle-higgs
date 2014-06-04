import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

import sys
#sys.path.append("/home/gilles/Sources/xgboost/python/")
sys.path.append("/home/glouppe/src/xgboost/python/")
import xgboost as xgb


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10,
                       max_depth=6,
                       eta=0.1,
                       min_child_weight=1.0,
                       scale_pos_weight=1.0,
                       subsample=1.0,
                       n_jobs=1,
                       missing=-999.0,
                       random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.missing = missing
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        # Check params
        self.n_features_ = X.shape[1]
        random_state = check_random_state(self.random_state)

        params = {}
        params["objective"] = "binary:logitraw"
        params["bst:eta"] = self.eta
        params["bst:min_child_weight"] = self.min_child_weight
        params["bst:subsample"] = self.subsample
        params["scale_pos_weight"] = self.scale_pos_weight
        params['silent'] = 1
        params['nthread'] = self.n_jobs
        params['seed'] = random_state.randint(999999999)

        # Convert data
        self.classes_ = np.unique(y)
        y = np.searchsorted(self.classes_, y)

        if sample_weight is None:
            sample_weight = np.ones(len(X))

        xgmat = xgb.DMatrix(X, label=y, weight=sample_weight, missing=self.missing)
        plst = list(params.items())

        # Run
        self.model_ = xgb.train(plst, xgmat, self.n_estimators, [])

        return self

    def predict(self, X):
        xgmat = xgb.DMatrix(X, missing=self.missing)
        pred = self.model_.predict(xgmat).astype(np.int32)
        return self.classes_[pred]

    def decision_function(self, X):
        xgmat = xgb.DMatrix(X, missing=self.missing)
        pred = self.model_.predict(xgmat)
        pred = pred.reshape((len(X), -1))
        return pred
