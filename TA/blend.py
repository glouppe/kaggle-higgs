# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn import preprocessing
import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn import metrics,preprocessing,cross_validation
from sklearn.cross_validation import StratifiedKFold
from scipy.optimize.optimize import fmin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import cPickle
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import scipy.io
import re
from collections import Counter
from sklearn.naive_bayes import BernoulliNB

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
import statsmodels.api as sm

from itertools import combinations
from sklearn.linear_model import LogisticRegression

# <codecell>

def group_data(data, degree=3, hash=hash):
    """ 
    numpy.array -> numpy.array
    
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T

from sklearn import ensemble, cross_validation, metrics
import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn import metrics,preprocessing,cross_validation
from sklearn.cross_validation import StratifiedKFold
from scipy.optimize.optimize import fmin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
import cPickle
from sklearn.decomposition import RandomizedPCA
import pandas as pd
import scipy.io
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from functools import partial
from sklearn.metrics import roc_curve, auc
import scipy as sp
from sklearn import preprocessing
import numpy as np


class AUCRegressor(object):
    def __init__(self):
        self.coef_ = 0

    def _auc_loss(self, coef, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, coef))
        return -auc(fpr, tpr)

    def fit(self, X, y):
        lr = LinearRegression()
        auc_partial = partial(self._auc_loss, X=X, y=y)
        initial_coef = lr.fit(X, y).coef_
        self.coef_ = sp.optimize.fmin(auc_partial, initial_coef)

    def predict(self, X):
        return sp.dot(X, self.coef_)

    def score(self, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, self.coef_))
        return auc(fpr, tpr)


class MLR(object):
    def __init__(self):
        self.coef_ = 0

    def fit(self, X, y):
        self.coef_ = sp.optimize.nnls(X, y)[0]
        self.coef_ = np.array(map(lambda x: x/sum(self.coef_), self.coef_))

    def predict(self, X):
        predictions = np.array(map(sum, self.coef_ * X))
        return predictions

    def score(self, X, y):
        fpr, tpr, _ = roc_curve(y, sp.dot(X, self.coef_))
        return auc(fpr, tpr)

n_folds = 5
verbose = True
shuffle = False

traindata = cPickle.load(open('traindata.pkl', 'rb'))
testdata = cPickle.load(open('testdata.pkl', 'rb'))
labels = cPickle.load(open('labels.pkl', 'rb'))

Xlr = traindata
Xlr_test = testdata
ylr = np.array(labels)

skf = list(StratifiedKFold(ylr, n_folds))

print traindata.shape, testdata.shape

aucR = AUCRegressor()
mlr = MLR()
rf = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=-1, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None)
gbm1 = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
	min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=1)

gbm2 = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, 
	min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0)

gbm3 = ensemble.GradientBoostingRegressor(loss='lad', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, 
	min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0)

rfr = ensemble.RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=2, min_density=None, compute_importances=None)

nb = BernoulliNB()


from sklearn import ensemble, cross_validation, metrics, grid_search, decomposition,svm, tree
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection, BiasUnit, SoftmaxLayer

lr = LogisticRegression()


abc = ensemble.AdaBoostClassifier(base_estimator=rf, n_estimators=10, learning_rate=1.0, random_state=None)

sv = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, 
        probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=0, max_iter=-1, random_state=None)

### Hide Classifiers from below to use only a few selected ones. The Default system uses all classifiers
### and Regressors available in model.classifiers module.
# clfs = [ 
# 		aucR,
# 		mlr,
#         lm.Ridge(),
# 		rf,
# 		rfr,
# 		gbm1,
# 		gbm2,
# 		gbm3,
# 		nb,
#         #abc,
#         #sv
# 		]

# print "Creating train and test sets for blending."
# dataset_blend_train = np.zeros((Xlr.shape[0], len(clfs)))
# dataset_blend_test = np.zeros((Xlr_test.shape[0], len(clfs)))

# for j, clf in enumerate(clfs):
#     print j, clf
#     dataset_blend_test_j = np.zeros((Xlr_test.shape[0], len(skf)))
#     for i, (train, test) in enumerate(skf):
#         print "Fold", i            

#         xtrain = Xlr[train]
#         xtest = Xlr[test]
#         y_train = ylr[train] 
#         xsub = Xlr_test
#         clf.fit(xtrain,y_train)    
#         try: 
#             dataset_blend_test_j[:, i] = clf.predict_proba(xsub)[:,1]
#             y_submission = clf.predict_proba(xtest)[:,1]
#         except:
#             dataset_blend_test_j[:, i] = clf.predict(xsub)
#             y_submission = clf.predict(xtest)

#         dataset_blend_train[test, j] = y_submission

#     dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)


# cPickle.dump(dataset_blend_train, open('dataset_blend_train1.pkl', 'wb'), -1)
# cPickle.dump(dataset_blend_test, open('dataset_blend_test1.pkl', 'wb'), -1)
# #clf = 


clf = LogisticRegression(C=1.0)


dataset_blend_train = cPickle.load(open('dataset_blend_train1.pkl', 'rb'))
dataset_blend_test = cPickle.load(open('dataset_blend_test1.pkl', 'rb'))

print dataset_blend_test
clf.fit(traindata, labels)
#preds = np.mean(dataset_blend_test, axis=1)
preds = clf.predict(testdata)#[:,1]

pp = []
for i in range(len(preds)):
    if preds[i] == 1:
        pp.append('s')
    else:
        pp.append('b')

sampleSub = pd.read_csv('submission.csv')

sampleSub['Class'] = pp
sampleSub.to_csv('predictions_blend.csv', index = False)


niter = 20
cv = cross_validation.KFold(n = dataset_blend_train.shape[0], n_folds=niter) #, test_size=0.2, random_state=rnd)
auc_list = []
mean_auc = 0.0; itr = 0       
for train, test in cv:
    xtrain = dataset_blend_train[train]
    xtest = dataset_blend_train[test]
    ytrain = ylr[train]
    ytest = ylr[test]       

    dataset = xtrain
    ytrue = ytrain
    #xopt = fmin(fopt, x0)
    #preds = fopt_pred(xopt, xtest)
    #
    #preds = xtest[:,-3]#np.mean(xtest[:,[0,-3]], axis=1)
    #clf.fit(xtrain,ytrue)
    # try:
    #preds = clf.predict_proba(xtest)[:,1]
    # except:
    preds = np.sum(xtest, axis = 1)#[:,1]

    #fpr, tpr, _ = metrics.roc_curve(ytest, preds)
    roc_auc = metrics.roc_auc_score(ytest, preds)
    auc_list.append(roc_auc)
    print "AUC (fold %d/%d): %f" % (itr + 1, niter, roc_auc)
    mean_auc += roc_auc ; itr += 1
print "Mean AUC: ", mean_auc/niter    
print "Min AUC: ", min(auc_list)
print "Max AUC:", max(auc_list)
  


