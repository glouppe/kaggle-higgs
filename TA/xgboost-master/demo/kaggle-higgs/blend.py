import pandas as pd
import numpy as np
import scipy as sp
import sklearn.linear_model as lm
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
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd
import numpy as np
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
import statsmodels.api as sm
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble, cross_validation, metrics
import numpy as np
import scipy as sp
import sklearn.linear_model as lm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from functools import partial
from sklearn.metrics import roc_curve, auc
import scipy as sp
from sklearn import preprocessing
import numpy as np
from sklearn import ensemble, cross_validation, metrics, grid_search, decomposition,svm, tree

n_folds = 5
verbose = True
shuffle = False

##########
##########  DECLARE ALL CLASSIFIERS / REGRESSORS HERE
##########
rf = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=-1, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None)
gbm1 = ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
	min_samples_split=2, min_samples_leaf=1, max_depth=3, init=None, random_state=None, max_features=None, verbose=1)

rfr = ensemble.RandomForestRegressor(n_estimators=50, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
	max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=2, min_density=None, compute_importances=None)

nb1 = BernoulliNB()

nb2 = MultinomialNB()

lr = LogisticRegression()

sgd = SGDClassifier()

sgdr = SGDRegressor()

sv = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, 
        probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=0, max_iter=-1, random_state=None)

clfs = [ 
		rf,
		nb1,
        sgd,
		]

##############
##############  CLASSIFIERS END HERE
##############

import inspect
import os
import sys
import numpy as np
# add path of xgboost python module
code_path = os.path.join(
    os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../python")

sys.path.append(code_path)

import xgboost as xgb

test_size = 550000

# path to where the data lies
dpath = '../../../'

# load in training data, directly use numpy
dtrain = np.loadtxt( dpath+'training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s'.encode('utf-8')) } )
print ('finish loading from csv ')

label  = dtrain[:,32]
traindata   = dtrain[:,1:31]
# rescale weight to make it same as test set
weight = dtrain[:,31] * float(test_size) / len(label)

sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )

# print weight statistics 
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

dtest = np.loadtxt( dpath+'test.csv', delimiter=',', skiprows=1 )
testdata   = dtest[:,1:31]
idx = dtest[:,0]

Xlr = traindata
Xlr_test = testdata
ylr = np.array(label)

skf = list(StratifiedKFold(ylr, n_folds))

print "Creating train and test sets for blending."
dataset_blend_train = np.zeros((Xlr.shape[0], len(clfs)))
dataset_blend_test = np.zeros((Xlr_test.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((Xlr_test.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i            

        xtrain = Xlr[train]
        xtest = Xlr[test]
        y_train = ylr[train] 
        w_train = weight[train]
        xsub = Xlr_test
        clf.fit(xtrain,y_train, sample_weight = w_train)    
        try: 
            dataset_blend_test_j[:, i] = clf.predict_proba(xsub)[:,1]
            y_submission = clf.predict_proba(xtest)[:,1]
        except:
            dataset_blend_test_j[:, i] = clf.predict(xsub)
            y_submission = clf.predict(xtest)

        dataset_blend_train[test, j] = y_submission

    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)


cPickle.dump(dataset_blend_train, open('dataset_blend_train.pkl', 'wb'), -1)
cPickle.dump(dataset_blend_test, open('dataset_blend_test.pkl', 'wb'), -1)



