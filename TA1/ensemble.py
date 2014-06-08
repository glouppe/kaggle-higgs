# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import pandas as pd
from sklearn import preprocessing, linear_model, naive_bayes, ensemble, neural_network, svm, decomposition, cross_validation, isotonic, tree

# <codecell>

all = list(csv.reader(open("training.csv","rb"), delimiter=','))

# <codecell>

header = np.array(all[0][1:-2])

# <codecell>

xs =dataset_blend_train = np.array([map(float, row[1:-2]) for row in all[1:]])

(numPoints,numFeatures) = xs.shape
print xs.shape

# <codecell>

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])

# <codecell>

weights = np.array([float(row[-2]) for row in all[1:]])
labels = np.array([row[-1] for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])

# <codecell>

testText = list(csv.reader(open("test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)

# <codecell>

### ensemble:

# <codecell>

clfs = [ensemble.RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),

        ensemble.RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=None, min_samples_split=2, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),
        
        ensemble.RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=25, min_samples_split=1, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),
        
        ensemble.RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=25, min_samples_split=1, 
                                     min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),
        
        ensemble.RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_split=1, 
                                     min_samples_leaf=1, max_features=None, max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),
        
        ensemble.RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=None, min_samples_split=1, 
                                     min_samples_leaf=1, max_features=None, max_leaf_nodes=None, bootstrap=True,
                                     oob_score=False, n_jobs=8, random_state=None, verbose=2, min_density=None, 
                                     compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_split=2, 
                                      min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='entropy', max_depth=None, min_samples_split=2, 
                                      min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='gini', max_depth=25, min_samples_split=1, 
                                      min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='entropy', max_depth=25, min_samples_split=1, 
                                      min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='gini', max_depth=None, min_samples_split=2, 
                                      min_samples_leaf=1, max_features=None, max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),
        
        ensemble.ExtraTreesClassifier(n_estimators=150, criterion='entropy', max_depth=None, min_samples_split=2, 
                                      min_samples_leaf=1, max_features=None, max_leaf_nodes=None, bootstrap=False, 
                                      oob_score=False, n_jobs=8, random_state=None, verbose=0, min_density=None, 
                                      compute_importances=None),       
        
        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),
        
        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight=None, random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),
        
        linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),
        
        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=10.0, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),

        linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=0.1, fit_intercept=True, intercept_scaling=1, 
                                        class_weight='auto', random_state=None),
    
        naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None),
        
        naive_bayes.BernoulliNB(alpha=0.1, binarize=0.0, fit_prior=True, class_prior=None),
        
        naive_bayes.BernoulliNB(alpha=10.0, binarize=0.0, fit_prior=True, class_prior=None),
        
        linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=0.1, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=10.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=1.0, fit_intercept=False, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=0.1, fit_intercept=False, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=10.0, fit_intercept=False, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=1.0, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=0.1, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=10.0, fit_intercept=True, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        
        linear_model.Ridge(alpha=1.0, fit_intercept=False, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=0.1, fit_intercept=False, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        linear_model.Ridge(alpha=10.0, fit_intercept=False, normalize=True, copy_X=True, max_iter=None, tol=0.001, solver='auto'),
        
        
        linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='hinge', penalty='l1', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='log', penalty='l1', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='modified_huber', penalty='l1', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),
        
        linear_model.SGDClassifier(loss='modified_huber', penalty='elasticnet', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=100, 
                                   shuffle=False, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, 
                                   learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False),

        
        linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=0.1, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=10.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=100.0, fit_intercept=True, normalize=False, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        
        linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=0.1, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=10.0, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=100.0, fit_intercept=True, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        
        linear_model.Lasso(alpha=1.0, fit_intercept=False, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=0.1, fit_intercept=False, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=10.0, fit_intercept=False, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),
        
        linear_model.Lasso(alpha=100.0, fit_intercept=False, normalize=True, precompute='auto', copy_X=True, max_iter=1000, 
                           tol=0.0001, warm_start=False, positive=False),   
        
        
        linear_model.Lars(fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, 
                          eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.Lars(fit_intercept=False, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, 
                          eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.Lars(fit_intercept=True, verbose=False, normalize=False, precompute='auto', n_nonzero_coefs=500, 
                          eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.Lars(fit_intercept=False, verbose=False, normalize=False, precompute='auto', n_nonzero_coefs=500, 
                          eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.LassoLars(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, 
                               eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.LassoLars(alpha=0.1, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, 
                               eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.LassoLars(alpha=10.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, 
                               eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        linear_model.LassoLars(alpha=100.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, 
                               eps=2.2204460492503131e-16, copy_X=True, fit_path=True),
        
        tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                    max_features=None, random_state=None, min_density=None, compute_importances=None, 
                                    max_leaf_nodes=None),
        
        tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                    max_features=None, random_state=None, min_density=None, compute_importances=None, 
                                    max_leaf_nodes=None)
        
        ]

# <codecell>

len(clfs)

# <codecell>

n_folds = 5
verbose = True
shuffle = False

# <codecell>

label = labels == 's'

# <codecell>

Xlr = xs
Xlr_test = xsTest
ylr = np.array(label)

# <codecell>

skf = list(cross_validation.StratifiedKFold(ylr, n_folds))

# <codecell>

print "Creating train and test sets for blending."
dataset_blend_train = np.zeros((Xlr.shape[0], len(clfs)))
dataset_blend_test = np.zeros((Xlr_test.shape[0], len(clfs)))

# <codecell>

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((Xlr_test.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i            

        xtrain = Xlr[train]
        xtest = Xlr[test]
        y_train = ylr[train] 

        xsub = Xlr_test

        clf.fit(xtrain,y_train)    
        try: 
            dataset_blend_test_j[:, i] = clf.predict_proba(xsub)[:,1]
            y_submission = clf.predict_proba(xtest)[:,1]
        except:
            dataset_blend_test_j[:, i] = clf.predict(xsub)
            y_submission = clf.predict(xtest)

        dataset_blend_train[test, j] = y_submission

    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    cPickle.dump(dataset_blend_train, open('dataset_blend_train_temp.pkl', 'wb'), -1)
    cPickle.dump(dataset_blend_test, open('dataset_blend_test_temp.pkl', 'wb'), -1)

# <codecell>#
cPickle.dump(dataset_blend_train, open('dataset_blend_train_76.pkl', 'wb'), -1)
cPickle.dump(dataset_blend_test, open('dataset_blend_test_76.pkl', 'wb'), -1)

