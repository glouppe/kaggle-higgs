# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import cv2
import pickle
import pandas as pd
import theano
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions import best_params

# <codecell>

image_size = 64
n_samples = 55000
bound = 50

import numpy as np
import cv2
import pickle
import cPickle

import pandas as pd
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.train import Train
from pylearn2.train_extensions import best_params


image_size = 64
n_samples = 55#000
bound = 50
n_neurons= 100
image_type = cv2.CV_LOAD_IMAGE_GRAYSCALE

import pandas as pd
import numpy as np
from sklearn import preprocessing
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
from sklearn.linear_model import LogisticRegression
# <codecell>

from sklearn import ensemble, cross_validation, metrics, grid_search, decomposition,svm
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection, BiasUnit, SoftmaxLayer


print len(range(17000, 23590))
# <codecell>

traindata = cPickle.load(open('traindata.pkl'))
testdata = cPickle.load(open('testdata.pkl'))
labels = cPickle.load(open('labels.pkl'))

std_sclr = preprocessing.StandardScaler()
std_sclr.fit(traindata)
traindata = std_sclr.transform(traindata)
testdata = std_sclr.transform(testdata)

ann = pickle.load(open('saved_clf.pkl', 'rb'))

#rf.fit(X,y)
#preds2 = rf.predict_proba(X_test)[:,1]


#y_preds = []
#for i in range(0,X_test.shape[0],100):
    #print i, ',' ,
y_preds = ann.fprop(theano.shared(testdata, name='inputs')).eval()
    #y_preds.append(y_pred[:,1])

print y_preds[:,1]

preds = y_preds[:,1] > 0.7

print preds 

#preds = np.argmax(y_preds, axis = 1)

pp = []
for i in range(len(preds)):
	if preds[i] == 1:
		pp.append('s')
	else:
		pp.append('b')


subfile = pd.read_csv('submission.csv')

subfile['Class'] = pp
subfile.to_csv('sub.csv', index = False)

#y_preds = np.clip(y_preds, 0, 1)
# #preds = (y_preds[:,1] + preds2)/2.



#sampleSub['Probability1'] = y_preds[:,1]
#sampleSub.to_csv('predictions.csv', index = False)

