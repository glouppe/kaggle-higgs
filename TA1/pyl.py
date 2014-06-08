import numpy as np
import cv2
import pickle
import cPickle

import pandas as pd
from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models import mlp
from pylearn2.costs.mlp.dropout import Dropout
from pylearn2.training_algorithms import sgd, learning_rule,bgd
from pylearn2.termination_criteria import EpochCounter, MonitorBased
from pylearn2.train import Train
from pylearn2.train_extensions import best_params, roc_auc
import pandas as pd
import numpy as np
from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
from sklearn.linear_model import LogisticRegression


# <codecell>

traindata = cPickle.load(open('traindata.pkl', 'rb'))
testdata = cPickle.load(open('testdata.pkl', 'rb'))
labels = cPickle.load(open('labels.pkl', 'rb'))

# <codecell>

#traindata = traindata.todense()
#testdata = testdata.todense()
# <codecell>

from sklearn import ensemble, cross_validation, metrics, grid_search, decomposition,svm
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection, BiasUnit, SoftmaxLayer
from pylearn2.datasets import preprocessing

import sklearn.preprocessing as prep

image_size = 64
n_samples = 55#000
bound = 50
n_neurons= 100

# <codecell>
#pca = prep
#print "PCA..."
scl = prep.StandardScaler()
traindata = scl.fit_transform(traindata)

X = np.array(traindata)
#pf = preprocessing.PolynomialFeatures(degree=2)
#print "pf..."
#X = pf.fit_transform(X)
print X.shape
#X = X - X.mean(axis=0)
onehot = prep.OneHotEncoder()
y = np.array(labels)
print y.shape

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

print onehot.fit_transform(np.reshape(y_train,(-1,1))).todense()

ds = DenseDesignMatrix(X=X_train, y=onehot.fit_transform(np.reshape(y_train,(-1,1))).todense())
ds_test = DenseDesignMatrix(X=X_test, y=onehot.fit_transform(np.reshape(y_test,(-1,1))).todense())

#preprocessor = preprocessing.ZCA()
#ds.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
#ds_test.apply_preprocessor(preprocessor = preprocessor, can_fit = True)


print X_train.shape, X_test.shape

l1 = mlp.RectifiedLinear(layer_name='l1',
                         #sparse_init=12,
                         irange=0.1,
                         dim=300,
                         #max_col_norm=1.
)

l2 = mlp.RectifiedLinear(layer_name='l2',
                         #sparse_init=12,
                         irange=0.01,
                         dim=300,
                         #max_col_norm=1.
)

l3 = mlp.RectifiedLinear(layer_name='l3',
                         #sparse_init=12,
                         irange=0.01,
                         dim=300,
                         #max_col_norm=1.
)

l4 = mlp.RectifiedLinear(layer_name='l4',
                         #sparse_init=12,
                         irange=0.01,
                         dim=300,
                         #max_col_norm=1.
)

l5 = mlp.RectifiedLinear(layer_name='l5',
                         #sparse_init=12,
                         irange=0.01,
                         dim=300,
                         #max_col_norm=1.
)

l6 = mlp.RectifiedLinear(layer_name='l6',
                         #sparse_init=12,
                         irange=0.01,
                         dim=300,
                         #max_col_norm=1.
)


output = mlp.Softmax(n_classes=2, layer_name='y', irange=.01)


#output = mlp.HingeLoss(layer_name='y',n_classes=2,irange=.05)

#layers = [l5, l6, output]
layers = [l1, l2, l3, l4, l5, output]

ann = mlp.MLP(layers, nvis=X[0].reshape(-1).shape[0])

lr = 0.1
epochs = 400
trainer = sgd.SGD(learning_rate=lr,
                  batch_size=100,
                  learning_rule=learning_rule.Momentum(.05),
                  # Remember, default dropout is .5
                  #cost=Dropout(input_include_probs={'l1': .5},
                  #             input_scales={'l1': 1.}),
                  termination_criterion=EpochCounter(epochs),
                  monitoring_dataset={'train': ds, 'valid':ds_test})

watcher = best_params.MonitorBasedSaveBest(
    channel_name='valid_roc_auc',
    save_path='saved_clf.pkl')

velocity = learning_rule.MomentumAdjustor(final_momentum=.9,
                                          start=1,
                                          saturate=250)

decay = sgd.LinearDecayOverEpoch(start=1,
                                 saturate=250,
                                 decay_factor=lr*.05)
rocauc = roc_auc.RocAucChannel()
experiment = Train(dataset=ds,
                   model=ann,
                   algorithm=trainer,
                   extensions=[watcher, velocity, decay, rocauc])

experiment.main_loop()
