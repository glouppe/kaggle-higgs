import random,string,math,csv,pandas
import numpy as np
import matplotlib.pyplot as plt


all = list(csv.reader(open("training.csv","rb"), delimiter=','))

header = np.array(all[0][1:-2])

xs = np.array([map(float, row[1:-2]) for row in all[1:]])
(numPoints,numFeatures) = xs.shape

sSelector = np.array([row[-1] == 's' for row in all[1:]])
bSelector = np.array([row[-1] == 'b' for row in all[1:]])

weights = np.array([float(row[-2]) for row in all[1:]])
labels = np.array([row[-1] for row in all[1:]])
sumWeights = np.sum(weights)
sumSWeights = np.sum(weights[sSelector])
sumBWeights = np.sum(weights[bSelector])


randomPermutation = random.sample(range(len(xs)), len(xs))
np.savetxt("randomPermutation.csv",randomPermutation,fmt='%d',delimiter=',')
#randomPermutation = np.array(map(int,np.array(list(csv.reader(open("randomPermutation.csv","rb"), delimiter=','))).flatten()))

numPointsTrain = int(numPoints*0.9)
numPointsValidation = numPoints - numPointsTrain

xsTrain = xs[randomPermutation[:numPointsTrain]]
xsValidation = xs[randomPermutation[numPointsTrain:]]

sSelectorTrain = sSelector[randomPermutation[:numPointsTrain]]
bSelectorTrain = bSelector[randomPermutation[:numPointsTrain]]
sSelectorValidation = sSelector[randomPermutation[numPointsTrain:]]
bSelectorValidation = bSelector[randomPermutation[numPointsTrain:]]

weightsTrain = weights[randomPermutation[:numPointsTrain]]
weightsValidation = weights[randomPermutation[numPointsTrain:]]

labelsTrain = labels[randomPermutation[:numPointsTrain]]
labelsValidation = labels[randomPermutation[numPointsTrain:]]

sumWeightsTrain = np.sum(weightsTrain)
sumSWeightsTrain = np.sum(weightsTrain[sSelectorTrain])
sumBWeightsTrain = np.sum(weightsTrain[bSelectorTrain])

def AMS(s,b):
    assert s >= 0
    assert b >= 0
    bReg = 10.
    return math.sqrt(2 * ((s + b + bReg) * 
                          math.log(1 + s / (b + bReg)) - s))



from sklearn import ensemble, preprocessing, linear_model

lbl_enc = preprocessing.LabelEncoder()
lbl_enc.fit(labels)
labels_enc = [] #lbl_enc.transform(labelsTrain)

# <codecell>

labels_enc, labelsTrain

# <codecell>

for i in range(len(labels)):
    if labels[i] == 's':
        labels_enc.append(1)
    else:
        labels_enc.append(0)

# <codecell>


# <codecell>

import cPickle

# <codecell>

dataset_blend_train = cPickle.load(open('dataset_blend_train1.pkl', 'rb'))

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

# <codecell>

weightsTrain

# <codecell>

rf = ensemble.RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                    max_features=None, max_leaf_nodes=None, bootstrap=True, 
                                    oob_score=True, n_jobs=1, random_state=None, verbose=2, 
                                    min_density=None, compute_importances=None)

rfr = ensemble.RandomForestRegressor(n_estimators=150, criterion='mse', max_depth=None, 
                               min_samples_split=2, min_samples_leaf=1, 
                               max_features='auto', bootstrap=True, oob_score=False, n_jobs=-1, 
                               random_state=None, verbose=2, min_density=None, compute_importances=None)

#rf.fit(dataset_blend_train[randomPermutation[:numPointsTrain]], np.array(labels_enc)[randomPermutation[:numPointsTrain]])

# <codecell>

#validationScoresText = list(csv.reader(open("scoresValidation.txt","rb"), delimiter=','))
#validationScores = np.array([float(score[0]) for score in validationScoresText])

# <codecell>

rfr.fit(xs, labels_enc)
validationScores = rfr.predict(xs[randomPermutation[numPointsTrain:]])

# <codecell>

#validationScores = (validationScores + validationScores2)

# <codecell>

tIIs = validationScores.argsort()

# <codecell>

tIIs

# <codecell>

wFactor = 1.* numPoints / numPointsValidation

# <codecell>

s = np.sum(weightsValidation[sSelectorValidation])
b = np.sum(weightsValidation[bSelectorValidation])
amss = np.empty([len(tIIs)])
amsMax = 0
threshold = 0.0
for tI in range(len(tIIs)):
    # don't forget to renormalize the weights to the same sum 
    # as in the complete training set
    amss[tI] = AMS(max(0,s * wFactor),max(0,b * wFactor))
    # careful with small regions, they fluctuate a lot
    if tI < 0.9 * len(tIIs) and amss[tI] > amsMax:
        amsMax = amss[tI]
        threshold = validationScores[tIIs[tI]]
        #print tI,threshold, sSelectorValidation[tIIs[tI]]
    if sSelectorValidation[tIIs[tI]]:
        s -= weightsValidation[tIIs[tI]]
    else:
        b -= weightsValidation[tIIs[tI]]

# <codecell>

fig = plt.figure()
fig.suptitle('MultiBoost AMS curves', fontsize=14, fontweight='bold')
vsRank = fig.add_subplot(111)
#fig.subplots_adjust(top=0.85)

vsRank.set_xlabel('rank')
vsRank.set_ylabel('AMS')

vsRank.plot(amss,'b-')

vsRank.axis([0,len(amss), 0, 4])

plt.show()

# <codecell>

fig = plt.figure()
fig.suptitle('MultiBoost AMS curves', fontsize=14, fontweight='bold')
vsScore = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

vsScore.set_xlabel('score')
vsScore.set_ylabel('AMS')

vsScore.plot(validationScores[tIIs],amss,'b-')

vsScore.axis([validationScores[tIIs[0]],validationScores[tIIs[-1]] , 0, 4])

plt.show()

# <codecell>

testText = list(csv.reader(open("test.csv","rb"), delimiter=','))
testIds = np.array([int(row[0]) for row in testText[1:]])
xsTest = np.array([map(float, row[1:]) for row in testText[1:]])
weightsTest = np.repeat(1.0,len(testText)-1)
labelsTest = np.repeat('s',len(testText)-1)
DataToArff(xsTest,labelsTest,weightsTest,header,"HiggsML_challenge_test","test")

# <codecell>

"""
multiboost --configfile configScoresTest.txt

using configScoresTest.txt to output the posterior scores scoresTest.txt on the test set. You can change the effective number of tree used for the test score in

posteriors test.arff shyp.xml scoresTest.txt numIterations
"""

# <codecell>

testScoresText = list(csv.reader(open("scoresTest.txt", "rb"),delimiter=','))
testScores = np.array([float(score[0]) for score in testScoresText])

# <codecell>

dataset_blend_test = cPickle.load(open('dataset_blend_test1.pkl', 'rb'))
testScores = (dataset_blend_test)[:,4]

# <codecell>

#testScores2 = rf.predict(xsTest)

# <codecell>

#testScores = (testScores + testScores2 + testScores3)

# <codecell>

testInversePermutation = testScores.argsort()

# <codecell>

testPermutation = list(testInversePermutation)
for tI,tII in zip(range(len(testInversePermutation)),
                  testInversePermutation):
    testPermutation[tII] = tI

# <codecell>

submission = np.array([[str(testIds[tI]),str(testPermutation[tI]+1),
                       's' if testScores[tI] >= threshold else 'b'] 
            for tI in range(len(testIds))])

# <codecell>

submission = np.append([['EventId','RankOrder','Class']],
                        submission, axis=0)

# <codecell>

np.savetxt("submission9.csv",submission,fmt='%s',delimiter=',')

# <codecell>

cPickle.dump(xs, open('traindata.pkl', 'wb'), -1)
cPickle.dump(xsTest, open('testdata.pkl', 'wb'), -1)
cPickle.dump(labels_enc, open('labels.pkl', 'wb'), -1)

# <codecell>

np.array(labels_enc).shape

# <codecell>


