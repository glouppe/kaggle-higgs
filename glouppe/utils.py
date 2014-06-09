import pandas as pd
import numpy as np

from sklearn.utils import safe_asarray, check_random_state

# Loaders

def load_train(filename="data/training.csv"):
    data = pd.read_csv(filename)
    X = data.values[:, 1:-2]

    y = data.Label.values
    y_ = np.zeros(len(X))
    y_[y == 's'] = 1.0
    y = y_

    sample_weight = data.Weight.values
    ids = data.EventId

    return X, y, sample_weight, ids

def load_test(filename="data/test.csv"):
    data = pd.read_csv(filename)
    X = data.values[:, 1:]
    y = None
    sample_weight = np.ones(len(X))
    ids = data.EventId

    return X, y, sample_weight, ids


# Rescale weights

def rescale(w, weight_sum=411691.83592984255):
    w = w.copy()
    w *= weight_sum / np.sum(w)

    return w

def rebalance(y, w):
    w = w.copy()
    w_pos = np.sum(w[y == 1])
    w_neg = np.sum(w[y == 0])
    w[y == 1] *= w_neg / w_pos

    return w


# Tune threshold

def ams(s, b, br=10):
    b += br
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))

def find_threshold(clf, X, y, sample_weight):
    d = np.zeros(len(X))

    if not isinstance(clf, list):
        clf = [clf]

    for c in clf:
        try:
            d += -c.decision_function(X)[:, 0]
        except:
            d += c.predict_proba(X)[:, 0]

    d /= len(clf)

    sample_weight = rescale(sample_weight)
    best_score = -np.inf
    best_threshold = 0
    best_weight = 0.0

    indices = np.argsort(d)
    s = 0.0
    b = 0.0

    a = []

    for i, j in zip(indices[:-1], indices[1:]):
        if y[i] == 1.0:
            s += sample_weight[i]
        else:
            b += sample_weight[i]

        score = ams(s, b)

        if score > best_score:
            threshold = (d[i] + d[j]) / 2.0
            best_score = score
            best_threshold = threshold
            best_weight = s + b

    return best_threshold, best_score, best_weight


# Submit

def make_submission(clf, threshold, output):
    X_test, _, _, ids = load_test()
    d = np.zeros(len(X_test))

    if not isinstance(clf, list):
        clf = [clf]

    for c in clf:
        try:
            d += -c.decision_function(X_test)[:, 0]
        except:
            d += c.predict_proba(X_test)[:, 0]

    d /= len(clf)

    r = np.argsort(-d) + 1
    p = np.empty(len(X_test), dtype=np.object)
    mask = (d <= threshold)
    p[mask] = 's'
    p[~mask] = 'b'

    df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
    df.to_csv(output, index=False, cols=["EventId", "RankOrder", "Class"])
