import pandas as pd
import numpy as np
import itertools

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.grid_search import ParameterGrid

from utils import load_train
from utils import find_threshold
from utils import rescale, rebalance
from xg import XGBoostClassifier

# Load training data
X, y, w, _ = load_train()

# Todo
# - train stacker from the fold predictions
# - adjust threshold
# - retrain and predict
