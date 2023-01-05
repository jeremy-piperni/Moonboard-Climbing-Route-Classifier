import os
import sys

import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression

from model_utils import normalize_grades, reshape_x, print_results, RANDOM_STATE

sys.path.append('pre_processing')

from dataset_utils import load_dataset

(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = reshape_x(x_train)
x_test = reshape_x(x_test)

# Normalize the y vector to start from 0
y_train = normalize_grades(y_train)
y_test = normalize_grades(y_test)

clf = LogisticRegression(random_state=RANDOM_STATE, multi_class='multinomial')

param_grid = {
    "C": np.logspace(-2, 2, 4),
    "penalty": ["l1", "l2"],
    "solver": ["newton-cg", "sag", "saga", "lbfgs"]
}
grid = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=3, cv=3).fit(x_train, y_train)

print_results(grid, y_test, x_test)
