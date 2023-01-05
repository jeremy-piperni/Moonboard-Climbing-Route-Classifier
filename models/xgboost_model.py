import os
import sys

import numpy as np
import sklearn.model_selection
import xgboost as xgb

from model_utils import normalize_grades, reshape_x, print_results, RANDOM_STATE

sys.path.append('pre_processing')

from dataset_utils import load_dataset
(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = reshape_x(x_train)
x_test = reshape_x(x_test)

# Normalize the y vector to start from 0
y_train = normalize_grades(y_train)
y_test = normalize_grades(y_test)

max_value = len(np.unique(y_test))

xgb_cl = xgb.XGBClassifier(
    seed=RANDOM_STATE,
    random_state=RANDOM_STATE,
    objective='multi:softmax',
    num_class=max_value
)

param_grid = {
    'learning_rate': [0.01, 0.02, 0.1],
    'max_depth': [5, 8, 15],
    'n_estimators': [100, 500, 1000],
}
grid = sklearn.model_selection.GridSearchCV(xgb_cl, param_grid, verbose=3, cv=3).fit(x_train, y_train)

print_results(grid, y_test, x_test)
