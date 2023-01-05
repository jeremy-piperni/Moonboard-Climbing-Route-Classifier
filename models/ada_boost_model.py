import os
import sys

import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier

from model_utils import normalize_grades, reshape_x, RANDOM_STATE, print_results

sys.path.append('pre_processing')

from dataset_utils import load_dataset

(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = reshape_x(x_train)
x_test = reshape_x(x_test)

# Normalize the y vector to start from 0
y_train = normalize_grades(y_train)
y_test = normalize_grades(y_test)

clf = AdaBoostClassifier(random_state=RANDOM_STATE)

param_grid = {
    'learning_rate': [0.6, 0.8, 1.0],
    'n_estimators': [10, 100],
}
grid = sklearn.model_selection.GridSearchCV(clf, param_grid, verbose=3, cv=3).fit(x_train, y_train)

print_results(grid, y_test, x_test)
