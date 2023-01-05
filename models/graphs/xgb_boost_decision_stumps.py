import sklearn.model_selection
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import xgboost as xgb


sys.path.append("models")

from model_utils import normalize_grades, reshape_x, RANDOM_STATE

sys.path.append('pre_processing')

from dataset_utils import load_dataset

(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = reshape_x(x_train)

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
xgb_cl = xgb_cl.fit(x_train,y_train)

xgb.plot_tree(xgb_cl)
plt.title("XGBoost tree")

plt.show()