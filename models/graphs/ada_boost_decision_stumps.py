import sklearn.model_selection
from sklearn.ensemble import AdaBoostClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


sys.path.append("models")

from model_utils import normalize_grades, reshape_x, RANDOM_STATE

sys.path.append('pre_processing')

from dataset_utils import load_dataset

(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = reshape_x(x_train)

# Normalize the y vector to start from 0
y_train = normalize_grades(y_train)

adaBoo = sklearn.ensemble.AdaBoostClassifier(random_state=RANDOM_STATE, n_estimators=100, learning_rate=0.6)
adaBoo = adaBoo.fit(x_train,y_train)


plt.figure()
for i in range(0, 100, 20):
    plt.subplot(5, 1, i//20 + 1)
    sklearn.tree.plot_tree(adaBoo.estimators_[i])
    plt.title('AdaBoost estimator ' + str(i + 1))
    i += 1

plt.show()