import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append("models")
from model_utils import normalize_grades, reshape_x, RANDOM_STATE

sys.path.append('pre_processing')
from dataset_utils import load_dataset

(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

y_labels = [18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
x_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def plot_heatMap(r, t):
    plt.figure()
    plt.imshow(r)
    plt.yticks(np.arange(0, 18), y_labels)
    plt.xticks(np.arange(0, 11), x_labels)
    plt.colorbar()
    plt.title(t)


data = x_train[0]
for i in range(1, 300):
    data += x_train[i]

# setting the max value to 1 and all the other relative to that
data = data / data.max()

plot_heatMap(data, "Relative percentage of holds used")

plt.show()
