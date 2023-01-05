import os
import sys

import numpy as np
import sklearn.model_selection
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Input, Dense, Flatten, concatenate
from keras.models import Model

from model_utils import print_results

sys.path.append('pre_processing')

from dataset_utils import load_dataset
(x_train, y_train), (x_test, y_test) = load_dataset(os.path.join("pre_processing", "dataset.npz"))

x_train = x_train.reshape((-1, 18, 11, 1))
x_test = x_test.reshape((-1, 18, 11, 1))

# to one-hot rep
max_value = np.max(np.unique(y_test)) + 1
y_train = np.eye(max_value)[y_train]
y_test = np.eye(max_value)[y_test]

batch_size = 4
inputs = Input(shape=(18, 11, 1))

x = Flatten()(inputs)
x = Dense(units=(18 * 11), activation='relu')(x)
x = Dense(units=32, activation='relu')(x)
outputs = Dense(units=max_value, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=20)
