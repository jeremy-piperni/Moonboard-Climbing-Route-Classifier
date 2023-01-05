import numpy as np


def save_dataset(file_name, routes, grades):
    dataset_size = len(grades)
    randidx = np.random.randint(dataset_size, size=dataset_size)
    # Training set will be 80% of the dataset and the test set will be the other 20%
    trainidx = randidx[:int(4 * dataset_size / 5)]
    testidx = randidx[int(4 * dataset_size / 5):]

    # X -> routes
    # y -> grades
    X_train = np.take(routes, trainidx, axis=0)
    y_train = np.take(grades, trainidx, axis=0)
    X_test = np.take(routes, testidx, axis=0)
    y_test = np.take(grades, testidx, axis=0)

    print('X_train', len(X_train))
    print('y_train', len(y_train))
    print('X_test', len(X_test))
    print('y_test', len(y_test))

    # Serialize our numpy arrays to a file that way each model can open them and then process it as its needed
    # i.e. into a one hot format or the 2D array to a flattened 1D array format
    np.savez(file_name, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)


def load_dataset(file_name):
    file = np.load(file_name)
    X_train, y_train, X_test,  y_test = file['x_train'], file['y_train'], file['x_test'],  file['y_test']

    # This allows the dataset to be loaded similar to sklearn datasets
    return (X_train, y_train), (X_test, y_test)
