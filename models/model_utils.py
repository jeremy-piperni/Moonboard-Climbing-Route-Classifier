import csv
import pandas as pd
import os
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

# Use a single random state value everywhere in the code for reproducibility
RANDOM_STATE = 42


def normalize_grades(y, min_y_val=3):
    # Since we don't work with climbing grades under V3 we subtract all the class
    # labels by 3 to normalize them at 0, this is the expected form of number class
    # for sklearn models, since we do this for both the train and test data no
    # other work needs to be done during validation to account for this
    return y - min_y_val


def reshape_x(x):
    # We store the climbs as 2D numpy arrays but the sklearn models take the input
    # as 1D array of features, so we reshape the array and scale the values to all
    # be between 0 and 1, this is because the start and end holds have values outside
    # that range
    x = x.reshape(len(x), 18 * 11)

    scaler = MinMaxScaler()

    x_temp = x.reshape([-1, 1])
    x_temp = scaler.fit_transform(x_temp)
    return x_temp.reshape(x.shape)


def accuracy_score_with_tolerance(y_test, preds):
    assert len(y_test) == len(preds), "labeled test data has a different length than the predictions"

    # Calculate the accuracy of the predictions but with a tolerance of +-1 climbing grade
    # this is done to give a better idea of 1 off errors in the grading since the classes
    # we are classifying with are sequential in nature it makes sense for some of the label
    # to be "next" to each other if they aren't correct
    correct_preds = 0
    for y, pred in zip(y_test, preds):
        if abs(y - pred) <= 1:
            correct_preds += 1

    return correct_preds / len(preds)


def print_results(grid, y_test, x_test):
    print("\n\n")
    print("=== RESULTS ===")

    print("Best parameters: {}".format(grid.best_params_))
    print("Best score: {:0.5f}".format(grid.best_score_))
    print("All model parameters: {}".format(grid.get_params()))

    preds = grid.best_estimator_.predict(x_test)
    acc_score = accuracy_score(y_test, preds)
    print(f"Accuracy on test data: {acc_score * 100} %")
    acc_score_tolerance = accuracy_score_with_tolerance(y_test, preds)
    print(f"Accuracy on test data with +- 1 grade tolerance: {acc_score_tolerance * 100} %")

    print(classification_report(y_test, preds))

    # For each model we will build a data frame of all the different combinations
    # of hyperparameters from grid search and save them to a CSV along with the
    # test data accuracy with and without tolerance scores for analysis later
    model_name = grid.get_params()["estimator"].__class__.__name__
    df = pd.concat([
        pd.DataFrame([{"Model Name": model_name}]),
        pd.DataFrame(grid.cv_results_["params"]),
        pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"]),
    ], axis=1)
    path = os.path.join("results", f"{model_name}.csv")
    df.to_csv(path, encoding='utf-8', index=False)

    df = pd.concat([
        pd.DataFrame([{
            "Accuracy": acc_score,
            "Accuracy With Tolerance": acc_score_tolerance,
            "Precision": precision_score(y_test, preds, average="macro"),
            "Recall": recall_score(y_test, preds, average="macro"),
            "F1": f1_score(y_test, preds, average="macro"),
        }]),
    ])
    df.to_csv(path, encoding='utf-8', index=False, mode='a')
