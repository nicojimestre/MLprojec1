import json
import numpy as np

from helper import load_csv_data, create_csv_submission
from processing import clean_data, build_poly
from implementations import ridge_regression, get_classification_pred
from metrics import f1_score


if __name__ == "__main__":
    # import train data
    print("importing train and test data...")
    y_train, x_train, id_train = load_csv_data("../MLprojec1/data/train.csv")
    y_test, x_test, id_test = load_csv_data("../MLprojec1/data/train.csv")

    # import feature names
    with open("col_name.json", "r") as handle:
        features = json.load(handle)["col_names"]
    print("import complete.")

    # import parameters for the ridge model we trained.
    with open("params.json", "r") as handle:
        params = json.load(handle)

    print("now cleaning the data...")

    # clean data and modify data
    x_tr_extended = build_poly(clean_data(x_train, features), 3)
    x_te_extended = build_poly(clean_data(x_test, features), 3)

    # weight final
    weight_trained, _ = ridge_regression(
        y=y_train, tx=x_tr_extended, lambda_=params["lambda_"]
    )

    # get train score
    y_pred_tr = get_classification_pred(x_tr_extended, weight_trained)
    print(f"training f1 score: {f1_score(y_train, y_pred_tr)}")

    y_test = get_classification_pred(x_tr_extended, weight_trained)
    create_csv_submission(id_test, y_test, "submission.csv")
