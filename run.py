import json
import numpy as np
from scripts.helper import load_csv_data
from processing import clean_data, standardize
from crossvalidation import build_k_indices

if __name__ == "__main__":
    # import train data
    print("importing train and test data...")
    y_train, x_train, id_train = load_csv_data("../data/train.csv")
    # y_test , x_test , id_test  = load_csv_data("../data/test.csv")

    # import feature names
    with open("../data/col_name.json", "r") as handle:
        features = json.load(handle)["col_names"]
    print("import complete.")

    print("now cleaning the data...")
    # standardize and clean data
    x_tr_cleaned = standardize(clean_data(x_train, features))
    # x_te_cleaned = standardize(clean_data(x_test , features))

    # check if there exists nan in both train and test data.
    assert np.isnan(x_tr_cleaned).sum() == 0, print(
        "there exists nan in the train data. Exit process."
    )
    # assert np.isnan(x_te_cleaned).sum() == 0, \
    #     print('there exists nan in the test data. Exit process.')
    print("cleaning complete.")

    print(build_k_indices(y_train, 5))
