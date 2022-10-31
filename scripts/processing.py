from feature_expansion import build_poly
import numpy as np
from feature_expansion import *

"""Pre-processing functions used to clean our dataset"""
"""Use function pre_process_data"""


def group_by_PRE_jet_num(X, y):
    """Group the data into four sets, depending on whether their PRI_jet_num is {0,1,2,3}
    Return the four sets and their label"""

    # create masks to extract the subsets
    # then, extract the elements from each subset
    """for pri_jet in [0, 1, 2, 3]:
        exec(f"mask_{pri_jet} = X[:,22] == {pri_jet}")
        exec(f"X_{pri_jet} = X[mask_{pri_jet},:]")

    # extract the corresponding labels
    if len(y) > 0:
        for pri_jet in [0, 1, 2, 3]:
            exec(f"y_{pri_jet} = np.asarray(y[mask_{pri_jet}])")"""
    #create masks to extract the subsets
    mask_0 = X[:,22] == 0
    mask_1 = X[:,22] == 1
    mask_2 = X[:,22] == 2
    mask_3 = X[:,22] == 3

    #extract the elements from each subset
    X_0 = X[mask_0,:]
    X_1 = X[mask_1,:]
    X_2 = X[mask_2,:]
    X_3 = X[mask_3,:]

    if(len(y)>0):
        y_0 = np.asarray(y[mask_0])
        y_1 = np.asarray(y[mask_1])
        y_2 = np.asarray(y[mask_2])
        y_3 = np.asarray(y[mask_3])

    return (
        X_0,
        y_0,
        X_1,
        y_1,
        np.concatenate((X_2, X_3), axis=0),
        np.concatenate((y_2, y_3), axis=0),
    )


def clean_data(tx: np.ndarray, features: list) -> np.ndarray:
    """
    cleans the original data by removing the outlier (-999.0) and truncates the variable
    that goes over the cap value (in terms of qunatiles) provided.

    Parameters
    tx: np.ndarray (n x k)
      input data to clean of n data rows, k features.
    features: list (k)
      list of feature names. length should be equal to k

    Output
    cleaned: np.ndarray
      cleaned data.
    """
    quantiles = np.ndarray([len(features), 2])
    cleaned = np.copy(tx)

    # find 75% quantile and set the upper bound
    for i, variable in enumerate(cleaned.T):
        quantiles[i] = np.quantile(variable, [0.75])
    upper_bound = [x[1] * 1.5 for x in quantiles]

    # given the column index, truncate the variables with
    # upper bound.
    to_modify = [0, 1, 2, 3, 5, 8, 9, 10, 13, 16, 19, 21, 23, 26, 20]
    for i in to_modify:
        u_capped_value = upper_bound[i]
        variable = np.where(variable < u_capped_value, variable, u_capped_value)
    cleaned = np.delete(cleaned, features.index("PRI_jet_num"), axis=1)

    j = 0
    for i, variable in enumerate(cleaned.T):
        # replace outlier with variables mode.
        # If all columns have values of -999, we delete the feature
        if len(variable[variable == -999.0]) > 0:
            if len(variable[variable != -999.0]) == 0:
                cleaned = np.delete(cleaned, j, axis=1)
                j -= 1
            else:
                variable[(variable == -999.0)] = np.round(
                    np.median(variable[variable != -999.0]), 2
                )
        #elif np.std(variable, axis=0) == 0:
        elif np.all(variable == variable[0]):
            cleaned = np.delete(cleaned, j, axis=1)
            j -= 1
        j += 1
    return cleaned


def standardize_(x):
    """Standardizes the dataset i.e. substract the mean and divide by the standard deviation"""
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)
    if std_x != 0 :
        x = x - mean_x
        x = x / std_x
    else :
        x = np.maximum(x) - np.minimum(x)

    return x


def pre_process_data(X,y, features: list):
     """Puts together all the pre-processing steps and returns the 2 processes datasets"""
     X_0, y_0, X_1, y_1, X_23, y_23 = group_by_PRE_jet_num(X,y)
     #remove invalid features
     X_0 = clean_data(X_0, features)
     X_1 = clean_data(X_1, features)
     X_23 = clean_data(X_23, features)

     # use feature engineering to get new Xs.
     #X_0_extended = build_new_x(X_0)
     #X_1_extended = build_new_x(X_1)
     #X_23_extended = build_new_x(X_23)

     #standardize the subsets
     #X_0[0:len(X_0), 0:len(X_0[0])] = standardize(X_0)
     #X_1[0:len(X_1), 0:len(X_1[0])] = standardize(X_1)
     #X_23[0:len(X_23), 0:len(X_23[0])] = standardize(X_23)
     
     #X_0 = standardize(X_0)
     #X_1 = standardize(X_1)
     #X_23 = standardize(X_23)

     #add a column of 1 to the subsets
     #print(len(X_0))
     #print(len(X_0[0]))
     #ones = np.ones([len(X_0), 1])
     #X_0 = np.append(ones, X_0, axis = 1)
     #ones = np.ones([len(X_1), 1])
     #X_1 = np.append(ones, X_1, axis = 1)
     #ones = np.ones([len(X_23), 1])
     #X_23 = np.append(ones, X_23, axis = 1)


     return X_0, y_0, X_1, y_1, X_23, y_23

def pre_process_data_ws(X,y, features: list, degree_0, degree_1, degree_23):
     """Puts together all the pre-processing steps and returns the 2 processes datasets"""
     X_0, y_0, X_1, y_1, X_23, y_23 = group_by_PRE_jet_num(X,y)
     #remove invalid features
     X_0 = clean_data(X_0, features)
     X_1 = clean_data(X_1, features)
     X_23 = clean_data(X_23, features)

     # use feature engineering to get new Xs.
     #X_0_extended = build_new_x(X_0)
     #X_1_extended = build_new_x(X_1)
     #X_23_extended = build_new_x(X_23)

     #standardize the subsets
     #X_0[0:len(X_0), 0:len(X_0[0])] = standardize(X_0)
     #X_1[0:len(X_1), 0:len(X_1[0])] = standardize(X_1)
     #X_23[0:len(X_23), 0:len(X_23[0])] = standardize(X_23)
     #X_0 = standardize(X_0)
     #X_1 = standardize(X_1)
     #X_23 = standardize(X_23)
     ones = np.ones([len(X_0), 1])
     X_0 = np.append(ones, X_0, axis = 1)
     ones = np.ones([len(X_1), 1])
     X_1 = np.append(ones, X_1, axis = 1)
     ones = np.ones([len(X_23), 1])
     X_23 = np.append(ones, X_23, axis = 1)

     X_0 = build_poly(X_0, degree_0)
     X_1 = build_poly(X_1,degree_1)
     X_23 = build_poly(X_23, degree_23)
     
     return X_0, y_0, X_1, y_1, X_23, y_23

def pre_process_whole_data(X, features: list):
     """Puts together all the pre-processing steps and returns the 2 processes datasets"""
     #remove invalid features
     X = clean_data(X, features)
     
     return X

