import numpy as np 

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_log_transformation(x, features) :
    """apply log transformation to the most skewed features of the dataset
    applying log(x+1) to avoid value of 0"""

    log = np.copy(x)

    for feature in features:
        log_column = x[:,feature]
        log_column = np.log(1+log_column)
        log[:,feature] = log_column

    return log