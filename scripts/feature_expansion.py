from turtle import shape
import numpy as np
import math

from numpy import NaN


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 0))
    degree = int(degree)
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def build_log_transformation(x, feature):
    """apply log transformation to the most skewed features of the dataset
    applying log(x+1) to avoid value of 0"""

    log = np.copy(x)
    for i in feature:
        log_column = x[:, i]
        if(any(a<0 for a in log_column)):
            log_column = np.log(1 + abs(min(log_column)) + log_column)
        else :
            log_column = np.log(1 + log_column)
        log[:, i] = log_column
    return log


def square_root_tansformation(x):
    """apply root transformation from 1/2 to 1/n"""
    sqrt = np.copy(x)
    for i in range(0, len(x[0])):
        sqrt_col = x[:, i]
        sqrt_col = [1 / val for val in sqrt_col]
        sqrt[:, i] = sqrt_col
    return sqrt


def reciprocical_tansformation(x):
    """apply the inverse transformation"""
    inv = np.copy(x)
    for i in range(0, len(x[0])):
        inv_col = x[:, i]
        inv_col = [1 / val for val in inv_col]
        inv[:, i] = inv_col
    return inv


def build_new_x(x):
    """puts together all the feature expansion methods and creates a new dataset"""
    x_new = np.copy(x)
    x_new = np.concatenate((x_new, build_poly(x, 3)), axis=1)
    x_new = np.concatenate((x_new, build_log_transformation(x)), axis=1)
    return x_new
