# CS-433: ML Project 1 - Higgs Boson Challenge

# Intro

This project aims to predict whether each decay signatures generated from collisions are actually from Higgs Boson. We used the experiment data supplied by CERN, which contains records of collision experiment of protons. For more information about the competition, its background, the data, please see the AI Crowd platform ( https://www.aicrowd.com/challenges/epfl-machine-learning-higgs ).

You will find more information about our source codes and methods in this readme.md file.

## Setting up

This code was tested with Python 3.8 or higher and below is the list of libraries used for this project.

    numpy
    matplotlib

As a default `numpy` is installed. For other packages, you will need to run `requirements.pip` to download the necessary packages. To do this, you will need to make a python virtual environment then run the command below:

`pip install -r requirements.pip`

## Repo Structure

The folder structure has to be the following:

    ├── Data              # Data files, in .csv
    |    ├── train.csv
    |    └── test.csv
    ├── col_name.json     # column names for train, test data.
    ├── scripts           # Source files as well as notebook files for data analysis
    ├── run.py            # runs the whole process and outputs submission.
    └── README.md


### Used Methods 

Below is the list of ML methods that we implemented. These can be found in `implementation.py`. Out of 6, `least_square`, `ridge_regression` use the normal equation whereas other 4 methods use optimization methods (in this case mostly gradient descent, SGD only for `least_squares_SGD`). 

| Function                  | Arguments                                                           |
| ------------------------- | ------------------------------------------------------------------- |
| `least_squares_GD`        | `y, tx, initial_w[, max_iters, gamma, *args, **kwargs]`             |
| `least_squares_SGD`       | `y, tx, initial_w[, batch_size, max_iters, gamma, *args, **kwargs]` |
| `least_squares`           | `y, tx[, **kwargs]`                                                 |
| `ridge_regression`        | `y, tx, lambda_[, **kwargs]`                                        |
| `logistic_regression`     | `y, x, [w, max_iters, gamma, **kwars]`                              |
| `reg_logistic_regression` | `y, x, lambda_, [initial_w, max_iters, gamma, **kwargs]`            |

The default values were chosen in order to get convergence on the GD algorithm.

### run.py

This script produces a csv file containing the predictions `Kaggle_CDM_submission.csv`. The following are executed:

a) loading the data

b) data processing:

        - applying the log transformations with translation
        - impute the missing values with the median
        - normalize the data
        - split the variable `num_jet` into 4 categorical variables

b) polynomial extension of degree 4 of the data

c) interactions between the categorical variables and the continuous features

d) training a Ridge regression model, using `cross_validation` to determine the hyper-parameter `lambda`

e) training the Ridge regression model on the whole training data set with the determined `lambda` to obtain the weight vector w

f) compute predictions and create the .csv file

The data preprocessing applies log transformation to a specific set of features after translating some of them. Then it imputes the mean for the missing values and take out the phi and eta features out.

### Implementation of class methods.

For the sake of automation we added a `pred` keyword argument (kwarg) to all our model functions. It is `False` by default, and if set to `True` the function returns as first output a pointer on the function to use in order to get the predictions for that model.

All functions using the `gradient_descent` algorithm have, in addition, the two kwargs `printing, all_step`, which are `False` by default. If `printing=True`, then at all GD steps you will see in the shell the actual mse value and the value of the first two parameters of `w`. If `all_step=True`, then the function returns all the computed w-s and errors (by default they are not stored and only the last value is given).

The following functions were implemented:

| Function                  | Arguments                                                           |
| ------------------------- | ------------------------------------------------------------------- |
| `least_squares_GD`        | `y, tx, initial_w[, max_iters, gamma, *args, **kwargs]`             |
| `least_squares_SGD`       | `y, tx, initial_w[, batch_size, max_iters, gamma, *args, **kwargs]` |
| `least_squares`           | `y, tx[, **kwargs]`                                                 |
| `ridge_regression`        | `y, tx, lambda_[, **kwargs]`                                        |
| `logistic_regression`     | `y, x, [w, max_iters, gamma, **kwars]`                              |
| `reg_logistic_regression` | `y, x, lambda_, [initial_w, max_iters, gamma, **kwargs]`            |

The default values were chosen in order to get convergence on the GD algorithm.

###### ATTENTION:

Since our goal is to find a classification model, we have that all functions compute the error vector **err** as categorical (i.e if **y_hat** is the vector of estimated categories and **y** the true categories, j-th coordinate of **err** will be `err[j]=1{y[j]=y_hat[j]}`, where **1** is the indicator function.). Furthermore, the loss value returned is the misclassification ratio (i.e. the number of wrong predictions over the total number of predictions).

If one desires to implement our functions for different tasks, it is enough to set the two global functions `err_f` and `loss_f` to the desired ones.

| Possible `loss_f` |      | Possible `err_f` |
| ----------------- | ---- | ---------------- | ----------------------------------------------------------------------------- |
| `calculate_mae`   | MAE  | `error`          | Return the distance between predicted end true values (continuous estimation) |
| `calculate_mse`   | MSE  | `category_error` | Return the error indicator (as explained above)                               |
| `calculate_rmse`  | RMSE |                  |

They can be set as follows:

    err_f = error #For continuous estimation.
    loss_f = calculate_mse #For mean squared error loss.

### Notes on `cross_validation` and `multi_cross_validation`

These are the two main functions inplemented in order to choose our model, and in particular to get an estimation of the prediction error.

- `cross_validation(y, tx, k_fold, method, *args_method[, k_indices, seed])` compute the k-fold cross validation for the estimation of `y` using a the method-function stored (as pointer) in the argument `method`. The arguments necessary for the `method` are to be passed freely after method. It returns `predictor, w, loss_tr, loss_te`, which are, in order, the predicting function, the mean of the trained weights, the mean of the train error and the estimate test error.

- `multi_cross_validation(y, x, k_fold[, transformations=[[id, []]], methods=[[least_squares, []]], seed=1, only_best=True])` Perform automatically the cross validation on all the combinations of transformations in the `transformations` list (their parameters have to be passed as a list coupled with the transformation) and methods with changing parameters in the `methods` list (the coupled list have in this case to be a list of the tuples of parameters combinations to test.) It then plots the estimated losses (both on train and test) and outputs `predictor, weight, losses_tr, losses_te, transformations_list, methods_list`. If `only_best=True`, those are the variables corresponding to the lowest test-error estimate, otherwise they contain the variables computed at each step. An implementation example can be found in the documentation.

## Authors

- _William Cappelletti_
- _Charles Dufour_
- _Marie Sadler_
