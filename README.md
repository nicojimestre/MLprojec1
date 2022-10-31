# CS-433: ML Project 1 - Higgs Boson Challenge

# Intro

This project aims to predict whether each decay signatures generated from collisions are actually from Higgs Boson. We used the experiment data supplied by CERN, which contains records of collision experiment of protons. For more information about the competition, its background, the data, please see the AI Crowd platform ( https://www.aicrowd.com/challenges/epfl-machine-learning-higgs ).

You will find more information about our source codes and methods in this readme.md file.

### Authors

- _Nicolas Jimenez_
- _Blanche Marion_
- _Ki Beom Kim_

## Setting up

This code was tested with Python 3.8 or higher and below is the list of libraries used for this project.

    numpy
    matplotlib

As a default `numpy` is installed. For other packages, you will need to run `requirements.pip` to download the necessary packages. To do this, you will need to make a python virtual environment then run the command below:

    pip install -r requirements.pip

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

| Function                  | Arguments                                        |
| ------------------------- | ------------------------------------------------ |
| `least_squares_GD`        | `y, tx, initial_w, max_iters, gamma`             |
| `least_squares_SGD`       | `y, tx, initial_w, batch_size, max_iters, gamma` |
| `least_squares`           | `y, tx`                                          |
| `ridge_regression`        | `y, tx, lambda_`                                 |
| `logistic_regression`     | `y, tx, initial_w, max_iters, gamma`             |
| `reg_logistic_regression` | `y, tx, lambda_, initial_w, max_iters, gamma`    |

### run.py

`run.py` generates a csv file read for the submission. This file runs as the procedure shown below.:

1. loading the `train.py`, `test.py`
2. data cleaning:
   - impute the missing values (shown as `-999`) with the median
   - removing outliers by replacing it with 1.5 x 75%iles of the data
3. Feature Engineering:

   - polynomial expansion
   - log-transformation
   - sqaure-root transform
   - reciprocal transform

4. using `HyperParameterTuner` to tune the hyperparameter of `ridge_regression` model. (hyperparameter: `lambda_`) Parameter tuning will be done by 5-fold cross-validation. Best parameter will be selected by observing validaiton F1 score.

5. obtain the weight vector w using the trained `lambda_`.

6. get predictions and save the prediction as .csv format
