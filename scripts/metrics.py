import numpy as np

"""this file has functions that measure the performance of model. We have F1 score since
the task of project is predict binary variable. 
"""


def mse_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    calculates mse loss of categorical target y
    """
    prediction = np.where(tx @ w > 0, 1, -1)
    error = y - prediction
    return 0.5 * np.mean(error**2)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    calculates f1 score of categorical target y
    """
    # find actual positive, negative events
    positive_events, negative_events = (y_true == 1), (y_true == -1)

    # calculate TP, FP, TN, FN
    true_positive = (y_true[positive_events] == y_pred[positive_events]).sum()
    false_negative = (y_true[positive_events] != y_pred[positive_events]).sum()
    true_negative = (y_true[negative_events] == y_pred[negative_events]).sum()
    false_positive = (y_true[negative_events] != y_pred[negative_events]).sum()

    # calculate precission and recall
    precission = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    # calculate F1 score
    f1_score = 2 * (precission * recall) / (precission + recall)
    return f1_score
