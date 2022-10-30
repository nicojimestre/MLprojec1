import numpy as np
from typing import Tuple
from metrics import mse_loss

def get_classification_pred(tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    pred = np.where(sigmoid(tx @ w) > 0.5, 1, -1)
    return pred


def least_squares(y: np.ndarray, tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    loss = mse_loss(y, tx, w)
    return w, loss


def ridge_regression(
    y: np.ndarray, tx: np.ndarray, lambda_: float
) -> Tuple[np.ndarray, float]:
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
    """
    d = tx.shape[1]  # number of columns (i.e. variables)
    lambda_dash = lambda_ / (2 * len(y))
    w = np.linalg.inv(tx.T @ tx + lambda_dash * np.eye(d)) @ tx.T @ y
    loss = mse_loss(y, tx, w)
    return w, loss


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    prediction = np.where(tx @ w > 0, 1, -1)
    error = y - prediction
    return -np.mean(tx.T @ error)


def mean_squared_error_gd(
    y: np.ndarray,
    tx: np.ndarray,
    initial_w: np.ndarray,
    max_iters: np.ndarray,
    gamma: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    losses = []
    w = initial_w
    for _ in range(max_iters):
        # calculate the loss and the gradient given the weight, w
        loss, gradient = mse_loss(y, tx, w), compute_gradient(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)
    return w, losses[-1]


def compute_stoch_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    n = len(y)
    rand_idx = np.random.randint(0, n)
    prediction = np.where(tx @ w > 0, 1, -1)
    error = y - prediction
    return -(tx[rand_idx].T * error[rand_idx])


def mean_squared_error_sgd(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, float]:
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
    """

    # Define parameters to w and loss
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # calculate the loss and the gradient given the weight, w
        loss, gradient = mse_loss(y, tx, w), compute_stoch_gradient(y, tx, w)
        w = w - gamma * gradient
        losses.append(loss)
    return w, losses[-1]


def sigmoid(t: float) -> float:
    """
    apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return np.exp(t) / (1 + np.exp(t))


def calculate_nll_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> float:
    """
    compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss: float
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    loss = np.log(1 + np.exp(tx @ w)) - y * (tx @ w)
    return np.mean(loss)


def calculate_logistic_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    n = len(y)
    gradient = tx.T @ (sigmoid(tx @ w) - y) / n
    return gradient


def logistic_regression(
    y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float
) -> Tuple[np.ndarray, float]:
    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        loss, gradient = calculate_nll_loss(y, tx, w), calculate_logistic_gradient(
            y, tx, w
        )
        w = w - gamma * gradient

        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]


def penalized_loss_and_gradient(
    y: np.ndarray, tx: np.ndarray, w: np.ndarray, lambda_: float
) -> Tuple[float, np.ndarray]:
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    n = len(y)
    loss = calculate_nll_loss(y, tx, w)
    loss += (w.T @ w).item() * lambda_

    # calculate regularized gradient
    gradient = tx.T @ (sigmoid(tx @ w) - y) / n + 2 * lambda_ * w
    return loss, gradient


def reg_logistic_regression(
    y: np.ndarray,
    tx: np.ndarray,
    lambda_: float,
    initial_w: np.ndarray,
    max_iters: int,
    gamma: float,
) -> Tuple[np.ndarray, float]:

    # init parameters
    threshold = 1e-8
    losses = []
    w = initial_w

    # start the logistic regression
    for _ in range(max_iters):
        # get loss and update w.
        _, gradient = penalized_loss_and_gradient(y, tx, w, lambda_)
        loss = calculate_nll_loss(
            y, tx, w
        )  # for recording losses, no penalty is added.
        w = w - gamma * gradient
        
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, losses[-1]