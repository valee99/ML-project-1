"""Useful functions for project 1."""

import numpy as np
from typing import Union


def batch_iter(
    y: np.array,
    tx: np.array,
    batch_size: int,
    num_batches: int = 1,
    shuffle: bool = True,
):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Args:
        y: shape(N,)
        tx: shape(N,2)
        batch_size: a scalar denoting the size of each batch
        num_batches: a scalar denoting the number of batches to generate
        shuffle: a boolean denoting if the batch should be shuffled
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def compute_loss_mse(y: np.array, tx: np.array, w: np.array) -> float:
    """Calculate the loss using MSE.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - np.dot(tx, w)
    N = y.shape[0]
    loss = np.dot(error.T, error) / (2 * N)
    return loss


def compute_gradient_mse(y: np.array, tx: np.array, w: np.array) -> np.array:
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    error = y - np.dot(tx, w)
    N = y.shape[0]
    gradient = -np.dot(tx.T, error) / N
    return gradient


def sigmoid(t: Union[float, np.array]) -> Union[float, np.array]:
    """Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    result = 1.0 / (1 + np.exp(-t))
    return result


def compute_loss_neg_log_lh_sigmoid(y: np.array, tx: np.array, w: np.array) -> float:
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a non-negative loss
    """
    N = y.shape[0]
    a = sigmoid(np.dot(tx, w))
    loss = (
       np.squeeze(-(np.dot(y.T, np.log(a)) + np.dot((1 - y).T, np.log(1 - a))))
       / N
    )
    return loss


def compute_gradient_neg_log_lh_sigmoid_weighted(
    y: np.array, tx: np.array, w: np.array, class_weight: dict
) -> np.array:
    """compute the weighted gradient of loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        class_weight: {class: class_weight}

    Returns:
        a vector of shape (D, )

    """
    N = y.shape[0]
    a = sigmoid(np.dot(tx, w)) - y
    weighted_a = np.where(a == 1, a * class_weight[1], a * class_weight[0])
    gradient = np.dot(tx.T, weighted_a) / N
    return gradient


def compute_gradient_neg_log_lh_sigmoid(
    y: np.array, tx: np.array, w: np.array
) -> np.array:
    """compute the gradient of loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )

    Returns:
        a vector of shape (D, )

    """
    N = y.shape[0]
    a = sigmoid(np.dot(tx, w)) - y
    gradient = np.dot(tx.T, a) / N
    return gradient


def mean_squared_error_gd(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> tuple[np.array, float]:
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient

    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


def mean_squared_error_sgd(
    y: np.array,
    tx: np.array,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
    batch_size: int = 1,
) -> tuple[np.array, float]:
    """The Stochastic Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            stoch_gradient = compute_gradient_mse(y_batch, tx_batch, w)
            w = w - gamma * stoch_gradient

    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


def least_squares(y: np.array, tx: np.array) -> tuple[np.array, float]:
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    gram = np.dot(tx.T, tx)
    w = np.linalg.solve(gram, np.dot(tx.T, y))
    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


def ridge_regression(
    y: np.array, tx: np.array, lambda_: float
) -> tuple[np.array, float]:
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    gram = np.dot(tx.T, tx)
    regu = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    w = np.linalg.solve(gram + regu, np.dot(tx.T, y))
    loss = compute_loss_mse(y, tx, w)
    return (w, loss)


def logistic_regression(
    y: np.array, tx: np.array, initial_w: np.array, max_iters: int, gamma: float
) -> tuple[np.array, float]:
    """Logistic regression using gradient descent.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_neg_log_lh_sigmoid(y, tx, w)
        w = w - gamma * gradient

    loss = compute_loss_neg_log_lh_sigmoid(y, tx, w)

    return (w, loss)


def reg_logistic_regression(
    y: np.array,
    tx: np.array,
    lambda_: float,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
) -> tuple[np.array, float]:
    """Regularized logistic regression using gradient descent.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        lambda_: a scalar denoting the strength of the regularization
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient_regu = compute_gradient_neg_log_lh_sigmoid(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient_regu

    loss = compute_loss_neg_log_lh_sigmoid(y, tx, w)

    return (w, loss)


def elastic_net_logistic_regression_weighted(
    y: np.array,
    tx: np.array,
    lambda_: float,
    initial_w: np.array,
    max_iters: int,
    gamma: float,
    class_weight: dict,
    alpha: float,
    batch_size: int,
) -> tuple[np.array, float]:
    """Elasticnet regularized logistic regression using stochastic gradient descent.

    Args:
        y: shape=(N, )
        tx: shape=(N, 2)
        lambda_: a scalar denoting the strength of the regularization
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        class_weight: {class: class_weight}
        alpha: a scalar denoting the balance between Lasso and Ridge regularization
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: the loss of the model with the optimal weights
    """

    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        if batch_size != 0:
            for y_batch, tx_batch in batch_iter(
                y, tx, batch_size=batch_size, num_batches=1
            ):
                gradient_regu = compute_gradient_neg_log_lh_sigmoid_weighted(
                    y_batch, tx_batch, w, class_weight
                ) + lambda_ * (2 * (1 - alpha) * w + alpha * np.sign(w))
                loss = compute_loss_neg_log_lh_sigmoid(y, tx, w)
                w = w - gamma * gradient_regu
                losses.append(loss)
        else:
            gradient_regu = compute_gradient_neg_log_lh_sigmoid_weighted(
                y, tx, w, class_weight
            ) + lambda_ * (2 * (1 - alpha) * w + alpha * np.sign(w))
            loss = compute_loss_neg_log_lh_sigmoid(y, tx, w)
            w = w - gamma * gradient_regu
            losses.append(loss)

    loss = compute_loss_neg_log_lh_sigmoid(y, tx, w)

    return (w, loss)
