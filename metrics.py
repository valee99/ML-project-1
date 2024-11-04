"""Functions to compute the metrics."""

import numpy as np


def compute_recall(y: np.array, y_pred: np.array) -> float:
    """Returns the recall of the prediction made

    Args:
        y:      shape=(N, )
        y_pred: shape=(N, )

    Returns:
        recall: a scalar
    """
    tp = (2 * y + y_pred == 3).sum()
    fn = (2 * y + y_pred == 1).sum()
    if tp == 0:
        return 0
    else:
        return tp / (tp + fn)


def compute_precision(y: np.array, y_pred: np.array) -> float:
    """Returns the precision of the prediction made

    Args:
        y:      shape=(N, )
        y_pred: shape=(N, )

    Returns:
        precision: a scalar
    """
    tp = (2 * y + y_pred == 3).sum()
    fp = (2 * y + y_pred == -1).sum()
    if tp == 0:
        return 0
    else:
        return tp / (tp + fp)


def compute_accuracy(y: np.array, y_pred: np.array) -> float:
    """Returns the accuracy of the prediction made

    Args:
        y:      shape=(N, )
        y_pred: shape=(N, )

    Returns:
        accuracy: a scalar
    """
    tp = (2 * y + y_pred == 3).sum()
    tn = (2 * y + y_pred == -3).sum()
    fn = (2 * y + y_pred == 1).sum()
    fp = (2 * y + y_pred == -1).sum()
    return (tp + tn) / (tp + tn + fp + fn)


def compute_f1_score(y: np.array, y_pred: np.array) -> float:
    """Returns the F1-Score of the prediction made

    Args:
        y:      shape=(N, )
        y_pred: shape=(N, )

    Returns:
        f1_score: a scalar
    """
    recall = compute_recall(y, y_pred)
    precision = compute_precision(y, y_pred)
    if recall == 0 and precision == 0:
        return 0
    else:
        return 2 * recall * precision / (recall + precision)
