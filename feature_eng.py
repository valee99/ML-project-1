"""Functions for feature engineering including polynomial expansions and kmeans."""

import numpy as np


def polynomial_expansion(
    x: np.array, feature: str, features_list: list[str], max_degree: int
) -> np.array:
    """Adds polynomial expansion of a selected feature

    Args:
        x: shape=(N, D)
        feature: a string with the name of the feature to expand
        features_list: the list of features in the dataset in the same order as the columns
        max_degree: a positive scalar denoting the maximum degree of the polynomial expansion

    Returns:
        x: shape=(N, D + max_degree - 1)
    """
    feature_vector = x[:, features_list.index(feature)]
    for degree in range(2, max_degree):
        feature_vector_poly = np.power(feature_vector, degree)
        features_list.append(feature + f"_{degree}")
        x = np.concatenate((x, feature_vector_poly[:, np.newaxis]), axis=1)
    return x


def compute_distances(x: np.array, centroids: np.array) -> np.array:
    """Computes the distances between each datapoint and each centroid

    Args:
        x: shape=(N, D)
        centroids: shape=(k, D)

    Returns:
        distances: shape=(N, k)
    """
    distances = np.zeros((x.shape[0], centroids.shape[0]))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(x - centroid, axis=1)
    return distances


def update_centroids(x: np.array, labels: np.array, k: int) -> np.array:
    """Updates the position of the centroids based on the members of each cluster

    Args:
        x: shape=(N, D)
        labels: shape=(N, 1)
        k: a scalar denoting the number of clusters

    Returns:
        centroids: shape=(k, D)
    """
    centroids = np.zeros((k, x.shape[1]))
    for i in range(k):
        cluster_points = x[labels == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids


def kmeans(
    x: np.array, k: int, seed: int, max_iters: int = 100, tol: float = 1e-4
) -> tuple[np.array]:
    """Runs the K-means algorithm and returns the results centroids and labels for each datapoint

    Args:
        x: shape=(N, D)
        k: a scalar denoting the number of clusters
        seed: a scalar setting the seed for the random operations
        max_iters: a scalar denoting the maximum number of iterations to run
        tol: a float denoting the minimum gap between two steps of the centroids to stop the algorithm earlier

    Returns:
        centroids: shape=(k, D)
        labels: shape=(N, 1)
    """

    np.random.seed(seed)
    indices = np.random.choice(x.shape[0], size=k, replace=False)
    centroids = x[indices]

    for i in range(max_iters):
        distances = compute_distances(x, centroids)
        labels = np.argmin(distances, axis=1)

        new_centroids = update_centroids(x, labels, k)

        if np.linalg.norm(centroids - new_centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels
