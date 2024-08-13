"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    part1 = 1/((2 * np.pi * mixture.var) ** (X.shape[1] / 2))
    part2 = -np.linalg.norm(X[:, None] - mixture.mu, axis=2) ** 2 / (2 * mixture.var)
    results = part1 * np.exp(part2)

    gama_1 = results*mixture.p
    gama_2 = np.sum(gama_1, 1).reshape(-1, 1)
    gama = gama_1 / gama_2
    log_lh = np.sum(np.log(gama_2), 0).item()
    return gama, log_lh


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    sum_gama = np.sum(post, axis=0)
    p = sum_gama / n
    mu = (post.T.dot(X))/sum_gama.reshape(-1, 1)
    var = np.sum(post*(np.linalg.norm(X[:, None] - mu, axis=2) ** 2), 0)/(sum_gama*d)
    return GaussianMixture(mu, var, p)



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    current_log_lh = None
    log_lh = None
    while current_log_lh is None or log_lh - current_log_lh > np.abs(log_lh)*10**-6:
        current_log_lh = log_lh
        post, log_lh = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, log_lh
