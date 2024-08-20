"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    f = np.zeros((n, mixture.mu.shape[0]), dtype=np.float64)
    for i in range(n):
        cu_indices = X[i, :] != 0
        part1 =  (-np.sum(cu_indices) / 2) *np.log(2 * np.pi * mixture.var)
        part2 = np.linalg.norm((X[i, cu_indices] - mixture.mu[:, cu_indices]), axis=1) ** 2 / (2 * mixture.var) # This will be (K,)
        f[i,:] = part1 - part2

    f = f + np.log(mixture.p + 1e-16)

    log_sum  = logsumexp(f, axis=1).reshape(-1,1)
    post = np.exp(f - log_sum)
    return post, np.sum(log_sum)



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu = mixture.mu
    p = np.sum(post, axis=0) / n

    delta = X.astype(bool).astype(int)
    denominator = post.T @ delta
    update_indices = np.where(denominator >= 1)
    mu[update_indices] = (post.T @ X)[update_indices]/denominator[update_indices]
    var = np.maximum(np.sum(post * (np.linalg.norm((X[:, None] - mu) * delta[:, None, :], axis=2) ** 2), 0) / (np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0)), min_variance)
    # mu = (post.T.dot(X))/sum_gama.reshape(-1, 1)
    # var = np.sum(post*(np.linalg.norm(X[:, None] - mu, axis=2) ** 2), 0)/(sum_gama*d)
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
    while current_log_lh is None or log_lh - current_log_lh > np.abs(log_lh) * 10 ** -6:
        current_log_lh = log_lh
        post, log_lh = estep(X, mixture)
        mixture = mstep(X, post, mixture)
    return mixture, post, log_lh


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n = X.shape[0]
    f = np.zeros((n, mixture.mu.shape[0]), dtype=np.float64)
    for i in range(n):
        cu_indices = X[i, :] != 0
        part1 =  (-np.sum(cu_indices) / 2) *np.log(2 * np.pi * mixture.var)
        part2 = np.linalg.norm((X[i, cu_indices] - mixture.mu[:, cu_indices]), axis=1) ** 2 / (2 * mixture.var)
        f[i,:] = part1 - part2

    f = f + np.log(mixture.p + 1e-16)
    post = np.exp(f - logsumexp(f, axis=1).reshape(-1,1))

    x_pred = X.copy()

    miss_indices = np.where(X == 0)
    x_pred[miss_indices] = (post @ mixture.mu)[miss_indices]

    return x_pred
