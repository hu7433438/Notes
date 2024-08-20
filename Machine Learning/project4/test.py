import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

mixture, post = common.init(X, K, seed)
new_post, log_lh = em.estep(X, mixture)
print(new_post, log_lh)
new_mixture = em.mstep(X, new_post, mixture)
print(new_mixture)