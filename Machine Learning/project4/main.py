import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4]

for k in K:
    costs = [0, 0, 0, 0, 0]
    logs = [0, 0, 0, 0, 0]
    kmeans_mixtures = [0, 0, 0, 0, 0]
    kmeans_posts = [0, 0, 0, 0, 0]
    naive_mixtures = [0, 0, 0, 0, 0]
    naive_posts = [0, 0, 0, 0, 0]
    for s in range(len(seed)):
        kmeans_mixtures[s], kmeans_posts[s], costs[s] = kmeans.run(X, *common.init(X, k, s))
        naive_mixtures[s], naive_posts[s], logs[s] = naive_em.run(X, *common.init(X, k, s))

    print(min(costs), max(logs))
    # common.plot(X, kmeans_mixtures[np.argmin(costs)], kmeans_posts[np.argmin(costs)], f"kmeans:{k}")
    # common.plot(X, naive_mixtures[np.argmax(logs)], naive_posts[np.argmax(logs)], f"naive:{k}")
    print(common.bic(X, naive_mixtures[np.argmax(logs)], logs[np.argmax(logs)]))
