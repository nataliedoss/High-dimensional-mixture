import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


def em(x, k, sigma, iter):
    '''
    param:
    x: shape n-by-d
    k: int, mixture order
    
    output: k centers, k weights
    '''
    n, d = x.shape
    mu = x[np.random.choice(n, k, replace=True)]
    p = np.ones(k)/k

    i = 0
    ll_old = 1
    ll_diff = 1
    while (i < iter and ll_diff > .0001):
        M = np.exp(-np.square(pairwise_distances(x, mu))/2/(sigma**2)) * p
        M /= M.sum(axis=1)[:, np.newaxis]
        p = M.sum(axis=0)/n
        mu = M.T @ x / n / p[:, np.newaxis]

        # Compute log lhood at new estimate
        M = np.exp(-np.square(pairwise_distances(x, mu))/2/(sigma**2)) * p
        ll_new = np.sum(np.log(M.sum(axis=1)))

        # Compute the difference
        ll_diff = np.abs((ll_new - ll_old)/ll_old)
        ll_old = ll_new
        i += 1
        
    return p, mu




