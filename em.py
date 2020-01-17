import numpy as np
from sklearn.metrics import pairwise_distances



def em(x, k, sigma, max_iter, tol):
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
    M = np.exp(-np.square(pairwise_distances(x, mu))/2/(sigma**2)) * p
    ll_old = np.sum(np.log(M.sum(axis=1)))
    ll_diff = 1
    while (i < max_iter and ll_diff > tol):
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

    print("Number of iterations for EM was ", i, "llhood gap for EM was ", ll_diff)
        
    return p, mu




def em_symm(x, sigma, max_iter, tol):
    '''
    Args:
    x: nxd array
    sigma: float
    max_iter: int
    tol: float
    
    Returns:
    Array of floats, 1 x 2
    '''
    
    n, d = x.shape
    mu = x[np.random.choice(n, 1, replace=True)]

    i = 0
    M1 = np.exp(-np.square(np.linalg.norm(x-mu, axis=1)) / (2*sigma**2))
    M2 = np.exp(-np.square(np.linalg.norm(x+mu, axis=1)) / (2*sigma**2))
    ll_old = np.sum(np.log(0.5*(M1 + M2)))
    ll_diff = 1
    while(i < max_iter and ll_diff > tol):
        coeff = (M2 - M1) / (M2 + M1)
        mu = (coeff.T @ x)/n

        # Compute log lhood at new estimate
        M1 = np.exp(-np.square(np.linalg.norm(x-mu, axis=1)) / (2*sigma**2))
        M2 = np.exp(-np.square(np.linalg.norm(x+mu, axis=1)) / (2*sigma**2))
        ll_new = np.sum(np.log(0.5*(M1 + M2)))

        # Compute the difference
        ll_diff = np.abs((ll_new - ll_old)/ll_old)
        ll_old = ll_new
        i += 1

    print("Number of iterations for EM was ", i, "llhood gap for EM was ", ll_diff)
        
    return np.array((mu, -mu))

    





