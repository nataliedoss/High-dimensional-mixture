"""
module for high-dimensional denoised method of moments algorithm
Note that the original DMM.estimate() function returns an object of class ModelGM
The DMM_HD.estimate_ld() and DMM_HD.estimate() functions return objects of class DiscreteRV_HD
"""

from dmm import DMM
from model_gm import ModelGM_HD, sample_gm
from discrete_rv import DiscreteRV_HD, wass_hd
from sklearn.metrics import pairwise_distances
import itertools
from scipy.special import comb
import numpy as np
import time

#################################################################################

class DMM_HD():
    """
    class for spectral sliced method of moments for high dimensional gaussian mixture model

    """

    def __init__(self, k, ld, sigma):
        """
        Args:
        k: Integer. The number of components we believe are in the model.
        ld: Integer. Our estimated latent dimension of the model. Usually k or k-1 (the latter if we centered).
           But if d < k-1, it will be d.
        sigma: Float. The known standard deviation.
        """
        
        self.k = k
        self.ld = ld
        self.sigma = sigma


    def estimate_center_space(self, sample):
        """
        Method to estimate the space spanned by the signal of sample.
        It assumes the data dimension d > ld.

        Args:
        sample: Array(float, n x d). Sample from a d-dim Gaussian mixture model.
        sigma: Float. Standard deviation of the model  .
        ld: Float. Estimate of the latent dimension (dimension of the center space).

        Returns:
        Array(float, d x ld). Top ld orthonormal evecs of (sample covariance - sigma^2 I_d). 

        """

        num = sample.shape[0]
        d = sample.shape[1]
        sample_cov = np.matmul(np.transpose(sample), sample) / num
        U, D, Ut = np.linalg.svd(sample_cov - np.square(self.sigma) * np.identity(d))
        return U[:, 0:self.ld]

    
    def compute_rate_inverse(self, num):
        """
        Method to compute the inverse rate given the number of samples. 

        Args:
        num: Int. Number of samples in dataset.

        Returns: 
        Int. The inverse minimax rate, to be used in grid search.
        """
        
        return round((num)**(1.0/(4.0*self.k - 2.0)))



    def generate_net_weights(self, num, factor_weights):
        """
        Method to generate epsilon net on weights. Epsilon is rate.

        Args:
        num: Int. Number of samples in dataset.
        factor_weights: Float. Factor by which to multiply the rate. 
            If > 1.0, makes grid finer.
        
        Returns:
        Array(float, L x k), where:
            L = (a+k-1) choose (k-1).
            a = factor_weights * rate_inverse.
        """
        
        rate_inverse = self.compute_rate_inverse(num)
        size = factor_weights * rate_inverse
        if self.k == 3:
            grid_1d = np.linspace(0.0, 1.0, size+1) # Include endpoints
            grid = [] 
            for i in range(len(grid_1d)):
                for j in range(len(grid_1d) - i):
                    grid.append(np.array((grid_1d[i], grid_1d[j], 1 - grid_1d[i] - grid_1d[j])))
            return np.asarray(grid)
        else:
            return simplex_grid(self.k, size) / (size) # Use the Oyama simplex_grid function


    def generate_net_thetas(self, num, factor_thetas):
        """
        Method to generate epsilon net on unit sphere S^{ld-1}.
        When ld=2, this method generates (1/epsilon) grid points on alpha, then forms the unit vectors (cos alpha, sin alpha).
        For ld!=2, his method generates (factor_thetas/epsilon))^(ld-1) random Gaussian vectors in ld space and normalizes them. Approximate grid. 
        This is intended to be used only when ld >= 2. It can be run for ld=1, but in that case will generate a net of size 1. 
        This is fine since this function is never invoked in the main algorithm when ld=1.
        Here epsilon is the rate, so 1/epsilon is rate_inverse.

        Args:
        num: Int. Number of samples in dataset.
        factor_thetas: Float. factor by which to multiply the rate.
            If > 1.0, makes grid finer.

        Returns:
        Array(float, (size of epsilon net in ld-dimensional unit sphere) x ld).
        """

        rate_inverse = self.compute_rate_inverse(num)
        
        if self.ld == 2:
            grid_angle = np.linspace(-np.pi, np.pi, factor_thetas*rate_inverse)
            return  np.array((np.cos(grid_angle), np.sin(grid_angle))).T
        
        else:
            nt = (factor_thetas*rate_inverse)**(self.ld-1)
            thetas = np.random.multivariate_normal(np.zeros(self.ld), np.identity(self.ld), int(nt))
            return (thetas.T / np.apply_along_axis(np.linalg.norm, 1, thetas)).T

    
    '''
    def generate_net_thetas(self, num, factor_thetas):
        """
        One method to generate epsilon net on unit vectors in ld space.
        This method generates (1/epsilon) grid points in each direction in ld space, then takes all combinations.
        This method leads to a net of size exponential in ld instead of ld-1.
        BUT this method only works for self.ld = 2. 

        Args:
        num: Int. Number of samples in dataset.
        factor_thetas: Float. factor by which to multiply the rate.
            If > 1.0, makes grid finer.

        Returns:
        Array(float, (size of net) x ld).
        """

        rate_inverse = self.compute_rate_inverse(num)
        grid_1d = np.arange(-1.0, 1.0, 1.0/(factor_thetas * rate_inverse))
        thetas = np.empty((len(grid_1d)**self.ld, self.ld))
        for i in range(len(grid_1d)):
            for j in range(len(grid_1d)):
                thetas[((i*len(grid_1d)) + j), ] = np.array((grid_1d[i], grid_1d[j]))
        thetas = thetas[~np.all(thetas == 0, axis=1)] # Remove any row with all zeros
        thetas = (thetas.T / np.apply_along_axis(np.linalg.norm, 1, thetas)).T # Take unit norm

        return np.unique(thetas.round(decimals=4), axis=0)
    '''
   

    def generate_candidates(self, sample_ld, net_weights):
        """
        Method to generate candidate distributions from one dimensional ones.

        Args:
        sample_ld: Array(float, num x ld). 
        net_weights: Array(float, L x k).

        Returns:
        List(DiscreteRV_HD). Atoms are ld dimensional. 
        Choices for our estimated distributions in the ld space.
        """
        
        mat_centers = np.empty(shape = (self.k, self.ld))
        nw = len(net_weights)

        for j in range(self.ld):
            dmm = DMM(k=self.k, sigma=self.sigma)
            est = dmm.estimate(sample_ld[:, j])
            mat_centers[:, j] = est.centers

        A = list(itertools.product(*mat_centers.T))
        S_centers = list(itertools.combinations(A, self.k))
        candidate_ests = [None] * (len(S_centers) * len(net_weights))

        for i in range(len(S_centers)):
            for j in range(len(net_weights)):
                candidate_ests[(i*nw) + j] = DiscreteRV_HD(net_weights[j], S_centers[i])

        return candidate_ests


    def generate_candidates_mle(self, sample_ld):
        """
        Method to generate candidate distributions from one dimensional ones via maximum likelihood selection of weights.

        Args:
        sample_ld: Array(float, num x ld). 

        Returns:
        List(DiscreteRV_HD). Atoms are ld dimensional. 
        Choices for our estimated distributions in the ld space.
        """
              
        mat_centers = np.empty(shape = (self.k, self.ld))
        n, d = sample_ld.shape
        for j in range(self.ld):
            dmm = DMM(k=self.k, sigma=self.sigma)
            est = dmm.estimate(sample_ld[:, j])
            mat_centers[:, j] = est.centers

            A = list(itertools.product(*mat_centers.T))
            S_centers = list(itertools.combinations(A, self.k))
            candidate_ests = [None] * (len(S_centers))
            S_centers_arr = np.asarray(S_centers)

        for i in range(len(S_centers_arr)):
            atoms = S_centers_arr[i, ]
            M = np.exp(-np.square(pairwise_distances(sample_ld, atoms))/2/(self.sigma**2))
            M /= M.sum(axis=1)[:, np.newaxis]
            p = M.sum(axis=0)/n
            candidate_ests[i] = DiscreteRV_HD(p, S_centers[i])

        return candidate_ests


    def generate_theta_ests(self, sample_ld, net_thetas):
        """
        Method to perform one dimensional DMM for data projected along net of unit vectors in ld.

        Args:
        sample_ld: Array(float, num x ld).
        net_thetas: Array(float, size x ld).

        Return object:
        List(DiscreteRV_HD). Atoms are one dimensional.
        """
        
        sample_theta = np.matmul(sample_ld, net_thetas.T)
        theta_ests = [None] * len(net_thetas)

        for j in range(len(net_thetas)):
            dmm = DMM(k=self.k, sigma=self.sigma)
            est = dmm.estimate(sample_theta[:, j])
            theta_ests[j] = DiscreteRV_HD(est.weights, est.centers)

        return theta_ests


    def estimate_ld(self, sample_ld, factor_weights, factor_thetas):
        """
        Method to perform the multivariate denoised method of moments, for any dimension ld.

        Args:
        sample_ld: Array(float, n x ld).
        factor_weights: Scalar(float).
        factor_thetas: Scalar(float).

        Returns:
        Object of class DiscreteRV_HD. Atoms are ld dimensional.
        """

        num = len(sample_ld)
        
        net_weights = self.generate_net_weights(num, factor_weights)
        net_thetas = self.generate_net_thetas(num, factor_thetas)    
        nt = len(net_thetas)

        candidate_ests = self.generate_candidates(sample_ld, net_weights)
        theta_ests = self.generate_theta_ests(sample_ld, net_thetas)

        candidate_ests_theta = [None] * len(candidate_ests) * nt
        errors_candidate_ests = np.empty(shape = (len(candidate_ests), nt))

        for i in range(len(candidate_ests)):
            for j in range(nt):
                weights = candidate_ests[i].weights
                atoms = np.matmul(candidate_ests[i].atoms, net_thetas[j].T)
                candidate_ests_theta[(i * nt) + j] = DiscreteRV_HD(weights, atoms)
                errors_candidate_ests[i, j] = wass_hd(candidate_ests_theta[(i * nt) + j], theta_ests[j])

        max_error = np.max(errors_candidate_ests, axis=1)
        est_selected = candidate_ests[np.argmin(max_error)]

        return est_selected


    def estimate_ld_mle(self, sample_ld):
        """
        Method to perform the multivariate denoised method of moments, for any dimension ld.

        Args:
        sample_ld: Array(float, n x ld).

        Returns:
        Object of class DiscreteRV_HD. Atoms are ld dimensional.
        """

        candidate_ests = self.generate_candidates_mle(sample_ld)
        llhood = np.empty(len(candidate_ests))
        for i in range(len(candidate_ests)):
            atoms = candidate_ests[i].atoms
            weights = candidate_ests[i].weights
            M = np.exp(-np.square(pairwise_distances(sample_ld, atoms))/2/(self.sigma**2)) * weights
            llhood[i] = np.sum(np.log(M.sum(axis=1)))

        return candidate_ests[np.argmax(llhood)]


    def estimate(self, sample, factor_weights, factor_thetas, MLE):
        """
        Method to perform the complete high dimensional DMM algorithm.

        Args:
        sample: Array(float, num x d). Sample from d-dim Gaussian mixture model.
           If d <= ld, will run low-dim DMM. 
        factor_weights: Scalar(float).
        factor_thetas: Scalar(float).
        MLE: Boolean. True if you want to do the MLE method for weights.

        Returns:
        Object of class DiscreteRV_HD. Atoms are d dimensional. 
        
        """

        num = sample.shape[0]
        d = sample.shape[1]
        
        if d > self.ld:
            U_ld = self.estimate_center_space(sample)
            sample_ld = np.matmul(sample, U_ld)
        else:
            sample_ld = sample

        if self.ld == 1: # In this case, put things in the correct format to run DMM() instead of DMM_HD()
            dmm = DMM(k=self.k, sigma=self.sigma)
            sample_ld = sample_ld.reshape(num, )
            est_ld = dmm.estimate(sample_ld) # Returns object of class ModelGM, with est_ld.weights and est_ld.centers
            est_ld_atoms = est_ld.centers.reshape(self.k, self.ld)

        else:
            if (MLE):
                est_ld = self.estimate_ld_mle(sample_ld)
            else:
                est_ld = self.estimate_ld(sample_ld, factor_weights, factor_thetas) # Returns object of class DiscreteRV_HD, with est_ld.weights and est_ld.atoms
            est_ld_atoms = est_ld.atoms

        if d > self.ld:
            est = DiscreteRV_HD(est_ld.weights, np.matmul(est_ld_atoms, U_ld.T))
        else:
            est = DiscreteRV_HD(est_ld.weights, est_ld_atoms)

        return est






def simplex_grid(m, n):
    """
    Method to construct a grid on the unit (m-1) dimensional simplex. 
    Adapted from https://github.com/oyamad/simplex_grid/blob/master/simplex_grid.py


    Args:
    m: Scalar(int). Dimension of simplex is m-1. Will be k. 
    n: Scalar(int). Size of simplex along each coordinate (inverse fineness of simplex). Will be rate_inverse.

    Returns:
    Array(float, L x m). L = (n+m-1) choose (m-1).  
    """
    
    L = comb(n+m-1, m-1, exact=True)
    out = np.empty((L, m), dtype=float)

    x = np.zeros(m, dtype=int)
    x[m-1] = n

    for j in range(m):
        out[0, j] = x[j]

    h = m

    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[m-1] = val - 1
        x[h-1] += 1

        for j in range(m):
            out[i, j] = x[j]

        if val != 1:
            h = m

    return out

































