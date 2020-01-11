from dmm_hd import *
from model_gm import *
from discrete_rv import *
from em import *

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
import random




###############################################################
# Simulation study: d varies
def sim_over_d(num_sims, k, ld, num, sigma, d_range, factor_weights, factor_thetas, niter_EM):
    k_est = k
    ld_est = ld

    error_mat_dmm = np.zeros(shape = (len(d_range), num_sims))
    error_mat_em = np.zeros(shape = (len(d_range), num_sims))
    time_mat_dmm = np.zeros(shape = (len(d_range), num_sims))
    time_mat_em = np.zeros(shape = (len(d_range), num_sims))

    for i in range(len(d_range)):
        d = d_range[i]
        # Standard normal model (no mixture):
        #x = np.zeros(k*d).reshape(k, d)
        
        # Unit sphere model:
        #x = np.random.multivariate_normal(np.zeros(k*d), np.identity(k*d), 1).reshape(k, d)
        #x = x / (np.apply_along_axis(np.linalg.norm, 1, x))[:, None]

        # Controllable unit sphere model
        #x1 = np.repeat(1/np.sqrt(d), d)
        #x2 = -x1
        #x3 = np.concatenate([np.repeat(1/np.sqrt(d), d/2), np.repeat(-1/np.sqrt(d), d/2)])
        #x = np.array((x1, x2, x3))
        
        # Symmetric unit sphere model (k = 2):
        x1 = np.random.multivariate_normal(np.zeros(d), np.identity(d), 1)
        x1 = (x1 / np.linalg.norm(x1)).reshape(d, )
        x1 = 2*x1
        x2 = -x1
        x = np.array((x1, x2))
        
        # Uniform between hypercube points model:
        #x = np.random.uniform(-1.0/np.sqrt(d), 1.0/np.sqrt(d), k*d).reshape(k, d)
        
        # Uniform on hypercube model:
        #x = np.asarray(random.choices([1/np.sqrt(d), -1/np.sqrt(d)], k=k*d)).reshape(k, d)

        weights = np.repeat(1.0/k, k)
        #weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
        
        # If you want the true model to be centered:
        #x_centered = x - np.average(x, axis=0, weights=weights)

        u_rv = DiscreteRV_HD(weights, x) # true model
        model = ModelGM_HD(w=weights, x=x, std=sigma)


        # Just for me, save a plot of the model for a single d
        #sample_tmp = sample_gm(model, k, 100000, d)
        #plt.scatter(sample_tmp[:, 0], sample_tmp[:, 1])
        #plt.savefig("sample_tmp.pdf")
        #plt.close()

        for j in range(num_sims):

            sample = sample_gm(model, k, num, d)

            # Run HD DMM
            dmm_hd = DMM_HD(k_est, ld_est, sigma)
            start_dmm = time.time()
            mean_est = np.mean(sample, axis=0)
            sample_centered = sample - mean_est
            v_rv_dmm = dmm_hd.estimate(sample_centered, factor_weights, factor_thetas)
            v_rv_dmm.atoms = v_rv_dmm.atoms + mean_est
            end_dmm = time.time()

            '''
            # Run EM with package
            # The option for keeping cov same across clusters is "tied'
            # But it seems like the algorithm will still estimate it
            # And it won't be restricted to be spherical
            # It doesn't seem possible to simply input the true covariance
            
            em = GaussianMixture(n_components= k, covariance_type = 'spherical',
                                 max_iter = niter_EM, random_state = 1)
            start_em = time.time()
            em.fit(sample)
            end_em = time.time()
            v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
            end_em = time.time()
            '''

            # Run our EM implementation
            start_em = time.time()
            p, mu = em(sample, k, sigma=sigma, iter=niter_EM)
            v_rv_em = DiscreteRV_HD(p, mu)
            end_em = time.time()

           
            error_mat_dmm[i, j] = wass_hd(u_rv, v_rv_dmm)
            error_mat_em[i, j] = wass_hd(u_rv, v_rv_em)
            time_mat_dmm[i, j] = end_dmm - start_dmm
            time_mat_em[i, j] = end_em - start_em

    # Compute mean and sd of each row (as d changes)
    error_mean_dmm = np.mean(error_mat_dmm, axis = 1)
    error_sd_dmm = np.std(error_mat_dmm, axis = 1)
    error_mean_em = np.mean(error_mat_em, axis = 1)
    error_sd_em = np.std(error_mat_em, axis = 1)

    time_mean_dmm = np.mean(time_mat_dmm, axis = 1)
    time_mean_em = np.mean(time_mat_em, axis = 1)

    return([error_mean_dmm, error_sd_dmm, error_mean_em, error_sd_em,
            time_mean_dmm, time_mean_em])





####################################################################
# Run sim study
k = 2
ld = k-1
num = 1000
num_sims = 10
sigma = 1.0
d_range = np.arange(5, 50, 5)
factor_weights = 1.0
factor_thetas = 0.2
niter_EM = 1000

# Quick check of size of theta_net
dmm_hd = DMM_HD(k, ld, sigma)
rate_inverse = dmm_hd.compute_rate_inverse(num)
grid_1d = np.arange(-1, 1.1, 1.0/(factor_thetas * rate_inverse))
net_weights = dmm_hd.generate_net_weights(num, factor_weights)
print(rate_inverse)
print(grid_1d)
print(net_weights)

# Run sim 
random.seed(3)
sim = sim_over_d(num_sims=num_sims, k=k, ld=ld, num=num,
                 sigma=sigma, d_range=d_range,
                 factor_weights=factor_weights,
                 factor_thetas=factor_thetas,
                 niter_EM=niter_EM)



# Plot

# Accuracy
plt.subplot(1, 2, 1)
#plt.plot()
p1 = plt.errorbar(d_range, sim[0], sim[1])
p2 = plt.errorbar(d_range, sim[2], sim[3])
plt.title("Accuracy as d grows")
plt.xlabel("d")
plt.ylabel("Wasserstein-1")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
# Time
plt.subplot(1, 2, 2)
p1 = plt.scatter(d_range, sim[4])
p2 = plt.scatter(d_range, sim[5])
plt.title("Time as d grows")
plt.xlabel("d")
plt.ylabel("Time in seconds")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.tight_layout()
plt.savefig("sim.pdf")
plt.close()

