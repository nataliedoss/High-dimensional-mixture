from dmm_hd import DMM_HD
from model_gm import ModelGM_HD, ModelGM, sample_gm
from discrete_rv import DiscreteRV_HD, wass_hd

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
import random




###############################################################
# Simulation study: d varies
def sim_over_d(num_sims, k, ld, num, sigma, d_range, factor_weights, factor_thetas):
    k_est = k
    ld_est = ld

    error_mat_dmm = np.zeros(shape = (len(d_range), num_sims))
    error_mat_em = np.zeros(shape = (len(d_range), num_sims))
    time_mat_dmm = np.zeros(shape = (len(d_range), num_sims))
    time_mat_em = np.zeros(shape = (len(d_range), num_sims))

    for i in range(len(d_range)):
        d = d_range[i]
        # One model:
        x = np.zeros(k*d).reshape(k, d)
        # One model:
        #x = np.random.multivariate_normal(np.zeros(k*d), np.identity(k*d), 1).reshape(k, d)
        #for j in range(k):
        #    x[j, ] = x[j, ] / np.linalg.norm(x[j, ])
        # One model:
        #x1 = np.random.multivariate_normal(np.zeros(d), np.identity(d), 1)
        #x1 = x1 / np.apply_along_axis(np.linalg.norm, 1, x1)
        #x1 = x1.T
        #x1 = x1.reshape(d, )
        #x2 = -x1
        #x = np.array((x1, x2))
        # One model:
        #x = np.random.uniform(-1.0/np.sqrt(d), 1.0/np.sqrt(d), k*d).reshape(k, d)
        # One model:
        #x = np.asarray(random.choices([1/np.sqrt(d), -1/np.sqrt(d)], k=k*d)).reshape(k, d)

        weights = np.repeat(1.0/k, k)
        #weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
        # If you want the true model to be centered:
        #x_centered = x - np.average(x, axis=0, weights=weights)
        
        u_rv = DiscreteRV_HD(weights, x) # true model
        model = ModelGM_HD(w=weights, x=x, std=sigma)

        for j in range(num_sims):

            sample = sample_gm(model, k, num, d)
            mean_est = np.mean(sample, axis=0)
            sample = sample - mean_est

            dmm_hd = DMM_HD(k_est, ld_est, sigma)
            start_dmm = time.time()
            v_rv_dmm = dmm_hd.estimate(sample, factor_weights, factor_thetas)
            v_rv_dmm.atoms = v_rv_dmm.atoms + mean_est
            end_dmm = time.time()

            em = GaussianMixture(n_components= k, covariance_type = 'spherical',
                                 max_iter = 20, random_state = 1)
            start_em = time.time()
            em.fit(sample)
            end_em = time.time()
            v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
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
num_sims = 10
num = 1000
sigma = 1.0
d_range = np.arange(10, 100, 10)
factor_weights = 1.0
factor_thetas = 1.0

# Quick check of size of theta_net
dmm_hd = DMM_HD(3, 2, sigma)
rate_inverse = dmm_hd.compute_rate_inverse(num)
grid_1d = np.arange(-1, 1.1, 1.0/(factor_thetas * rate_inverse))
net_weights = dmm_hd.generate_net_weights(num, factor_weights)
print(rate_inverse)
print(grid_1d)
print(net_weights)


sim = sim_over_d(num_sims=num_sims, k=2, ld=1, num=num,
                 sigma=sigma, d_range=d_range,
                 factor_weights=factor_weights, factor_thetas=factor_thetas)


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
plt.savefig("sim.pdf")
plt.close()

