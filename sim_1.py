from dmm_hd import DMM_HD
from model_gm import ModelGM_HD, ModelGM, sample_gm
from discrete_rv import DiscreteRV_HD, wass_hd

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time




###############################################################
# Simulation study: n varies
def sim_over_n(num_sims, k, ld, d, sigma, n_range, factor_weights, factor_thetas):
    k_est = k
    ld_est = ld

    error_mat_dmm = np.zeros(shape = (len(n_range), num_sims))
    error_mat_em = np.zeros(shape = (len(n_range), num_sims))
    time_mat_dmm = np.zeros(shape = (len(n_range), num_sims))
    time_mat_em = np.zeros(shape = (len(n_range), num_sims))

    # A simple model:
    x1 = np.repeat(1, d)
    x2 = np.repeat(-1, d)
    x = np.array((x1, x2))
    weights = np.repeat(1.0/k, k)
    u_rv = DiscreteRV_HD(weights, x) # true model
    model = ModelGM_HD(w=weights, x=x, std=sigma)
    
    for i in range(len(n_range)):
        num = n_range[i]
        
        for j in range(num_sims):
            
            sample = sample_gm(model, k, num, d)

            dmm_hd = DMM_HD(k_est, ld_est, sigma)
            start_dmm = time.time()
            v_rv_dmm = dmm_hd.estimate(sample, factor_weights, factor_thetas)
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

    # Compute mean and sd of each row (as n changes)
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
d = 500
num_sims = 10
sigma = 1.0
n_range = np.arange(10000, 20000, 1000)
factor_weights = 10.0
factor_thetas = 0.2


sim_k2 = sim_over_n(num_sims=num_sims, k=2, ld=2, d=d,
                    sigma=sigma, n_range=n_range,
                    factor_weights=factor_weights, factor_thetas=factor_thetas)



# Plots
# Accuracy
plt.subplot(1, 2, 1)
#plt.plot()
p1 = plt.errorbar(n_range, sim_k2[0], sim_k2[1])
p2 = plt.errorbar(n_range, sim_k2[2], sim_k2[3])
plt.title("k = 2: Accuracy as n grows")
plt.xlabel("n")
plt.ylabel("Wasserstein-1")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
# Time
plt.subplot(1, 2, 2)
p1 = plt.scatter(n_range, sim_k2[4])
p2 = plt.scatter(n_range, sim_k2[5])
plt.title("k = 2: Time as n grows")
plt.xlabel("n")
plt.ylabel("Time")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
plt.savefig("sim_k2.pdf")
plt.close()


