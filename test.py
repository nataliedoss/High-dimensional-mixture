#################################################################################
# Script for testing 
#################################################################################


from dmm_hd import DMM_HD   
from model_gm import ModelGM_HD, ModelGM, sample_gm
from discrete_rv import DiscreteRV_HD, wass_hd, wass

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time


# Set the model parameters.
d = 10
k = 3
k_est = k
ld_est = k_est - 1
x1 = np.repeat(2, d)
x2 = np.repeat(-2, d)
x3 = np.repeat(0, d)
x = np.array((x1, x2, x3))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # truth
num = 1000 # sample size
sigma = 0.5
factor = 1.0


# Generate a sample from this model.
model = ModelGM_HD(w=weights, x=x, std=sigma)
sample = sample_gm(model, k, num, d)


# Plot the sample.
plt.scatter(sample[:, 0], sample[:, 1])
plt.show()


# Run the high dimensional DMM on this sample.
alg = DMM_HD(k_est, ld_est, 1.0)
start_dmm = time.time()
v_rv = alg.estimate(sample, factor)
end_dmm = time.time()
print(end_dmm - start_dmm)
print(wass_hd(u_rv, v_rv))


# Run EM on this sample.
em = GaussianMixture(n_components = k, covariance_type = 'spherical',
                     max_iter = 100, random_state = 1)
start_em = time.time()
em.fit(sample)
end_em = time.time()
print(end_em - start_em)
v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
print(wass_hd(u_rv, v_rv_em))



#################################################################################
# Check the function for Wasserstein-1 distance in d dimensions.
# It should equal the wass function when d = 1.
u_rv = DiscreteRV_HD((0.5, 0.5), (1, -1))
v_rv = DiscreteRV_HD((0.4, 0.6), (.9, -1.2))
wass(u_rv, v_rv)
wass_hd(u_rv, v_rv)


# TESTS
U_ld = alg.estimate_center_space(sample)
sample_ld = np.matmul(sample, U_ld)
net_weights = alg.generate_net_weights(num, factor)
net_thetas = alg.generate_net_thetas(num)
tmp = alg.estimate_ld(sample_ld, factor)


