#################################################################################
# Script for testing 
#################################################################################


from dmm_hd import *
from model_gm import *
from discrete_rv import *

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time


# Set the model parameters.
d = 100
k = 3
k_est = k
ld_est = k_est - 1
x1 = np.repeat(1, d)
x2 = np.repeat(-1, d)
x3 = np.repeat(0, d)
x = np.array((x1, x2, x3))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # truth
num = 500 # sample size
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
print("The time to run HD DMM on this sample was", end_dmm-start_dmm)
print("The error from HD DMM was", wass_hd(u_rv, v_rv))


# Run EM on this sample.
em = GaussianMixture(n_components = k, covariance_type = 'spherical',
                     max_iter = 100, random_state = 1)
start_em = time.time()
em.fit(sample)
end_em = time.time()
v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
print("The time to run EM on this sample was", end_em - start_em)
print("The error from EM was", wass_hd(u_rv, v_rv_em))



#################################################################################
# Test some functions in the HD DMM algorithm

U_ld = alg.estimate_center_space(sample)
sample_ld = np.matmul(sample, U_ld)
net_weights = alg.generate_net_weights(num, factor)
candidate_ests = alg.generate_candidates(sample_ld, net_weights)

#net_weights = alg.generate_net_weights(num, factor)
#net_thetas = alg.generate_net_thetas(num)
#tmp = alg.estimate_ld(sample_ld, factor)

# In this case, there are (9 choose 6) candidate estimates
# And a weights net of size 6
# Total number of candidate estimates is 504.
# Computing W_1 distance for each of these against each projected estimate is
# what causes algorithm slowness.
# It's not anything else (wass_hd isn't slow, etc, I checked).




#################################################################################
# Check the function for Wasserstein-1 distance in d dimensions.
u_rv = DiscreteRV_HD((0.5, 0.5), (1, -1))
v_rv = DiscreteRV_HD((0.3, 0.7), (2, -1.2))
# The following two should be the same:
print(wass(u_rv, v_rv))
print(wass_hd(u_rv, v_rv))

