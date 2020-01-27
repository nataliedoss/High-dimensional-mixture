#################################################################################
# Script for testing 
#################################################################################

from dmm_hd import *
from model_gm import *
from discrete_rv import *
from em import *

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import time
import random




#################################################################################
# Set the model parameters.
random.seed(2)
d = 2
k = 3
ld = k-1
sigma = 1.0
factor_model = 2
x1 = np.repeat(1/np.sqrt(d), d)
x1 = factor_model*x1
x2 = -x1
x3 = np.repeat(0, d)
x = np.array((x1, x2, x3))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # true model
model = ModelGM_HD(w=weights, x=x, std=sigma)
# Generate a sample
num = 10000
sample = sample_gm(model, k, num, d)


# Plot the sample.
plt.scatter(sample[:, 0], sample[:, 1])
plt.show()

# Algorithm parameters
factor_weights = 1
factor_thetas = 4
max_iter_EM = 1000
tol_EM = .000001

# Run the high dimensional DMM on this sample
alg = DMM_HD(k, ld, sigma)
print(alg.generate_net_thetas(num, factor_thetas))
start_dmm = time.time()
mean_est = np.mean(sample, axis=0)
sample_centered = sample - mean_est
v_rv = alg.estimate(sample_centered, factor_weights, factor_thetas, False)
v_rv.atoms = v_rv.atoms + mean_est
end_dmm = time.time()
print("The time to run HD DMM on this sample was", end_dmm-start_dmm)
print("The error from HD DMM was", wass_hd(u_rv, v_rv))
print(v_rv.weights)


'''
# A test
rate_inverse = alg.compute_rate_inverse(num)
net_weights = alg.generate_net_weights(num, factor_weights)
tmp = alg.generate_candidates(sample, net_weights)
'''


#################################################################################
# What causes slowness of HD DMM

    
# When k = 3, there are (9 choose 6) candidate estimates
# And a weights net of size 6
# Total number of candidate estimates is 504.
# Computing W_1 distance for each of these against each projected estimate is
# what causes algorithm slowness.
# It's not anything else (wass_hd isn't slow, etc, I checked).



'''
#################################################################################
# Check the function for Wasserstein-1 distance in d dimensions.
u_rv = DiscreteRV_HD((0.5, 0.5), (1, -1))
v_rv = DiscreteRV_HD((0.3, 0.7), (2, -1.2))
# The following two should be the same:
print(wass(u_rv, v_rv))
print(wass_hd(u_rv, v_rv))
'''


#################################################################################
# Check the time of one-dim DMM for k = 5
'''
d = 1
k = 5
interval = 1
sigma = 1.0
num = 1000
x = np.random.uniform(-1.0, 1.0, k*d).reshape(k, d)
weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
model = ModelGM_HD(w=weights, x=x, std=sigma)
sample_d1 = sample_gm(model, k, num, d)
sample = sample_d1[:,0]
dmm = DMM(k=5, sigma=1)
'''






'''
#################################################################################
# Test center space estimation
d = 2
k = 2
simga = 1.0
factor_model = 4.0
x1 = np.repeat(1/np.sqrt(d), d)
x1 = factor_model*x1
x2 = -x1
x = np.array((x1, x2))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # true model
model = ModelGM_HD(w=weights, x=x, std=sigma)

# Generate a sample
num = 1000
sample = sample_gm(model, k, num, d)
alg = DMM_HD(k_est, ld_est, sigma)
dir = alg.estimate_center_space(sample)
print(dir)
print(1/np.sqrt(2))
'''



'''
#################################################################################
# Test centering
d = 2
k = 2
simga = 1.0
factor_model = 4.0
x1 = np.repeat(1/np.sqrt(d), d)
x1 = factor_model*x1
x2 = np.repeat(0, d)
x = np.array((x1, x2))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # true model
model = ModelGM_HD(w=weights, x=x, std=sigma)
# Generate a sample
num = 1000
sample = sample_gm(model, k, num, d)
mean_est = np.mean(sample, axis=0)
sample_centered = sample - mean_est
plt.scatter(sample[:, 0], sample[:, 1])
plt.scatter(sample_centered[:, 0], sample_centered[:, 1])
plt.show()
'''
