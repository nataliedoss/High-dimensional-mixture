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
random.seed(21)
d = 2
k = 3
k_est = k
ld_est = k-1
sigma = 1.0



# Standard normal model (no mixture):
#x = np.zeros(k*d).reshape(k, d)
# Unit sphere model:
#x = np.random.multivariate_normal(np.zeros(k*d), np.identity(k*d), 1).reshape(k, d)
#x = x / (np.apply_along_axis(np.linalg.norm, 1, x))[:, None]
# Symmetric unit sphere model (k = 2):
#x1 = np.random.multivariate_normal(np.zeros(d), np.identity(d), 1)
#x1 = (x1 / np.linalg.norm(x1)).reshape(d, )
#x2 = -x1
#x3 = np.repeat(0, d)
#x = np.array((x1, x2))
# Symmetric +-1 model
x1 = np.repeat(1/np.sqrt(d), d)
x2 = -x1
x3 = np.concatenate([np.repeat(1/np.sqrt(d), d/2), np.repeat(-1/np.sqrt(d), d/2)])
x = np.array((x1, x2, x3))
# Uniform between hypercube points model:
#x = np.random.uniform(-1.0/np.sqrt(d), 1.0/np.sqrt(d), k*d).reshape(k, d)
# Uniform on hypercube model::
#x = np.asarray(random.choices([1/np.sqrt(d), -1/np.sqrt(d)], k=k*d)).reshape(k, d)

weights = np.repeat(1.0/k, k)
#weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
# If you want the true model to be centered:
#x_centered = x - np.average(x, axis=0, weights=weights)

u_rv = DiscreteRV_HD(weights, x) # true model
model = ModelGM_HD(w=weights, x=x, std=sigma)

# Generate a sample
num = 10000
sample = sample_gm(model, k, num, d)


# Plot the sample.
plt.scatter(sample[:, 0], sample[:, 1])
plt.show()



# Algorithm parameters
factor_weights = 1.0
factor_thetas = 0.2
niter_EM = 1000


# Testing grid size
dmm_hd = DMM_HD(k_est, ld_est, sigma)
rate_inverse = dmm_hd.compute_rate_inverse(num)
grid_1d = np.arange(-1, 1.1, 1.0/(factor_thetas * rate_inverse))
net_weights = dmm_hd.generate_net_weights(num, factor_weights)
net_thetas = dmm_hd.generate_net_thetas(num, factor_thetas)
print(rate_inverse)
print(grid_1d)
print(net_weights)
print(net_thetas)




# Run the high dimensional DMM on this sample
alg = DMM_HD(k_est, ld_est, sigma)
start_dmm = time.time()
mean_est = np.mean(sample, axis=0)
sample_centered = sample - mean_est
v_rv = alg.estimate(sample_centered, factor_weights, factor_thetas)
v_rv.atoms = v_rv.atoms + mean_est
end_dmm = time.time()
print("The time to run HD DMM on this sample was", end_dmm-start_dmm)
print("The error from HD DMM was", wass_hd(u_rv, v_rv))



# Run our EM on this sample
start_em = time.time()
p, mu = em(sample, k, sigma=sigma, iter=niter_EM)
end_em = time.time()
v_rv_em = DiscreteRV_HD(p, mu)
print("The time to run EM on this sample was", end_em - start_em)
print("The error from EM was", wass_hd(u_rv, v_rv_em))


'''
# Run EM package on this sample
em = GaussianMixture(n_components = k, covariance_type = 'spherical',
                     max_iter=niter_EM, random_state = 1)
start_em = time.time()
em.fit(sample)
end_em = time.time()
v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
print("The time to run EM on this sample was", end_em - start_em)
print("The error from EM was", wass_hd(u_rv, v_rv_em))

'''







'''

#################################################################################
# Test some functions in the HD DMM algorithm

# Check that DMM works on each coordinate
# And that we generate an S containing a good candidate
mat_centers = np.empty(shape = (alg.k, alg.ld))

for j in range(alg.ld):
    dmm = DMM(alg.k, sigma=None)
    est = dmm.estimate(sample[:, j])
    mat_centers[:, j] = est.centers

A = list(itertools.product(*mat_centers.T))
S_centers = list(itertools.combinations(A, alg.k))


# Define the nets and parameters to do sliced MoM
net_weights = alg.generate_net_weights(num, factor_weights)
net_thetas = alg.generate_net_thetas(num, factor_thetas)
rate_inverse = alg.compute_rate_inverse(num)
nt = len(net_thetas)


# Generate the candidate estimates
candidate_ests = alg.generate_candidates(sample, net_weights)
theta_ests = alg.generate_theta_ests(sample, net_thetas)



# Check that this is indeed what we select
candidate_ests_theta = [None] * len(candidate_ests) * nt
errors_candidate_ests = np.empty(shape = (len(candidate_ests), nt))

for i in range(len(candidate_ests)):
    for j in range(nt):
        weights = candidate_ests[i].weights
        atoms = np.matmul(candidate_ests[i].atoms, net_thetas[j].T)
        candidate_ests_theta[(i * nt) + j] = DiscreteRV_HD(weights, atoms)
        errors_candidate_ests[i, j] = wass_hd(candidate_ests_theta[(i * nt) + j], theta_ests[j])


avg_errors = np.mean(errors_candidate_ests, axis=1)
est_selected = candidate_ests[np.argmin(avg_errors)]
print(np.argmin(avg_errors))
print(est_selected.atoms)
print(est_selected.weights)

# Whereas what you should have selected was:
print(wass_hd(u_rv, est_selected))



    
# When k = 3, there are (9 choose 6) candidate estimates
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
#print(wass(u_rv, v_rv))
#print(wass_hd(u_rv, v_rv))




#################################################################################
# Check the time of DMM for k = 5
d = 1
k = 5
num = 1000
x = np.random.uniform(-1.0, 1.0, k*d).reshape(k, d)
weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
model = ModelGM_HD(w=weights, x=x, std=sigma)
sample_d1 = sample_gm(model, k, num, d)
sample = sample_d1[:,0]
dmm = DMM(k, sigma=None)
start_dmm = time.time()
est = dmm.estimate(sample)
end_dmm = time.time()
print(end_dmm - start_dmm)


'''

