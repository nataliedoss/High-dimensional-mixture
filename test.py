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
d = 2
k = 2
k_est = k
ld_est = k_est 
x1 = np.repeat(1, d)
x2 = np.repeat(-1, d)
x = np.array((x1, x2))
weights = np.repeat(1.0/k, k)
u_rv = DiscreteRV_HD(weights, x) # truth
num = 1000 # sample size
sigma = 0.5
factor_weights = 20.0
factor_thetas = 1.0


# Generate a sample from this model.
model = ModelGM_HD(w=weights, x=x, std=sigma)
sample = sample_gm(model, k, num, d)


# Plot the sample.
plt.scatter(sample[:, 0], sample[:, 1])
plt.show()


# Run the high dimensional DMM on this sample.
alg = DMM_HD(k_est, ld_est, 1.0)
start_dmm = time.time()
v_rv = alg.estimate_ld(sample, factor_weights, factor_thetas)
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

