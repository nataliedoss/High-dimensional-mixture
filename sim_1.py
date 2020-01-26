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
# Simulation study: n varies
def sim_over_n(num_sims, k, ld, d, factor_model, sigma, n_range,
               factor_weights, factor_thetas, MLE,
               max_iter_EM, tol_EM, name):
    k_est = k
    ld_est = ld

    error_mat_dmm = np.zeros(shape = (len(n_range), num_sims))
    error_mat_em = np.zeros(shape = (len(n_range), num_sims))
    time_mat_dmm = np.zeros(shape = (len(n_range), num_sims))
    time_mat_em = np.zeros(shape = (len(n_range), num_sims))

    # Standard normal model (no mixture):
    #x = np.zeros(k*d).reshape(k, d)

    # Symmetric unit sphere model (k = 2 or k = 3):
    x1 = np.random.multivariate_normal(np.zeros(d), np.identity(d), 1)
    x1 = (x1 / np.linalg.norm(x1)).reshape(d, )
    x1 = factor_model * x1
    x2 = -x1
    x3 = np.repeat(0, d)
    x = np.array((x1, x2, x3))

    # Controllable unit sphere model (k = 2 or k = 3):
    #x1 = np.repeat(1/np.sqrt(d), d)
    #x1 = factor_model * x1
    #x2 = -x1
    #x3 = np.concatenate([np.repeat(1/np.sqrt(d), d/2), np.repeat(-1/np.sqrt(d), d/2)])
    #x3 = factor_model * x3
    #x = np.array((x1, x2, x3))
    
    # Unit sphere model:
    #x = np.random.multivariate_normal(np.zeros(k*d), np.identity(k*d), 1).reshape(k, d)
    #x = x / (np.apply_along_axis(np.linalg.norm, 1, x))[:, None]
    #x = factor_model * x
    
    # Uniform between hypercube points model:
    #x = np.random.uniform(-1.0/np.sqrt(d), 1.0/np.sqrt(d), k*d).reshape(k, d)
    
    # Uniform on hypercube model:
    #x = np.asarray(random.choices([1/np.sqrt(d), -1/np.sqrt(d)], k=k*d)).reshape(k, d)

    weights = np.repeat(1.0/k, k)
    #weights = np.array((0.25, 0.75))
    #weights = np.random.dirichlet(np.repeat(1.0, k), 1).reshape(k, )
    
    # If you want the true model to be centered:
    #x_centered = x - np.average(x, axis=0, weights=weights)

    u_rv = DiscreteRV_HD(weights, x) # true model
    model = ModelGM_HD(w=weights, x=x, std=sigma)
    

    # There is only one model for all the simulations; save it:
    np.savetxt("sim_csv/weights_" + name + ".csv", model.weights)
    np.savetxt("sim_csv/centers_" + name + ".csv", model.centers)

    # Save a plot of the model:
    sample_tmp = sample_gm(model, k, 1000, d)
    plt.scatter(sample_tmp[:, 0], sample_tmp[:, 1])
    plt.savefig("sample_pic.pdf")
    plt.close()
    
    for i in range(len(n_range)):
        num = n_range[i]
        for j in range(num_sims):
            sample = sample_gm(model, k, num, d)
            dmm_hd = DMM_HD(k_est, ld_est, sigma)
            start_dmm = time.time()
            mean_est = np.mean(sample, axis=0)
            sample_centered = sample - mean_est
            v_rv_dmm = dmm_hd.estimate(sample_centered, factor_weights, factor_thetas, MLE)
            v_rv_dmm.atoms = v_rv_dmm.atoms + mean_est
            end_dmm = time.time()

            '''
            # Run EM with package
            # The option for keeping cov same across clusters is "tied'
            em = GaussianMixture(n_components=k, covariance_type='tied',
                     tol=tol_EM, max_iter=max_iter_EM,
                     n_init=1, init_params='random',
                     weights_init=None, means_init=None, precisions_init=None,
                     random_state=None, warm_start=False, verbose=1, verbose_interval=10)
            start_em = time.time()
            em.fit(sample)
            end_em = time.time()
            v_rv_em = DiscreteRV_HD(em.weights_, em.means_)
            end_em = time.time()
            '''

            # Run our EM implementation
            start_em = time.time()
            p, mu = em(sample, k, sigma, max_iter_EM, tol_EM)
            v_rv_em = DiscreteRV_HD(p, mu)
            end_em = time.time()
            

            '''            
            # Run our symmetric, 2-GM EM implementation
            start_em = time.time()
            mu = em_symm(sample, sigma, max_iter_EM, tol_EM)
            end_em = time.time()
            v_rv_em = DiscreteRV_HD(weights, mu) # Plug in the true weights, which must be (1/2, 1/2)
            '''
            
            
            error_mat_dmm[i, j] = wass_hd(u_rv, v_rv_dmm)
            error_mat_em[i, j] = wass_hd(u_rv, v_rv_em)
            time_mat_dmm[i, j] = end_dmm - start_dmm
            time_mat_em[i, j] = end_em - start_em

            print("Error from HD DMM:", error_mat_dmm[i, j])
            print("Error from EM:", error_mat_em[i, j])
            print("Time from HD DMM:", time_mat_dmm[i, j])
            print("Time from EM:", time_mat_em[i, j])

    # Compute mean and sd of each row (as n changes)
    error_mean_dmm = np.mean(error_mat_dmm, axis = 1)
    error_sd_dmm = np.std(error_mat_dmm, axis = 1)
    error_mean_em = np.mean(error_mat_em, axis = 1)
    error_sd_em = np.std(error_mat_em, axis = 1)

    time_mean_dmm = np.mean(time_mat_dmm, axis = 1)
    time_sd_dmm = np.std(time_mat_dmm, axis = 1)
    time_mean_em = np.mean(time_mat_em, axis = 1)
    time_sd_em = np.std(time_mat_em, axis = 1)

    np.savetxt("sim_csv/dmmerrormean_" + name + ".csv", error_mean_dmm)
    np.savetxt("sim_csv/dmmerrorsd_" + name + ".csv", error_sd_dmm)
    np.savetxt("sim_csv/emerrormean_" + name + ".csv", error_mean_em)
    np.savetxt("sim_csv/emerrorsd_" + name + ".csv", error_sd_em)
    np.savetxt("sim_csv/dmmtimemean_" + name + ".csv", time_mean_dmm)
    np.savetxt("sim_csv/dmmtimesd_" + name + ".csv", time_sd_dmm)
    np.savetxt("sim_csv/emtimemean_" + name + ".csv", time_mean_em)
    np.savetxt("sim_csv/emtimesd_" + name + ".csv", time_sd_dmm)

    return([error_mean_dmm, error_sd_dmm, error_mean_em, error_sd_em,
            time_mean_dmm, time_sd_dmm, time_mean_em, time_sd_em])



####################################################################
# Run sim study
k = 3
ld = k-1
d = 100
factor_model = 2
num_sims = 10
sigma = 1
n_range = np.arange(10000, 200000, 10000)
factor_weights = 1
factor_thetas = 4
MLE = False
max_iter_EM = 1000
tol_EM = .000001
name = "k" + str(k) + "_d" + str(d) + "_factormodel" + str(factor_model) + "_weightseven" + "_sigma" + str(sigma) +  "_factorweights" + str(factor_weights) +  "_factorthetas" + str(factor_thetas) 


# Quick check of size of theta_net
num_test = n_range[0]
dmm_hd = DMM_HD(3, 2, sigma)
rate_inverse = dmm_hd.compute_rate_inverse(num_test)
net_weights = dmm_hd.generate_net_weights(num_test, factor_weights)
net_thetas = dmm_hd.generate_net_thetas(num_test, factor_thetas)
print(rate_inverse)
print(net_weights)
print(net_thetas.shape)
print(net_thetas)
plt.scatter(net_thetas[:, 0], net_thetas[:, 1])
plt.show()




# Run sim
random.seed(3)
sim = sim_over_n(num_sims=num_sims, k=k, ld=ld, d=d, factor_model=factor_model, 
                 sigma=sigma, n_range=n_range,
                 factor_weights=factor_weights,
                 factor_thetas=factor_thetas,
                 MLE=MLE, max_iter_EM=max_iter_EM, tol_EM=tol_EM, name=name)





# Plots
SMALL_SIZE = 14
MEDIUM_SIZE = 20
# Accuracy
plt.plot()
p1 = plt.errorbar(n_range/1000, sim[0], sim[1])
p2 = plt.errorbar(n_range/1000, sim[2], sim[3])
plt.ylim(0.0, 1.0)
#plt.title("Accuracy as n grows")
plt.xlabel("n/1000")
plt.ylabel("Wasserstein-1")
plt.rc('font', size=SMALL_SIZE) 
plt.rc('axes', titlesize=SMALL_SIZE)     
plt.rc('axes', labelsize=SMALL_SIZE)    
#plt.rc('xtick', labelsize=SMALL_SIZE)    
#plt.rc('ytick', labelsize=SMALL_SIZE)   
plt.rc('legend', fontsize=MEDIUM_SIZE)    
plt.legend((p1, p2), ("DMM", "EM"), loc='upper right', shadow=True)
plt.savefig("sim_overn_" + name + "_mean.pdf")
plt.close()

# Time
plt.plot()
p1 = plt.scatter(n_range/1000, sim[4])
p2 = plt.scatter(n_range/1000, sim[6])
plt.title("Time as n grows")
plt.xlabel("n/1000")
plt.ylabel("Time in seconds")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
plt.tight_layout()
plt.savefig("sim_overn_" + name + "_time.pdf")
plt.close()





'''
# Two figures in subplots
# Accuracy
plt.subplot(1, 2, 1)
#plt.plot()
p1 = plt.errorbar(n_range/1000, sim[0], sim[1])
p2 = plt.errorbar(n_range/1000, sim[2], sim[3])
plt.title("Accuracy as n grows")
plt.xlabel("n/1000")
plt.ylabel("Wasserstein-1")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper right', shadow=True)
# Time
plt.subplot(1, 2, 2)
p1 = plt.scatter(n_range/1000, sim[4])
p2 = plt.scatter(n_range/1000, sim[6])
plt.title("Time as n grows")
plt.xlabel("n/1000")
plt.ylabel("Time in seconds")
plt.legend((p1, p2), ("DMM", "EM"), loc='upper left', shadow=True)
#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
plt.tight_layout()
plt.savefig("sim_overn_" + name + ".pdf")
plt.close()
'''



'''
# Another way to format
f = plt.figure(figsize=(10,3))
p1 = f.add_subplot(121)
#p2 = f.add_subplot(122)
p1.errorbar(n_range/1000, sim[0], sim[1])
p1.errorbar(n_range/1000, sim[2], sim[3])
#p1.title("Accuracy as n grows")
#p1.xlabel("n/1000")
#p1.ylabel("Wasserstein-1")
#p1.legend(p1, ("DMM", "EM"), loc='upper left', shadow=True)
f.savefig("temp.pdf")
'''

