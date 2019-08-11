# High-dimensional-mixture
This is a Python3 implementation of the High Dimensional Denoised Method of Moments algorithm of CITE. The algorithm builds on the one dimensional [Denoised Method of Moments algorithm](https://arxiv.org/abs/1807.07237). This code builds on the [implementation](https://github.com/Albuso0/mixture) of the one dimensional algorithm.

A sample script to test the algorithm is below.

## External dependencies:

[Numpy](http://numpy.org/)

[CVXPY](https://www.cvxpy.org)

[CVXOPT](http://cvxopt.org)

[POT](https://pot.readthedocs.io/en/stable/)

## Example:

```
from dmm_hd import *
from model_gm import *
from discrete_rv import *

# Set the model parameters for a test run
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
factor = 1.0 # factor to increase size of weights net
model = ModelGM_HD(w=weights, x=x, std=sigma)
sample = sample_gm(model, k, num, d)

# Run the algorithm on this sample
alg = DMM_HD(k_est, ld_est, 1.0) 
v_rv = alg.estimate(sample, factor)
print(wass_hd(u_rv, v_rv))
```
