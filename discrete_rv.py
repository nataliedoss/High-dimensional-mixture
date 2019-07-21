"""
module for common operations on discrete random variables (distributions)
"""


import numpy as np
import ot

def assert_shape_equal(x_var, y_var):
    """
    Assert shape equal
    """
    if x_var.shape != y_var.shape:
        raise AssertionError('Shape mismatch!')


class DiscreteRV_HD:
    """ 
    class for d-dimensional finite discrete RV
    """
    
    def __init__(self, w, x):
        """
        w: Array(float, TK x TK). Discrete distribution weights.
        x: Array(float, k x d). Discrete distribution atoms. 
        """
        
        self.weights = np.asarray(w)
        self.atoms = np.asarray(x)
        #assert_shape_equal(self.weights, self.atoms[:,0])
        


class DiscreteRV:
    """ 
    class for 1-d finite discrete RV
    """
    
    def __init__(self, w, x):
        """
        weights: probabilites masses
        atoms: atoms
        """
        self.weights = np.asarray(w)
        self.atoms = np.asarray(x)
        assert_shape_equal(self.weights, self.atoms)

    def __repr__(self):
        return "atom: %s\nwght: %s" % (self.atoms, self.weights)

    def moment(self, degree=1):
        """ 
        Compute the moments of the input RV up to the given degree (start from first degree)

        Args:
        degree: int
        highest degree k

        Returns:
        array of moments from the first degree to degree k
        """
        
        moments = np.zeros(degree)
        monomial = np.ones(self.atoms.shape)

        for i in range(degree):
            monomial *= self.atoms
            moments[i] = np.dot(self.weights, monomial)

        return moments





#################################################################################


def wass_hd(u_rv, v_rv):
    """
    Compute W1 distance between d-dim DiscreteRVs U and V.

    Args:
    u_rv, v_rv: objects of class DiscreteRV_HD.

    Returns: 
    W1 distance: Float.

    """
    
    Ku = len(u_rv.atoms)
    Kv = len(v_rv.atoms)

    C = np.zeros((Ku, Kv))
    for i in range(Ku):
        for j in range(Kv):
            C[i, j] = np.linalg.norm(u_rv.atoms[i, ] - v_rv.atoms[j, ])
            
    gamma = ot.emd(u_rv.weights, v_rv.weights, C)
    return np.sum(gamma * C)


    
def wass(u_rv, v_rv):
    """
    compute W1 distance between 1-dim DiscreteRVs U and V

    Args:
    u_rv, v_v: objects of class DiscreteRV.

    Returns: 
    W1 distance: Float.
    """
    
    if len(u_rv.atoms) == 0 or len(v_rv.atoms) == 0:
        return 0.

    x_u, p_u = zip(*sorted(zip(u_rv.atoms, u_rv.weights)))
    x_v, p_v = zip(*sorted(zip(v_rv.atoms, v_rv.weights)))
    l_u, l_v, diff_cdf, dist, pre = 0, 0, 0., 0., 0.
    while l_u < len(x_u) or l_v < len(x_v):
        if l_v == len(x_v) or (l_u < len(x_u) and x_v[l_v] > x_u[l_u]):
            dist += abs(diff_cdf)*(x_u[l_u]-pre)
            pre = x_u[l_u]
            diff_cdf += p_u[l_u]
            l_u += 1
        else:
            dist += abs(diff_cdf)*(x_v[l_v]-pre)
            pre = x_v[l_v]
            diff_cdf -= p_v[l_v]
            l_v += 1

    return dist





