import numpy as np
from cosmo_model import Model
from scipy.linalg import cho_solve

def chisq_sne(params, idx_plus, idx_minus, z, mb, L, mu_ceph, calibrator): 

    H0_p, om_p, M_p, H0_m, om_m, M_m = params
    """Compute χ² for a subset of SNe given their indices""" 
    if len(idx_plus) == 0 or len(idx_minus) == 0:
        return np.inf
    
    mu_p = mb[idx_plus] - M_p
    z_p = z[idx_plus]

    cosmo_p = Model(H0_p, om_p)
    dL_p = np.vectorize(cosmo_p.dLz)(z_p)

    mu_th_p = 5*np.log10(dL_p) + 25

    Q_p = mu_p - mu_th_p

    # ---------- MINUS hemisphere ----------

    mu_m = mb[idx_minus] - M_m
    z_m = z[idx_minus]

    cosmo_m = Model(H0_m, om_m)
    dL_m = np.vectorize(cosmo_m.dLz)(z_m)

    mu_th_m = 5*np.log10(dL_m) + 25

    Q_m = mu_m - mu_th_m

    # ---------- Joint residual vector ----------

    idx_all = np.concatenate([idx_plus, idx_minus])

    Q_all = np.concatenate([Q_p, Q_m])

    # ---------- Full covariance block ----------

    #cov_sub = cov[np.ix_(idx_all, idx_all)]
    y = cho_solve((L, True), Q_all)
    chi2 = Q_all @ y #np.linalg.solve(cov_sub, Q_all)

    return chi2

