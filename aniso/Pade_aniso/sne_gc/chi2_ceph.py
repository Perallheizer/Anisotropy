import numpy as np
from cosmo_model import Model
from scipy.linalg import cho_solve

def chisq_sne(params, idx_plus, idx_minus, z, mb, L, mu_ceph, calibrator):
    H0_p, q0_p, M_p, H0_m, q0_m, M_m = params

    if len(idx_plus) == 0 or len(idx_minus) == 0:
        return np.inf

    idx_all = np.concatenate([idx_plus, idx_minus])

    # --- theory for plus hemisphere ---
    z_p = z[idx_plus]
    mu_p = mb[idx_plus] - M_p
    mu_ceph_p = mu_ceph[idx_plus]
    calib_p = calibrator[idx_plus]
    cosmo_p = Model(H0_p, q0_p)
    dL_p = np.vectorize(cosmo_p.dLz)(z_p)
    mu_th_p = 5 * np.log10(dL_p) + 25
    Q_p = np.where(calib_p == 1, mu_p - mu_ceph_p, mu_p - mu_th_p)

    # --- theory for minus hemisphere ---
    z_m = z[idx_minus]
    mu_m = mb[idx_minus] - M_m
    mu_ceph_m = mu_ceph[idx_minus]
    calib_m = calibrator[idx_minus]
    cosmo_m = Model(H0_m, q0_m)
    dL_m = np.vectorize(cosmo_m.dLz)(z_m)
    mu_th_m = 5 * np.log10(dL_m) + 25
    Q_m = np.where(calib_m == 1, mu_m - mu_ceph_m, mu_m - mu_th_m)

    # --- combine, using the FULL covariance sub-block (includes cross terms) ---
    Q_all = np.concatenate([Q_p, Q_m])
    #cov_sub = cov[np.ix_(idx_all, idx_all)] 

    y= cho_solve((L, True), Q_all)
    chi2 = Q_all @ y

    return chi2

