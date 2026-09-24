import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np 
import pandas as pd

from scipy.integrate import quad 
import scipy.linalg as la 
from scipy.constants import c 
from scipy.optimize import minimize 
from iminuit import Minuit

from chi2_ceph import chisq_sne
# ==================================================================================================

# Load Grid points

cut_name = sys.argv[1]

DIR = "./aniso/Pade_aniso/sne_rev"

grid = pd.read_csv(f"{DIR}/data/scan_grid_equatorial.csv") 
vx_g = grid["vx_gal"] 
vy_g = grid["vy_gal"] 
vz_g = grid["vz_gal"] 

v_grid = np.vstack([vx_g, vy_g, vz_g]).T 
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:,None] 

print("No. of directions:", len(v_grid))
Npix = len(v_grid) 

print("Npix:", Npix)

# Load Supernovae information

pantheon = np.loadtxt(f"{DIR}/data/pantheon_{cut_name}.txt", skiprows = 1) 
vx_sn = pantheon[:,0] 
vy_sn = pantheon[:,1]
vz_sn = pantheon[:,2] 
z_sn = pantheon[:,3] 
mb = pantheon[:,4] 
mb_err = pantheon[:,5] 
mu_ceph = pantheon[:,6] 
calibrator = pantheon[:,7] 

cov = np.loadtxt(f"{DIR}/data/pantheon_cov_{cut_name}.txt")
N = len(z_sn)
cov = cov.reshape(N,N)
print("Length of subset:", N)
print("Shape of covariance matrix:", cov.shape)

# Ensure SN vectors are normalized (safety check) 
v_sn = np.vstack([vx_sn, vy_sn, vz_sn]).T 
v_sn = v_sn / np.linalg.norm(v_sn, axis=1)[:,None] 

dot_grid_sn = np.dot(v_grid, v_sn.T)  
 
# =================================================================================================
 
hem_plus_sn = [] 
hem_minus_sn = []

for i in range(len(v_grid)): 
    # main direction 
    d_sn = dot_grid_sn[i] 
    hem_plus_sn.append( np.where(d_sn > 0)[0] ) 
    hem_minus_sn.append( np.where(d_sn <= 0)[0] ) 
# ================================================================================================
    
def fit_joint(idx_plus, idx_minus, z, mb, cov, mu_ceph, calibrator):
    if len(idx_plus) == 0 or len(idx_minus) == 0:
        return (np.nan,)*6#, (np.nan,)*6
        
    idx_all_sn = np.concatenate([idx_plus, idx_minus])
    cov_sub = cov[np.ix_(idx_all_sn, idx_all_sn)]
    L = np.linalg.cholesky(cov_sub)

    func = lambda H0_p, q0_p, M_p, H0_m, q0_m, M_m: chisq_sne(
        [H0_p, q0_p, M_p, H0_m, q0_m, M_m],
        idx_plus, idx_minus, z, mb, L, mu_ceph, calibrator)

    m = Minuit(func, H0_p=70, q0_p=-0.55, M_p=-19.3,
                     H0_m=70, q0_m=-0.55, M_m=-19.3)
    m.limits = [(0,150), (-2, 2), (-20,-18), (0,150), (-2, 2), (-20,-18)]
    m.errordef = 1
    m.migrad()
    m.hesse()

    best = np.array([m.values["H0_p"], m.values["q0_p"], m.values["M_p"],
                      m.values["H0_m"], m.values["q0_m"], m.values["M_m"]])
    errs = np.array([m.errors["H0_p"], m.errors["q0_p"], m.errors["M_p"],
                     m.errors["H0_m"], m.errors["q0_m"], m.errors["M_m"]])
    return best, errs

#====================================================================================

H0_p = np.zeros(Npix) # + hemisphere
H0_m = np.zeros(Npix) # - hemisphere

q0_p = np.zeros(Npix) # + hemisphere
q0_m = np.zeros(Npix) # - hemisphere

M_p = np.zeros(Npix) # + hemisphere
M_m = np.zeros(Npix) # - hemisphere

# Errors

H0_p_err = np.zeros(Npix) # + hemisphere
H0_m_err = np.zeros(Npix) # - hemisphere

q0_p_err = np.zeros(Npix) # + hemisphere
q0_m_err = np.zeros(Npix) # - hemisphere

M_p_err = np.zeros(Npix) # + hemisphere
M_m_err = np.zeros(Npix) # - hemisphere


if __name__ == "__main__":

    for i in range(Npix):
        (best, err) = fit_joint(hem_plus_sn[i], hem_minus_sn[i], z_sn, mb, cov, mu_ceph, calibrator)
        H0_p[i], q0_p[i], M_p[i], H0_m[i], q0_m[i], M_m[i] = best
        H0_p_err[i], q0_p_err[i], M_p_err[i], H0_m_err[i], q0_m_err[i], M_m_err[i] = err
        print(f"Pixel {i+1}/{Npix} done.", flush=True)

    df = pd.DataFrame({
    "H0_p": H0_p,
    "H0_m": H0_m,
    "q0_p": q0_p,
    "q0_m": q0_m,
    "M_p": M_p,
    "M_m": M_m,
    "H0_p_err": H0_p_err,
    "H0_m_err": H0_m_err,
    "q0_p_err": q0_p_err,
    "q0_m_err": q0_m_err,
    "M_p_err": M_p_err,
    "M_m_err": M_m_err
})
    df.to_csv(f"{DIR}/results/H0_anisotropy_{cut_name}.csv", index=False)
    print("Joint fit (with cross-hemisphere terms) done.")

         
         
