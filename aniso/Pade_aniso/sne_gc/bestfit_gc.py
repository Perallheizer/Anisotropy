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

from chi2_sne import chisq_sne
from chi2_gc import chisq_gc

# ==================================================================================================

# Load Grid points

cut_name = sys.argv[1]

DIR = "./aniso/Pade_aniso/sne_gc"

grid = pd.read_csv(f"{DIR}/data/scan_grid_equatorial.csv") 
vx_g = grid["vx_gal"] # grid (Npix,) 
vy_g = grid["vy_gal"] 
vz_g = grid["vz_gal"] 

v_grid = np.vstack([vx_g, vy_g, vz_g]).T 
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:,None] # shape (Npix, 3) 

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
 
# Load Galaxy Clusters information

cluster = np.loadtxt(f"{DIR}/data/cluster_data.txt", skiprows = 1) 
vx_gc = cluster[:,0] 
vy_gc = cluster[:,1] 
vz_gc = cluster[:,2] 
z_gc = cluster[:,3] 
sig_L = cluster[:,4] 
f = cluster[:,5] 
logT_err = cluster[:,6] 
tau = cluster[:,7]

# Ensure cluster vectors are normalized (safety check) 
v_gc = np.vstack([vx_gc, vy_gc, vz_gc]).T 
v_gc = v_gc / np.linalg.norm(v_gc, axis=1)[:,None] 

dot_grid_gc = np.dot(v_grid, v_gc.T) 

# =================================================================================================
 
hem_plus_sn = [] 
hem_minus_sn = []
hem_plus_gc = [] 
hem_minus_gc = [] 

for i in range(len(v_grid)): 
    # main direction 
    d_sn = dot_grid_sn[i] 
    hem_plus_sn.append( np.where(d_sn > 0)[0] ) 
    hem_minus_sn.append( np.where(d_sn <= 0)[0] ) 
    d_gc = dot_grid_gc[i] 
    hem_plus_gc.append( np.where(d_gc > 0)[0] ) 
    hem_minus_gc.append( np.where(d_gc <= 0)[0] ) 

# ================================================================================================
    
def chi2_joint(H0_p, q0_p, M_p, s_p, sig_p,
    H0_m, q0_m, M_m, s_m, sig_m,

    idx_sn_p, idx_sn_m,
    idx_gc_p, idx_gc_m,

    z_sn, mb, L, mu_ceph, calibrator,
    z_gc, sig_L, f, logT_err, tau):
    
    params1 = [H0_p, q0_p, M_p, H0_m, q0_m, M_m]
    
    chi_sn = chisq_sne(params1, idx_sn_p, idx_sn_m, z_sn, mb, L, mu_ceph, calibrator)
    chi_gc_p = chisq_gc([H0_p, q0_p, s_p, sig_p], idx_gc_p, z_gc, sig_L, f, logT_err, tau)
    chi_gc_m = chisq_gc([H0_m, q0_m, s_m, sig_m], idx_gc_m, z_gc, sig_L, f, logT_err, tau)
    
    return (chi_sn + chi_gc_p + chi_gc_m)

def fit_joint(idx_sn_p, idx_sn_m, idx_gc_p, idx_gc_m, z_sn, mb, cov, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau): 
    
    idx_all_sn = np.concatenate([idx_sn_p, idx_sn_m])
    cov_sub = cov = cov[np.ix_(idx_all_sn, idx_all_sn)]
    L = np.linalg.cholesky(cov_sub)
    
    func = lambda H0_p, q0_p, M_p, s_p, sig_p, H0_m, q0_m, M_m, s_m, sig_m: chi2_joint(H0_p, q0_p, M_p, s_p, sig_p, H0_m, q0_m, M_m, s_m, sig_m, idx_sn_p, idx_sn_m, idx_gc_p, idx_gc_m, z_sn, mb, L, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau)
       
    m = Minuit(func, H0_p=70, q0_p=-0.55, M_p=-19.3, s_p=2.102, sig_p=0.24, H0_m=70, q0_m=-0.55, M_m=-19.3, s_m=2.102, sig_m=0.24)
    m.limits = [(0,150), (-2, 2), (-20,-18), (1e-4, 5), (1e-4, 2), (0,150), (-2, 2), (-20,-18), (1e-4, 5), (1e-4, 2)]
    m.errordef = 1

    m.migrad()
    #m.hesse()

    best = np.array([m.values["H0_p"], m.values["q0_p"], m.values["M_p"], m.values["s_p"], m.values["sig_p"], m.values["H0_m"], m.values["q0_m"], m.values["M_m"], m.values["s_m"], m.values["sig_m"]])
    #errs = np.array([m.errors["H0_p"], m.errors["q0_p"], m.errors["M_p"], m.errors["s_p"], m.errors["sig_p"], m.errors["H0_m"], m.errors["q0_m"], m.errors["M_m"], m.errors["s_m"], m.errors["sig_m"]])
    
    return best#, errs

# =================================================================================================

# Values

H0_p = np.zeros(Npix) # + hemisphere
H0_m = np.zeros(Npix) # - hemisphere

q0_p = np.zeros(Npix) # + hemisphere
q0_m = np.zeros(Npix) # - hemisphere

M_p = np.zeros(Npix) # + hemisphere
M_m = np.zeros(Npix) # - hemisphere

s_p = np.zeros(Npix) # + hemisphere
s_m = np.zeros(Npix) # - hemisphere

sig_p = np.zeros(Npix) # + hemisphere
sig_m = np.zeros(Npix) # - hemisphere

# Errors

H0_p_err = np.zeros(Npix) # + hemisphere
H0_m_err = np.zeros(Npix) # - hemisphere

q0_p_err = np.zeros(Npix) # + hemisphere
q0_m_err = np.zeros(Npix) # - hemisphere

M_p_err = np.zeros(Npix) # + hemisphere
M_m_err = np.zeros(Npix) # - hemisphere

s_p_err = np.zeros(Npix) # + hemisphere
s_m_err = np.zeros(Npix) # - hemisphere

sig_p_err = np.zeros(Npix) # + hemisphere
sig_m_err = np.zeros(Npix) # - hemisphere

if __name__=="__main__":

     for i in range(Npix): 
    # + hemisphere
         (best, err) = fit_joint(hem_plus_sn[i], hem_minus_sn[i], hem_plus_gc[i], hem_minus_gc[i], z_sn, mb, cov, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau)
         H0_p[i], q0_p[i], M_p[i], s_p[i], sig_p[i], H0_m[i], q0_m[i], M_m[i], s_m[i], sig_m[i] = best
         H0_p_err[i], q0_p_err[i], M_p_err[i], s_p_err[i], sig_p_err[i], H0_m_err[i], q0_m_err[i], M_m_err[i], s_m_err[i], sig_m_err[i] = err

         print(f"Pixel {i+1}/{Npix} done.")

     df = pd.DataFrame({
          "H0_p": H0_p,
          "H0_m": H0_m,
          "H0_p_err": H0_p_err,
          "H0_m_err": H0_m_err,
          "q0_p": q0_p,
          "q0_m": q0_m,
          "q0_p_err": q0_p_err,
          "q0_m_err": q0_m_err,
          "M_p": M_p,
          "M_m": M_m,
          "M_p_err": M_p_err,
          "M_m_err": M_m_err,
          "s_p": s_p,
          "s_m": s_m,
          "s_p_err": s_p_err,
          "s_m_err": s_m_err,
          "sig_p": sig_p,
          "sig_m": sig_m,
          "sig_p_err": sig_p_err,
          "sig_m_err": sig_m_err,
     })

     df.to_csv(f"{DIR}/results/H0_gc_{cut_name}.csv", index=False)

     print("Minimization done.")


         
         
