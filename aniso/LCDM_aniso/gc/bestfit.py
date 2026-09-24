import sys
import numpy as np 
import pandas as pd

from scipy.integrate import quad 
import scipy.linalg as la 
from scipy.constants import c 
from scipy.optimize import minimize 
from iminuit import Minuit

from chi2_gc import chisq_gc

# ==================================================================================================

# Load Grid points

cut_name = sys.argv[1]
k_fixed = float(sys.argv[2])
#DIR = ""

print(f"Using k = {k_fixed} for cut {cut_name}", flush=True)

grid = pd.read_csv("./data/scan_grid_equatorial.csv") 
vx_g = grid["vx_gal"] 
vy_g = grid["vy_gal"] 
vz_g = grid["vz_gal"] 

v_grid = np.vstack([vx_g, vy_g, vz_g]).T 
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:,None] # shape (Npix, 3) 

print("No. of directions:", len(v_grid))
Npix = len(v_grid) 

print("Npix:", Npix)
 
# Load Galaxy Clusters information

cluster = np.loadtxt(f"./data/{cut_name}.txt", skiprows = 1) 
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

hem_plus_gc = [] 
hem_minus_gc = [] 

for i in range(len(v_grid)): 
    # main direction 
    d_gc = dot_grid_gc[i] 
    hem_plus_gc.append( np.where(d_gc > 0)[0] ) 
    hem_minus_gc.append( np.where(d_gc <= 0)[0] ) 

# ================================================================================================
    
def chi2_wrapper(H0, om, s, sig, idx_gc, z_gc, sig_L, f, logT_err, tau, k):
    params = [H0, om, s, sig]
    return chisq_gc(params, idx_gc, z_gc, sig_L, f, logT_err, tau, k)

def fit_params(idx_gc, z_gc, sig_L, f, logT_err, tau, k): 
    if len(idx_gc) == 0: 
        return (np.nan, np.nan, np.nan, np.nan)#, (np.nan, np.nan, np.nan, np.nan) 
    
    func2 = lambda H0, om, s, sig: chi2_wrapper(H0, om, s, sig, idx_gc, z_gc, sig_L, f, logT_err, tau, k)
       
    m = Minuit(func2, H0=70, om=0.3, s=2.102, sig=0.24)
    m.limits = [(0,150), (0,1), (1e-4, 5), (1e-4, 2)]
    m.errordef = 1

    m.migrad()
    #m.hesse()

    best = np.array([m.values["H0"], m.values["om"], m.values["s"], m.values["sig"]])
    #errs = np.array([m.errors["H0"], m.errors["om"], m.errors["s"], m.errors["sig"]])
    
    return best#, errs

# =================================================================================================

# Values

H0_p = np.zeros(Npix) # + hemisphere
H0_m = np.zeros(Npix) # - hemisphere

om_p = np.zeros(Npix) # + hemisphere
om_m = np.zeros(Npix) # - hemisphere

s_p = np.zeros(Npix) # + hemisphere
s_m = np.zeros(Npix) # - hemisphere

sig_p = np.zeros(Npix) # + hemisphere
sig_m = np.zeros(Npix) # - hemisphere

# Errors

H0_p_err = np.zeros(Npix) # + hemisphere
H0_m_err = np.zeros(Npix) # - hemisphere

om_p_err = np.zeros(Npix) # + hemisphere
om_m_err = np.zeros(Npix) # - hemisphere

s_p_err = np.zeros(Npix) # + hemisphere
s_m_err = np.zeros(Npix) # - hemisphere

sig_p_err = np.zeros(Npix) # + hemisphere
sig_m_err = np.zeros(Npix) # - hemisphere

if __name__=="__main__":

     for i in range(Npix): 
    # + hemisphere
         (best, err) = fit_params(hem_plus_gc[i], z_gc, sig_L, f, logT_err, tau, k_fixed)
         H0_p[i], om_p[i], s_p[i], sig_p[i] = best
         H0_p_err[i], om_p_err[i], s_p_err[i], sig_p_err[i] = err

    # – hemisphere
         (best, err) = fit_params(hem_minus_gc[i], z_gc, sig_L, f, logT_err, tau, k_fixed)
         H0_m[i], om_m[i], s_m[i], sig_m[i] = best
         H0_m_err[i], om_m_err[i], s_m_err[i], sig_m_err[i] = err

         print(f"Pixel {i+1}/{Npix} done.")

     df = pd.DataFrame({
             "H0_p": H0_p,
             "H0_m": H0_m,
             "H0_p_err": H0_p_err,
             "H0_m_err": H0_m_err,
             "om_p": om_p,
             "om_m": om_m,
             "om_p_err": om_p_err,
             "om_m_err": om_m_err,
             "s_p": s_p,
             "s_m": s_m,
             "s_p_err": s_p_err,
             "s_m_err": s_m_err,
             "sig_p": sig_p,
             "sig_m": sig_m,
             "sig_p_err": sig_p_err,
             "sig_m_err": sig_m_err,
        })

     df.to_csv(f"./results/H0_gc_{cut_name}.csv", index=False)

     print("Minimization done.")

         
         

