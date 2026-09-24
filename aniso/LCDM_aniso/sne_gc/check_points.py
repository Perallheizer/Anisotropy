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

grid = pd.read_csv("./data/scan_grid_equatorial.csv") 
vx_g = grid["vx_gal"] 
vy_g = grid["vy_gal"] 
vz_g = grid["vz_gal"] 

v_grid = np.vstack([vx_g, vy_g, vz_g]).T 
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:,None] 

print("No. of directions:", len(v_grid))
Npix = len(v_grid) 

print("Npix:", Npix)

# Load Supernovae information

pantheon = np.loadtxt(f"./data/pantheon_{cut_name}.txt", skiprows = 1) 
vx_sn = pantheon[:,0] 
vy_sn = pantheon[:,1]
vz_sn = pantheon[:,2] 
z_sn = pantheon[:,3] 
mb = pantheon[:,4] 
mb_err = pantheon[:,5] 
mu_ceph = pantheon[:,6] 
calibrator = pantheon[:,7] 

cov = np.loadtxt(f"./data/pantheon_cov_{cut_name}.txt")
N = len(z_sn)
cov = cov.reshape(N,N)
print("Length of subset:", N)
print("Shape of covariance matrix:", cov.shape)

# Ensure SN vectors are normalized (safety check) 
v_sn = np.vstack([vx_sn, vy_sn, vz_sn]).T 
v_sn = v_sn / np.linalg.norm(v_sn, axis=1)[:,None] 

dot_grid_sn = np.dot(v_grid, v_sn.T) 

cluster = np.loadtxt("./data/cluster_data.txt", skiprows = 1) 
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

sn_counts = [len(idx) for idx in hem_plus_sn] + [len(idx) for idx in hem_minus_sn]
gc_counts = [len(idx) for idx in hem_plus_gc] + [len(idx) for idx in hem_minus_gc]

print("--- SNe side ---")
print("Min:", min(sn_counts), "Median:", np.median(sn_counts))
print("Fraction <10:", np.mean([c < 10 for c in sn_counts]))

print("--- GC side ---")
print("Min:", min(gc_counts), "Median:", np.median(gc_counts))
print("Fraction <10:", np.mean([c < 10 for c in gc_counts]))  
    
    
    
    
    
     
