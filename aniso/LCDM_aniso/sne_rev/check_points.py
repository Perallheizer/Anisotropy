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
 
# =================================================================================================
 
hem_plus_sn = [] 
hem_minus_sn = []

for i in range(len(v_grid)): 
    # main direction 
    d_sn = dot_grid_sn[i] 
    hem_plus_sn.append( np.where(d_sn > 0)[0] ) 
    hem_minus_sn.append( np.where(d_sn <= 0)[0] )
    
counts_plus = [len(idx) for idx in hem_plus_sn]
counts_minus = [len(idx) for idx in hem_minus_sn]

print("Min points per hemisphere:", min(counts_plus + counts_minus))
print("Median points per hemisphere:", np.median(counts_plus + counts_minus))
print("Fraction of pixels with <10 points on either side:",
      np.mean([(p < 10 or m < 10) for p, m in zip(counts_plus, counts_minus)]))    
    
    
    
    
    
     
