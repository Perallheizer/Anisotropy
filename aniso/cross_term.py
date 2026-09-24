import sys
import numpy as np 
import pandas as pd

# ==================================================================================================

# Load Grid points

cut_name = sys.argv[1]

grid = pd.read_csv("./LCDM_aniso/sne/data/scan_grid_equatorial.csv") 
vx_g = grid["vx_gal"] 
vy_g = grid["vy_gal"] 
vz_g = grid["vz_gal"] 

v_grid = np.vstack([vx_g, vy_g, vz_g]).T 
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:,None] 

print("No. of directions:", len(v_grid))
Npix = len(v_grid) 

print("Npix:", Npix)

# Load Supernovae information

pantheon = np.loadtxt(f"./LCDM_aniso/sne/data/pantheon_{cut_name}.txt", skiprows = 1) 
vx_sn = pantheon[:,0] 
vy_sn = pantheon[:,1]
vz_sn = pantheon[:,2] 
z_sn = pantheon[:,3] 
mb = pantheon[:,4] 
mb_err = pantheon[:,5] 
mu_ceph = pantheon[:,6] 
calibrator = pantheon[:,7] 

cov = np.loadtxt(f"./LCDM_aniso/sne/data/pantheon_cov_{cut_name}.txt")
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
    
data = pd.read_csv(f"./LCDM_aniso/sne/results/H0_anisotropy_{cut_name}.csv")
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))

idx_max = np.nanargmax(AL)
    
plus = hem_plus_sn[idx_max]
minus = hem_minus_sn[idx_max]

Cpp = cov[np.ix_(plus, plus)]
Cmm = cov[np.ix_(minus, minus)]
Cpm = cov[np.ix_(plus, minus)]

def offdiag_mean_abs(M, is_square):
    if is_square:
        mask = ~np.eye(M.shape[0], dtype=bool)
        return np.mean(np.abs(M[mask]))
    else:
        return np.mean(np.abs(M))

print("Cpp off-diagonal mean:", offdiag_mean_abs(Cpp, is_square=True))
print("Cmm off-diagonal mean:", offdiag_mean_abs(Cmm, is_square=True))
print("Cpm mean:", offdiag_mean_abs(Cpm, is_square=False))
    
corr = cov / np.outer(np.sqrt(np.diag(cov)),
                      np.sqrt(np.diag(cov)))

Cpp_corr = corr[np.ix_(plus, plus)]
Cmm_corr = corr[np.ix_(minus, minus)]
Cpm_corr = corr[np.ix_(plus, minus)]

print("Cpp corr off-diagonal:",
      offdiag_mean_abs(Cpp_corr, True))

print("Cmm corr off-diagonal:",
      offdiag_mean_abs(Cmm_corr, True))

print("Cpm corr mean:",
      offdiag_mean_abs(Cpm_corr, False))   
    
    
    
    
    
    
    
    
