import sys
import numpy as np
import pandas as pd
from iminuit import Minuit

cut_name = sys.argv[1]   # change as needed

DIR = "./Pade_aniso/sne_rev"

# ---- Load real data results, find the known max-AL pixel index ----
data = pd.read_csv(f"{DIR}/results/H0_anisotropy_{cut_name}.csv")
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values
AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
idx_max = np.nanargmax(AL)
print(f"Using max-AL pixel index: {idx_max}")

# ---- Load grid, load SNe, rebuild hemisphere split JUST for this one pixel ----
grid = pd.read_csv(f"{DIR}/data/scan_grid_equatorial.csv")
v_grid = np.vstack([grid["vx_gal"], grid["vy_gal"], grid["vz_gal"]]).T
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:, None]

pantheon = np.loadtxt(f"{DIR}/data/pantheon_{cut_name}.txt", skiprows=1)
vx_sn, vy_sn, vz_sn = pantheon[:,0], pantheon[:,1], pantheon[:,2]
mb = pantheon[:,4]
mu_ceph = pantheon[:,6]
calibrator = pantheon[:,7]

cov = np.loadtxt(f"{DIR}/data/pantheon_cov_{cut_name}.txt")
N = len(mb)
cov = cov.reshape(N, N)

v_sn = np.vstack([vx_sn, vy_sn, vz_sn]).T
v_sn = v_sn / np.linalg.norm(v_sn, axis=1)[:, None]

d = np.dot(v_grid[idx_max], v_sn.T)   # ONLY this one direction, not all 768
hem_plus = np.where(d > 0)[0]
hem_minus = np.where(d <= 0)[0]

# ---- Restrict to calibrators only, within each hemisphere ----
calib_plus = hem_plus[calibrator[hem_plus] == 1]
calib_minus = hem_minus[calibrator[hem_minus] == 1]
print(f"Calibrators: plus={len(calib_plus)}, minus={len(calib_minus)}")

# ---- Fit M only, using calibrators, no cosmology needed ----
A_plus = np.zeros(N)
A_minus = np.zeros(N)

A_plus[calib_plus] = 1.0
A_minus[calib_minus] = 1.0

use = (A_plus + A_minus) > 0

A = np.column_stack([A_plus[use], A_minus[use]])

y = mb[use] - mu_ceph[use]

C = cov[np.ix_(use, use)]
Cinv = np.linalg.inv(C)

F = A.T @ Cinv @ A

# Parameter covariance matrix
Cov_par = np.linalg.inv(F)

# Best-fit parameters
theta = Cov_par @ (A.T @ Cinv @ y)

M_plus, M_minus = theta

# Marginalized uncertainties
M_plus_err = np.sqrt(Cov_par[0, 0])
M_minus_err = np.sqrt(Cov_par[1, 1])

# Cross-covariance between hemispheres
Cov_pm = Cov_par[0, 1]

# Difference and significance
dM = M_plus - M_minus

sigma_dM = dM / np.sqrt(Cov_par[0, 0] + Cov_par[1, 1] - 2 * Cov_pm)

print(f"M_plus  = {M_plus:.4f} +/- {M_plus_err:.4f}  (n={len(calib_plus)})")
print(f"M_minus = {M_minus:.4f} +/- {M_minus_err:.4f}  (n={len(calib_minus)})")
print(f"Delta M = {M_plus - M_minus:.4f}")
print(f"Significance sigma = {sigma_dM:.2f}")

with open(f"./summary/{DIR}/ceph_M_{cut_name}.txt", "w") as f:
    f.write(f"Cut: {cut_name}\n")
    f.write(f"Max-AL pixel index: {idx_max}\n")
    f.write(f"M_plus  = {M_plus:.4f} +/- {M_plus_err:.4f} (n={len(calib_plus)})\n")
    f.write(f"M_minus = {M_minus:.4f} +/- {M_minus_err:.4f} (n={len(calib_minus)})\n")
    f.write(f"Delta M = {M_plus - M_minus:.4f}\n")
    f.write(f"Significance (sigma) = {sigma_dM:.2f}\n")
