import sys
import numpy as np
import pandas as pd

cut_name = sys.argv[1]

DIR = "./LCDM_aniso/sne_gc"
RESULTS_DIR = "./results"

# ---- Load real-data fit results, find max AL pixel ----
data = pd.read_csv(f"{DIR}/{RESULTS_DIR}/H0_ceph_{cut_name}.csv")
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values
M_p, M_m = data["M_p"].values, data["M_m"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
idx_max = np.nanargmax(AL)

print(f"Max AL pixel index: {idx_max}, AL = {AL[idx_max]:.4f}")
print(f"M_p (plus hemisphere)  = {M_p[idx_max]:.4f}")
print(f"M_m (minus hemisphere) = {M_m[idx_max]:.4f}")
print(f"Delta M = M_p - M_m = {M_p[idx_max] - M_m[idx_max]:.4f}")

# ---- Reload SNe data and grid, to get the hemisphere split at this specific pixel ----
grid = pd.read_csv(f"{DIR}/data/scan_grid_equatorial.csv")
vx_g, vy_g, vz_g = grid["vx_gal"].values, grid["vy_gal"].values, grid["vz_gal"].values
v_grid = np.vstack([vx_g, vy_g, vz_g]).T
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:, None]

pantheon = np.loadtxt(f"{DIR}/data/pantheon_{cut_name}.txt", skiprows=1)
vx_sn, vy_sn, vz_sn = pantheon[:,0], pantheon[:,1], pantheon[:,2]
calibrator = pantheon[:,7]

v_sn = np.vstack([vx_sn, vy_sn, vz_sn]).T
v_sn = v_sn / np.linalg.norm(v_sn, axis=1)[:, None]

# dot product for JUST this one pixel's direction
d = np.dot(v_grid[idx_max], v_sn.T)
hem_plus = np.where(d > 0)[0]
hem_minus = np.where(d <= 0)[0]

n_calib_plus = np.sum(calibrator[hem_plus] == 1)
n_calib_minus = np.sum(calibrator[hem_minus] == 1)
n_total_plus = len(hem_plus)
n_total_minus = len(hem_minus)

print()
print(f"--- Cepheid calibrator distribution at max-AL direction ---")
print(f"Plus hemisphere:  {n_total_plus} total SNe, {n_calib_plus} Cepheid calibrators")
print(f"Minus hemisphere: {n_total_minus} total SNe, {n_calib_minus} Cepheid calibrators")

# ---- Save to a text summary ----
with open(f"{DIR}/{RESULTS_DIR}/ceph_check_{cut_name}.txt", "w") as f_out:
    f_out.write(f"Cut: {cut_name}\n")
    f_out.write(f"Max AL pixel index: {idx_max}, AL = {AL[idx_max]:.4f}\n\n")
    f_out.write(f"M_p = {M_p[idx_max]:.4f}\n")
    f_out.write(f"M_m = {M_m[idx_max]:.4f}\n")
    f_out.write(f"Delta M = {M_p[idx_max] - M_m[idx_max]:.4f}\n\n")
    f_out.write(f"Plus hemisphere:  {n_total_plus} total SNe, {n_calib_plus} Cepheid calibrators\n")
    f_out.write(f"Minus hemisphere: {n_total_minus} total SNe, {n_calib_minus} Cepheid calibrators\n")

print(f"\nSaved to {DIR}/{RESULTS_DIR}/ceph_check_{cut_name}.txt")
