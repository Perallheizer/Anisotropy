import numpy as np
import pandas as pd
import healpy as hp
import sys

nside = 8
param = "omega_m"
cut_name = sys.argv[1]

data = pd.read_csv(f"./LCDM_aniso/sne_gc/results/H0_gc_{cut_name}.csv")   # Pade version
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values
H0_p_err, H0_m_err = data["H0_p_err"].values, data["H0_m_err"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))

idx_max = np.nanargmax(AL)

theta_p = data[f"{param}_p"].values
theta_m = data[f"{param}_m"].values
theta_p_err = data[f"{param}_p_err"].values
theta_m_err = data[f"{param}_m_err"].values

print(f"theta_p = {theta_p[idx_max]:.3f} +/- {theta_p_err[idx_max]:.3f}")
print(f"theta_m = {theta_m[idx_max]:.3f} +/- {theta_m_err[idx_max]:.3f}")


