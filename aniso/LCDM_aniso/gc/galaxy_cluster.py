import numpy as np
import pandas as pd

data = pd.read_csv('cluster_sample.csv', delimiter = ",")

z = data["z"]

l = data["l"]
b = data["b"]

T = data["T"]
T_plus = data["T_plus"]
T_minus = data["T_minus"]

# Lx = data["Lx"]
sig_L = data["sigma_L"]

f = data["f"]
instr = data["Instrument"]

# ===========================================================================

cond = (instr == "Chandra")

z = z[cond]

l = l[cond]
b = b[cond]

T = T[cond]
T_plus = T_plus[cond]
T_minus = T_minus[cond]
sig_L = sig_L[cond]
f = f[cond]

# ===========================================================================

#T_err = (2*T_plus*T_minus/(T_plus+T_minus) + np.sqrt(T_plus*T_minus))*0.5
logT_err = np.log10(np.e)*((T_plus - T_minus)/(2*T))

tau = T/4

l_rad = np.deg2rad(l)
b_rad = np.deg2rad(b)

vx = np.cos(b_rad)*np.cos(l_rad)
vy = np.cos(b_rad)*np.sin(l_rad)
vz = np.sin(b_rad)

data = np.column_stack([vx, vy, vz, z, sig_L, f, logT_err, tau])

header = "vx vy vz z sig_L f logT_err tau"
print(np.min(z), np.max(z), len(z))

np.savetxt("./data/chandra.txt", data, header=header, comments='')

print("Galaxy Cluster data is saved !")
