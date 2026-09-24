import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd

cmb_r = 168
cmb_d = -7

cmb_r   = np.radians(cmb_r)
cmb_d   = np.radians(cmb_d)

cmb_lon = cmb_r - np.pi
cmb_lat = cmb_d

data = pd.read_csv("./results/H0_anisotropy_14.csv")

delta_h0_v = data["Delta_H0_v"]

grid = pd.read_csv("./data/scan_grid_equatorial.csv")
RA = grid["RA_eq"]
DEC = grid["DEC_eq"]

x = np.cos(DEC) * np.cos(RA)
y = np.cos(DEC) * np.sin(RA)
z = np.sin(DEC)

points = np.vstack([x,y,z]).T
values = delta_h0_v

N = 300

R_dense = np.linspace(0, 2*np.pi, N)
D_dense = np.linspace(-np.pi/2, np.pi/2, N)

R_grid, D_grid = np.meshgrid(R_dense, D_dense)

points_2d = np.vstack([RA, DEC]).T
                          
Delta_dense = griddata(points_2d, values, (R_grid, D_grid), method='cubic')
Delta_dense = Delta_dense.reshape(N, N)

lon = R_grid - np.pi
lat = D_grid

idx_max = np.argmax(delta_h0_v)
max_val = delta_h0_v[idx_max]

RA_max = RA[idx_max]
DEC_max = DEC[idx_max]


print("Max ΔH0 =", max_val)
print("RA (rad) =", RA_max, "  DEC (rad) =", DEC_max)
print("RA (deg) =", np.degrees(RA_max), "  DEC (deg) =", np.degrees(DEC_max))

lon_max = RA_max - np.pi
lat_max = DEC_max

# Max on interpolated grid
flat_index = np.nanargmax(Delta_dense)
i, j = np.unravel_index(flat_index, Delta_dense.shape)

RA_interp_max = R_dense[j]
DEC_interp_max = D_dense[i]

print("Interpolated grid max ΔH0 =", Delta_dense[i, j])
print("RA_interp (deg) =", np.degrees(RA_interp_max))
print("DEC_interp (deg) =", np.degrees(DEC_interp_max))

    
fig = plt.figure(figsize=(10,5))
ax = plt.subplot(111, projection='aitoff')

im = ax.pcolormesh(lon, lat, Delta_dense, shading='auto', cmap='jet')
plt.colorbar(im, orientation='horizontal', pad=0.05)
ax.scatter(cmb_lon, cmb_lat, s = 150, color = 'black', marker = '*')
ax.scatter(lon_max, lat_max, s=100, color='white', edgecolor='black', marker='o')

ax.grid(True)
plt.title("Interpolated Map  ")

plt.savefig("./results/Anisotropy_map_14.pdf")

plt.show()

#==============================xxxxxx==================================








