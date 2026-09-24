import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.coordinates import SkyCoord
import astropy.units as u

cut_name = sys.argv[1]

DIR = "./LCDM_aniso/gc_om_fix"
RESULTS = "./results"
DATA_DIR = "./data"
OUT_DIR = f"./summary/{DIR}"

# ---- Load real-data fit results ----

data = pd.read_csv(f"{DIR}/{RESULTS}/H0_gc_{cut_name}.csv")
H0_p = data["H0_p"].values
H0_m = data["H0_m"].values
H0_p_err = data["H0_p_err"].values
H0_m_err = data["H0_m_err"].values

# ---- Load grid ----

grid = pd.read_csv(f"{DIR}/{DATA_DIR}/scan_grid_equatorial.csv")
l = grid["l_g"].values
b = grid["b_g"].values

#print(np.min(l), np.max(l))
#print(np.min(b), np.max(b))

# =====================================================================
# 1. Delta H0 (raw difference between hemispheres)
# =====================================================================

Delta_H0 = H0_p - H0_m
Delta_H0_err = np.sqrt(H0_p_err**2 + H0_m_err**2)
sigma_Delta_H0 = Delta_H0 / Delta_H0_err

idx_max_dH0 = np.nanargmax(Delta_H0)
dH0_max = Delta_H0[idx_max_dH0]
dH0_max_err = Delta_H0_err[idx_max_dH0]
dH0_max_sigma = sigma_Delta_H0[idx_max_dH0]
l_max_dH0, b_max_dH0 = l[idx_max_dH0], b[idx_max_dH0]

# =====================================================================
# 2. Anisotropy Level (AL) — normalized difference
# =====================================================================

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
AL_err = (4 * np.sqrt((H0_p_err**2) * (H0_m**2) + (H0_m_err**2) * (H0_p**2))
          / (np.abs(H0_p) + np.abs(H0_m))**2)
sigma_AL = AL / AL_err

idx_max_AL = np.nanargmax(AL)
AL_max = AL[idx_max_AL]
AL_max_err = AL_err[idx_max_AL]
AL_max_sigma = sigma_AL[idx_max_AL]
l_max_AL, b_max_AL = l[idx_max_AL], b[idx_max_AL]

# =====================================================================
# 4. CMB direction, for angular separation reference
# =====================================================================

cmb_ra, cmb_dec = 168, -7
coords_gal = SkyCoord(ra=cmb_ra*u.degree, dec=cmb_dec*u.degree, frame='icrs').galactic
l_cmb = np.radians(coords_gal.l.value)
b_cmb = np.radians(coords_gal.b.value)
print("CMB galactic:", l_cmb, b_cmb)

sep_dH0 = SkyCoord(l=np.degrees(l_max_dH0)*u.degree, b=np.degrees(b_max_dH0)*u.degree, frame='galactic').separation(coords_gal).degree
sep_AL = SkyCoord(l=np.degrees(l_max_AL)*u.degree, b=np.degrees(b_max_AL)*u.degree, frame='galactic').separation(coords_gal).degree
         
# =====================================================================
# 5. Write everything to one systematic summary file
# =====================================================================

out_txt = f"{OUT_DIR}/full_summary_gc_{cut_name}.txt"
with open(out_txt, "w") as f:
    f.write(f"===== Anisotropy summary, cut {cut_name} =====\n\n")
    f.write("--- Anisotropy Level (AL) ---\n")
    f.write(f"Max AL               = {AL_max:.4f}\n")
    f.write(f"Uncertainty          = {AL_max_err:.4f}\n")
    f.write(f"Local significance   = {AL_max_sigma:.2f} sigma\n")
    f.write(f"Direction: l = {np.degrees(l_max_AL):.2f} deg, b = {np.degrees(b_max_AL):.2f} deg\n")
    f.write(f"Separation from CMB dipole = {sep_AL:.2f} deg\n\n")

print(f"Saved full summary to {out_txt}")
print(open(out_txt).read())

# =====================================================================
# 6. Plotting helper — reusable for both Delta H0 and AL maps
# =====================================================================

def wrap_lon(lon):
    return np.where(lon > np.pi, lon - 2*np.pi, lon)

def make_map(values, l_max, b_max, sep_val, fname, cmap='jet'):
    N = 300
    l_dense = np.linspace(0, 2*np.pi, N)
    b_dense = np.linspace(-np.pi/2, np.pi/2, N)
    l_grid, b_grid = np.meshgrid(l_dense, b_dense)

    pad_width = np.radians(30)
    mask_low = l < pad_width
    mask_high = l > (2*np.pi - pad_width)
    l_padded = np.concatenate([l, l[mask_low] + 2*np.pi, l[mask_high] - 2*np.pi])
    b_padded = np.concatenate([b, b[mask_low], b[mask_high]])
    v_padded = np.concatenate([values, values[mask_low], values[mask_high]])

    points_2d = np.vstack([l_padded, b_padded]).T
    v_dense = griddata(points_2d, v_padded, (l_grid, b_grid), method='cubic')

    lon_grid_wrapped = wrap_lon(l_grid)
    sort_idx = np.argsort(lon_grid_wrapped[0])
    lon_sorted = lon_grid_wrapped[:, sort_idx]
    b_sorted = b_grid[:, sort_idx]
    v_sorted = v_dense[:, sort_idx]

    lon_max_wrapped = wrap_lon(np.array([l_max]))[0]
    lon_cmb_wrapped = wrap_lon(np.array([l_cmb]))[0]
    
    fig = plt.figure(figsize=(8, 5))
    ax = plt.subplot(111, projection='aitoff')
    im = ax.pcolormesh(lon_sorted, b_sorted, v_sorted, shading='auto', cmap=cmap)
    plt.colorbar(im, orientation='horizontal', pad=0.05)

    ax.scatter(lon_cmb_wrapped, b_cmb, s=150, color='black', marker='X', zorder=5)
    ax.scatter(lon_max_wrapped, b_max, s=150, color='white', edgecolor='black', marker='o', zorder=5)

    ax.grid(True)
    #ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1))
    #plt.title(f"{label}, cut {cut_name}")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved map to {fname}")


# ---- Delta H0 map ----
#make_map(Delta_H0, l_max_dH0, b_max_dH0, sep_dH0, f"{DIR}/{RESULTS_DIR}/dH0_map_{cut_name}.pdf")

# ---- AL map ----
make_map(AL, l_max_AL, b_max_AL, sep_AL, f"{OUT_DIR}/AL_map_gc_{cut_name}.pdf")












