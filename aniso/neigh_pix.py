import numpy as np
import pandas as pd
import healpy as hp

nside = 8
cut_name = "A"
DIR = "./Pade_aniso/sne/results"

data = pd.read_csv(f"{DIR}/H0_anisotropy_{cut_name}.csv")
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values
H0_p_err, H0_m_err = data["H0_p_err"].values, data["H0_m_err"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
AL_err = (4 * np.sqrt((H0_p_err**2)*(H0_m**2) + (H0_m_err**2)*(H0_p**2))
          / (np.abs(H0_p) + np.abs(H0_m))**2)

Npix = len(AL)
min_sigma_vs_neighbors = np.full(Npix, np.nan)

for i in range(Npix):
    if np.isnan(AL[i]):
        continue
    neighbors = hp.get_all_neighbours(nside, i)
    neighbors = neighbors[neighbors >= 0]
    neighbors = neighbors[~np.isnan(AL[neighbors])]
    if len(neighbors) == 0:
        continue

    diffs = AL[i] - AL[neighbors]
    diff_errs = np.sqrt(AL_err[i]**2 + AL_err[neighbors]**2)
    sigmas = diffs / diff_errs

    # the WEAKEST significance vs any neighbor - i.e. is this pixel
    # significantly higher than ALL its neighbors, or just some?
    min_sigma_vs_neighbors[i] = np.min(sigmas)

# Rank pixels by how significantly they stand out from their WORST-matching neighbor
idx_sorted = np.argsort(min_sigma_vs_neighbors)[::-1]  # descending

print("Top 10 most sharply-peaked pixels (highest AL relative to ALL their neighbors):")
for idx in idx_sorted[:10]:
    l, b = hp.pix2ang(nside, idx, lonlat=True)
    print(f"  pixel {idx}: AL={AL[idx]:.4f}, min sigma vs neighbors={min_sigma_vs_neighbors[idx]:.2f}, (l,b)=({l:.2f},{b:.2f})")
