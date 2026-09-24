import numpy as np
import pandas as pd
import healpy as hp

nside = 8
cut_name = "A"   # Padé, z<=0.1

data = pd.read_csv(f"./Pade_aniso/sne/results/H0_anisotropy_{cut_name}.csv")   # Pade version
H0_p, H0_m = data["H0_p"].values, data["H0_m"].values
H0_p_err, H0_m_err = data["H0_p_err"].values, data["H0_m_err"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
AL_err = 4 * np.sqrt((H0_p_err**2)*(H0_m**2) + (H0_m_err**2)*(H0_p**2))/(np.abs(H0_p) + np.abs(H0_m))**2


idx_max = np.nanargmax(AL)
l_max, b_max = hp.pix2ang(nside, idx_max, lonlat=True)
print(f"Peak pixel: {idx_max}, AL = {AL[idx_max]:.4f}+/- {AL_err[idx_max]:.4f}, "
      f"(l,b) = {l_max:.2f}, "
      f"{b_max:.2f}")

# Find the pixel closest to the "other" direction (275.62, 30)
other_l, other_b = 275.62, 30
other_vec = hp.ang2vec(other_l, other_b, lonlat=True)
idx_other = hp.vec2pix(nside, *other_vec)
print(f"'Other' direction pixel: {idx_other}, AL = {AL[idx_other]:.4f}+/- {AL_err[idx_other]:.4f}")

diff = AL[idx_max] - AL[idx_other]
diff_err = np.sqrt(AL_err[idx_max]**2 + AL_err[idx_other]**2)
sigma_diff = diff / diff_err
print(f"\nDifference in AL between peak and 'other' direction: {diff:.4f} +/- {diff_err:.4f}")
print(f"Significance of this difference: {sigma_diff:.2f} sigma")


print("\n--- Neighbors of peak pixel, in significance terms ---")
neighbors = hp.get_all_neighbours(nside, idx_max)
neighbors = neighbors[neighbors >= 0]
for n in neighbors:
    d = AL[idx_max] - AL[n]
    d_err = np.sqrt(AL_err[idx_max]**2 + AL_err[n]**2)
    sig = d / d_err if d_err > 0 else np.nan
    print(f"  pixel {n}: AL = {AL[n]:.4f} +/- {AL_err[n]:.4f}, "
          f"difference from peak = {sig:.2f} sigma")

