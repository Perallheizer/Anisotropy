import numpy as np
import pandas as pd
import healpy as hp

nside = 8

npix = hp.nside2npix(nside)
print("Total pixels:", npix)

ipix = np.arange(npix)

theta, phi = hp.pix2ang(nside, ipix)

lon = phi
lat = np.pi/2 - theta

vx, vy, vz = hp.pix2vec(nside, ipix)

df = pd.DataFrame({
    "l_g": lon,
    "b_g": lat,
    "vx_gal": vx,
    "vy_gal": vy,
    "vz_gal": vz
})

df.to_csv("./data/scan_grid_equatorial.csv", index=False)
print("Saved grid of shape:", df.shape)
