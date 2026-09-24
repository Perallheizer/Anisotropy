import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import Counter
import healpy

cut_name = sys.argv[1]

DIR = "/home/shubhambarua/Downloads/Ceph/Pade_aniso/sne"
RESULTS_DIR = "results"
DATA_DIR = "./data"

data = pd.read_csv(f"{DIR}/{RESULTS_DIR}/AL_null_ceph_redistrib_{cut_name}.csv")

l_vals = data["l_max"].values
b_vals = data["b_max"].values

l_rounded = np.round(l_vals).astype(int)
b_rounded = np.round(b_vals).astype(int)

pairs = list(zip(l_rounded, b_rounded))
unique_pairs = set(pairs)

print(f"Total mocks: {len(pairs)}")
print(f"Unique (l,b) pairs (rounded to nearest degree): {len(unique_pairs)}")

counts = Counter(pairs)
print("\nMost common (l,b) directions:")
for (l, b), n in counts.most_common(10):
    print(f"  ({l}, {b}): {n} mocks")


real_l, real_b = 275.62, 30   # your actual observed direction, adjust as needed

real_coord = SkyCoord(l=real_l*u.degree, b=real_b*u.degree, frame='galactic')

top_dirs = [(242,0),(264,-10),(231,10),(332,30),(354,48),(343,30),(315,14),(326,5),(321,0),(135,72)]
for l, b in top_dirs:
    c = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    sep = real_coord.separation(c).degree
    print(f"({l},{b}): {sep:.1f} deg from real direction")
    
    
print("\n Checking all spots for closeness to real position:")


mock_coords = SkyCoord(l=l_vals*u.degree, b=b_vals*u.degree, frame='galactic')
seps = real_coord.separation(mock_coords).degree

print(f"Mean separation: {seps.mean():.1f} deg")
print(f"Median separation: {np.median(seps):.1f} deg")
print(f"Fraction within 20 deg: {np.mean(seps < 20)*100:.1f}%")
print(f"Fraction within 40 deg: {np.mean(seps < 40)*100:.1f}%")
print(f"Minimum separation across all 1000 mocks: {seps.min():.1f} deg")

print(healpy.nside2pixarea(8, degrees=True))













    
