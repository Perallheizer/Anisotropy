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

DIR = "./LCDM_aniso/sne_gc"
RESULTS_DIR = "results"

data = pd.read_csv(f"{DIR}/{RESULTS_DIR}/AL_null_ceph_redis_{cut_name}.csv")
obs_data = pd.read_csv(f"{DIR}/{RESULTS_DIR}/H0_ceph_{cut_name}.csv")

idx = data["idx_max"].values

H0_p = obs_data["H0_p"].values
H0_m = obs_data["H0_m"].values

AL = 2 * (H0_p - H0_m) / (np.abs(H0_p) + np.abs(H0_m))
idx_obs = np.nanargmax(AL)

print("Index = ", idx_obs)
print(np.sum(idx == idx_obs))

counts = Counter(idx)

rank = sorted(counts.values(), reverse=True)
#print(counts[idx_obs])
#print(rank[:20])













    
