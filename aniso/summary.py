import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

cut_name = sys.argv[1]

DIR = "./LCDM_aniso/sne_gc/results"

def compute_AL(theta_p, theta_m):
    return 2 * (theta_p - theta_m) / (np.abs(theta_p) + np.abs(theta_m))

final_df = pd.read_csv(f"{DIR}/AL_null_ceph_redis_{cut_name}.csv") # Mocks
AL_null = final_df["AL_null"].values

data = pd.read_csv(f"{DIR}/H0_ceph_{cut_name}.csv")
AL_obs = np.nanmax(compute_AL(data["H0_p"].values, data["H0_m"].values))

p_global = np.mean(AL_null >= AL_obs)
sigma_global = norm.isf(p_global)

print("Observed AL:", AL_obs)
print("Global p-value:", p_global)
print("Global significance (sigma):", sigma_global)

with open(f"{DIR}/AL_significance_summary_{cut_name}_ceph_redis.txt", "w") as f:
     f.write(f"Cut: {cut_name}\n")
     f.write(f"Observed AL: {AL_obs}\n")
     f.write(f"Global p-value: {p_global}\n")
     f.write(f"Global significance (sigma): {sigma_global}\n")
     
print("Number of null values:", len(AL_null))
print("Observed AL:", AL_obs)
print("Null median:", np.median(AL_null))
print("Null 95th percentile:", np.percentile(AL_null, 95))
print("Null 99th percentile:", np.percentile(AL_null, 99))
print("Number >= observed:", np.sum(AL_null >= AL_obs))

print(np.min(AL_null))
print(np.median(AL_null))
print(np.max(AL_null))
