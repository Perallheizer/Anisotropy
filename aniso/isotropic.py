import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cut_name = sys.argv[1]

DIR = "./LCDM_aniso/gc_k"
RESULTS_DIR = "./results"
OUT_DIR = f"./summary/{DIR}"

data = pd.read_csv(f"{DIR}/{RESULTS_DIR}/AL_null_gc_{cut_name}.csv")
AL_values = data["AL_null"]

def read_observed_al(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Observed AL:"):
                return float(line.split(":")[1].strip())
    raise ValueError(f"Could not find 'Observed AL' in {path}")

AL_obs = read_observed_al(f"{DIR}/{RESULTS_DIR}/AL_significance_summary_{cut_name}_gc.txt")

def read_p_value(path):
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Global p-value:"):
                return float(line.split(":")[1].strip())
    raise ValueError(f"Could not find 'Global p-value' in {path}")

p_global = read_p_value(f"{DIR}/{RESULTS_DIR}/AL_significance_summary_{cut_name}_gc.txt")

fig, ax = plt.subplots(figsize=(5,3))

ax.hist(AL_values, bins = 30, 
        alpha=0.5, density = True,
        edgecolor='black',
        linewidth=0.8, 
        label = "Mock realizations")
ax.axvline(AL_obs, color = 'red', ls = '--', label=f"Observed AL = {AL_obs:.3f}")
ax.plot([], [], ' ', label=f"p-value={p_global:.3f}")
ax.set_xlabel(r"$AL_\mathrm{max}$")
ax.set_ylabel("Proportion")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/AL_hist_{cut_name}_gc.pdf", bbox_inches='tight')
plt.show()
