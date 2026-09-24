import time
print("=== SCRIPT STARTED ===", time.strftime("%H:%M:%S"), flush=True)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
from bestfit import (fit_joint, v_grid, Npix,
                      hem_plus_sn, hem_minus_sn,
                      z_sn, mb, mb_err, mu_ceph, calibrator, cov)
from multiprocessing import Pool, Manager

cut_name = sys.argv[1]

DIR = "./aniso/LCDM_aniso/sne_rev"

def compute_AL(theta_p, theta_m):
    return 2 * (theta_p - theta_m) / (np.abs(theta_p) + np.abs(theta_m))

def make_mock_sn(z_sn, mb, mb_err, mu_ceph, calibrator, cov, rng):
    N = len(z_sn)
    idx = rng.permutation(N)
    return (z_sn[idx], mb[idx], mb_err[idx], mu_ceph[idx], calibrator[idx],
            cov[np.ix_(idx, idx)])

def run_scan_sn(z_sn, mb, mb_err, mu_ceph, calibrator, cov,
                 hem_plus_sn, hem_minus_sn, Npix):
    H0_p = np.zeros(Npix)
    H0_m = np.zeros(Npix)
    for i in range(Npix):
        best = fit_joint(hem_plus_sn[i], hem_minus_sn[i], z_sn, mb, cov, mu_ceph, calibrator)
        H0_p[i] = best[0]
        H0_m[i] = best[3]
        if i == 0 or i == Npix-1:
           print(f"[mock progress] pixel {i+1}/{Npix} done", flush=True)
    return H0_p, H0_m

data = pd.read_csv(f"{DIR}/results/H0_anisotropy_{cut_name}.csv")
AL_obs = np.nanmax(compute_AL(data["H0_p"].values, data["H0_m"].values))

n_mock = 1000
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("Using cores:", n_cores, flush=True)
out_path = f"{DIR}/results/AL_null_{cut_name}.csv"

def one_mock(args):
    
    seed, lock = args
    t_start = time.time()
    print(f"[seed {seed}] starting, PID={os.getpid()}", flush=True)
    rng = np.random.default_rng(seed)

    z_m, mb_m, mb_err_m, mu_ceph_m, cal_m, cov_m = make_mock_sn(
        z_sn, mb, mb_err, mu_ceph, calibrator, cov, rng
    )

    H0_p_m, H0_m_m = run_scan_sn(
        z_m, mb_m, mb_err_m,
        mu_ceph_m, cal_m, cov_m,
        hem_plus_sn, hem_minus_sn, Npix
    )

    AL_m =  compute_AL(H0_p_m, H0_m_m)
    idx_max = np.nanargmax(AL_m)
    
    AL_val = AL_m[idx_max]
    H0_p_at_max = H0_p_m[idx_max]
    H0_m_at_max = H0_m_m[idx_max]

    row = pd.DataFrame({
        "seed": [seed],
        "AL_null": [AL_val],
        "H0_p_max": [H0_p_at_max],
        "H0_m_max": [H0_m_at_max],
    })
    
    with lock:
         row.to_csv(out_path, mode='a', header=not os.path.exists(out_path), index = False)
    elapsed = time.time() - t_start
    print(f"[seed {seed}] DONE in {elapsed:.1f}s, AL={AL_val:.4f}", flush=True)
    return seed, AL_val


if __name__ == "__main__":

    if os.path.exists(out_path):
        done_df = pd.read_csv(out_path)
        completed_seeds = set(done_df["seed"])
    else:
        completed_seeds = set()

    remaining_seeds = [s for s in range(n_mock) if s not in completed_seeds]
    print(f"Already done: {len(completed_seeds)}, remaining: {len(remaining_seeds)}", flush=True)

    manager = Manager()
    lock = manager.Lock()
    tasks = [(s, lock) for s in remaining_seeds]

    with Pool(processes=n_cores) as pool:
         for seed, result in pool.imap_unordered(one_mock, tasks):
             print(f"mock seed {seed} done", flush=True)

    final_df = pd.read_csv(out_path)
    AL_null = final_df["AL_null"].values
    
    p_global = np.mean(AL_null >= AL_obs)
    sigma_global = norm.isf(p_global)

    print("Observed AL:", AL_obs)
    print("Global p-value:", p_global)
    print("Global significance (sigma):", sigma_global)

    with open(f"{DIR}/results/AL_significance_summary_{cut_name}.txt", "w") as f:
         f.write(f"Cut: {cut_name}\n")
         f.write(f"Observed AL: {AL_obs}\n")
         f.write(f"Global p-value: {p_global}\n")
         f.write(f"Global significance (sigma): {sigma_global}\n")



