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
from bestfit import (fit_params, v_grid, Npix, hem_plus_gc, hem_minus_gc,
                       z_gc, sig_L, f, logT_err, tau)
from multiprocessing import Pool, Manager

cut_name = sys.argv[1]
k_fixed = float(sys.argv[2])

DIR = "./aniso/Pade_aniso/gc"

def compute_AL(theta_p, theta_m):
    return 2 * (theta_p - theta_m) / (np.abs(theta_p) + np.abs(theta_m))

def make_mock_gc(z_gc, sig_L, f, logT_err, tau, rng):
    N = len(z_gc)
    idx = rng.permutation(N)
    return z_gc[idx], sig_L[idx], f[idx], logT_err[idx], tau[idx]

def run_scan_gc(z_gc, sig_L, f, logT_err, tau, hem_plus_gc, hem_minus_gc, Npix, k):
    H0_p = np.zeros(Npix)
    H0_m = np.zeros(Npix)
    s_p = np.zeros(Npix)    
    s_m = np.zeros(Npix)
    sig_p = np.zeros(Npix)
    sig_m = np.zeros(Npix)    

    H0err_p = np.zeros(Npix)
    H0err_m = np.zeros(Npix)
    s_err_p = np.zeros(Npix)
    s_err_m = np.zeros(Npix)
    sig_err_p = np.zeros(Npix)
    sig_err_m = np.zeros(Npix)

    for i in range(Npix):
        best_p, err_p = fit_params(hem_plus_gc[i], z_gc, sig_L, f, logT_err, tau, k)
        best_m, err_m = fit_params(hem_minus_gc[i], z_gc, sig_L, f, logT_err, tau, k)
        
        H0_p[i] = best_p[0]
        H0_m[i] = best_m[0]
        s_p[i] = best_p[1]
        s_m[i] = best_m[1]
        sig_p[i] = best_p[2]
        sig_m[i] = best_m[2]
       
        H0err_p[i] = err_p[0]
        H0err_m[i] = err_m[0]
        s_err_p[i] = err_p[1]
        s_err_m[i] = err_m[1]
        sig_err_p[i] = err_p[2]
        sig_err_m[i] = err_m[2]
        
        if i%300 == 0:
           print(f"[mock progress] pixel {i+1}/{Npix} done", flush=True)
    return H0_p, H0_m, H0err_p, H0err_m, s_p, s_m, s_err_p, s_err_m, sig_p, sig_m, sig_err_p, sig_err_m

data = pd.read_csv(f"{DIR}/results/H0_gc_{cut_name}.csv")
AL_obs = np.nanmax(compute_AL(data["H0_p"].values, data["H0_m"].values))

n_mock = 1000
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("Using cores:", n_cores, flush=True)
out_path = f"{DIR}/results/AL_null_gc_{cut_name}.csv"

def one_mock(args):

    seed, lock = args
    t_start = time.time()
    print(f"[seed {seed}] starting, PID={os.getpid()}", flush=True)
    rng = np.random.default_rng(seed)

    z_gc_m, sig_L_m, f_m, logT_err_m, tau_m = make_mock_gc(
        z_gc, sig_L, f, logT_err, tau, rng)

    H0_p_m, H0_m_m, H0err_p_m, H0err_m_m, s_p_m, s_m_m, s_err_p_m, s_err_m_m, sig_p_m, sig_m_m, sig_err_p_m, sig_err_m_m = run_scan_gc(
        z_gc_m, sig_L_m, f_m, logT_err_m, tau_m,
        hem_plus_gc, hem_minus_gc, Npix, k_fixed
    )

    AL_m =  compute_AL(H0_p_m, H0_m_m)
    idx_max = np.nanargmax(AL_m)
    
    AL_val = AL_m[idx_max]
    H0_p_at_max = H0_p_m[idx_max]
    H0_m_at_max = H0_m_m[idx_max]
    H0err_p_at_max = H0err_p_m[idx_max]
    H0err_m_at_max = H0err_m_m[idx_max]
    s_p_at_max = s_p_m[idx_max]
    s_m_at_max = s_m_m[idx_max]
    s_err_p_at_max = s_err_p_m[idx_max]
    s_err_m_at_max = s_err_m_m[idx_max]
    sig_p_at_max = sig_p_m[idx_max]
    sig_m_at_max = sig_m_m[idx_max]
    sig_err_p_at_max = sig_err_p_m[idx_max]
    sig_err_m_at_max = sig_err_m_m[idx_max]


    row = pd.DataFrame({
        "seed": [seed],
        "AL_null": [AL_val],
        "H0_p_max": [H0_p_at_max],
        "H0_m_max": [H0_m_at_max],
        "H0_p_err": [H0err_p_at_max],
        "H0_m_err": [H0err_m_at_max],
        "s_p_max": [s_p_at_max],
        "s_m_max": [s_m_at_max],
        "s_p_err": [s_err_p_at_max],
        "s_m_err": [s_err_m_at_max],
        "sig_p_max": [sig_p_at_max],
        "sig_m_max": [sig_m_at_max],
        "sig_p_err": [sig_err_p_at_max],
        "sig_m_err": [sig_err_m_at_max],
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

    with open(f"{DIR}/results/AL_significance_summary_{cut_name}_gc.txt", "w") as f:
         f.write(f"Cut: {cut_name}\n")
         f.write(f"Observed AL: {AL_obs}\n")
         f.write(f"Global p-value: {p_global}\n")
         f.write(f"Global significance (sigma): {sigma_global}\n")



