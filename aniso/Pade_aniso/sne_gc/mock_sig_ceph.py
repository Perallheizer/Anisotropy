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
from bestfit_ceph import (fit_joint, v_grid, Npix,
                      hem_plus_sn, hem_minus_sn, hem_plus_gc, hem_minus_gc,
                      z_sn, mb, mb_err, mu_ceph, calibrator, cov, z_gc, sig_L, f, logT_err, tau)
from multiprocessing import Pool, Manager

cut_name = sys.argv[1]

seed_start = int(sys.argv[2])
seed_end = int(sys.argv[3])

DIR = "./aniso/Pade_aniso/sne_gc"

def compute_AL(theta_p, theta_m):
    return 2 * (theta_p - theta_m) / (np.abs(theta_p) + np.abs(theta_m))

def make_mock_sn(z_sn, mb, mb_err, mu_ceph, calibrator, cov, rng):
    N = len(z_sn)
    idx = rng.permutation(N)
    return (z_sn[idx], mb[idx], mb_err[idx], mu_ceph[idx], calibrator[idx],
            cov[np.ix_(idx, idx)])

def make_mock_gc(z_gc, sig_L, f, logT_err, tau, rng):
    N = len(z_gc)
    idx = rng.permutation(N)
    return z_gc[idx], sig_L[idx], f[idx], logT_err[idx], tau[idx]

def run_scan_joint(z_sn, mb, mb_err, mu_ceph, calibrator, cov, z_gc, sig_L, f, logT_err, tau, hem_plus_sn, hem_minus_sn, hem_plus_gc, hem_minus_gc, Npix):
    H0_p = np.zeros(Npix)
    H0_m = np.zeros(Npix)
    for i in range(Npix):
        best = fit_joint(hem_plus_sn[i], hem_minus_sn[i], hem_plus_gc[i], hem_minus_gc[i], z_sn, mb, cov, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau)
        H0_p[i] = best[0]
        H0_m[i] = best[6]
        if i%200 == 0:
           print(f"[mock progress] pixel {i+1}/{Npix} done", flush=True)
    return H0_p, H0_m

#data = pd.read_csv(f"{DIR}/results/H0_ceph_{cut_name}.csv")
#AL_obs = np.nanmax(compute_AL(data["H0_p"].values, data["H0_m"].values))

#n_mock = 1000
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("Using cores:", n_cores, flush=True)
print(f"Seed range: {seed_start} -> {seed_end - 1}", flush=True)
out_path = (f"{DIR}/results/"
            f"AL_null_ceph_{cut_name}_{seed_start}_{seed_end-1}.csv")
            
if not os.path.exists(out_path):
    pd.DataFrame(columns=[
         "seed",
         "AL_null",
         "H0_p_max",
         "H0_m_max"
         ]).to_csv(out_path,index=False)

def one_mock(args):

    seed, lock = args
    t_start = time.time()
    print(f"[seed {seed}] starting, PID={os.getpid()}", flush=True)
    rng = np.random.default_rng(seed)

    z_sn_m, mb_m, mb_err_m, mu_ceph_m, cal_m, cov_m = make_mock_sn(
        z_sn, mb, mb_err, mu_ceph, calibrator, cov, rng
    )
    z_gc_m, sig_L_m, f_m, logT_err_m, tau_m = make_mock_gc(
        z_gc, sig_L, f, logT_err, tau, rng)

    H0_p_m, H0_m_m = run_scan_joint(
        z_sn_m, mb_m, mb_err_m, mu_ceph_m, cal_m, cov_m,
        z_gc_m, sig_L_m, f_m, logT_err_m, tau_m,
        hem_plus_sn, hem_minus_sn, hem_plus_gc, hem_minus_gc, Npix
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
         row.to_csv(out_path, mode='a', header=False, index = False)
    elapsed = time.time() - t_start
    print(f"[seed {seed}] DONE in {elapsed:.1f}s, AL={AL_val:.4f}", flush=True)
    return seed, AL_val


if __name__ == "__main__":

    if os.path.exists(out_path):
        done_df = pd.read_csv(out_path)
        completed_seeds = set(done_df["seed"])
    else:
        completed_seeds = set()

    remaining_seeds = [s for s in range(seed_start, seed_end) if s not in completed_seeds]
    print(f"Already done: {len(completed_seeds)}, remaining: {len(remaining_seeds)}", flush=True)
    
    manager = Manager()
    lock = manager.Lock()
    tasks = [(s, lock) for s in remaining_seeds]

    with Pool(processes=n_cores) as pool:
         for seed, result in pool.imap_unordered(one_mock, tasks):
             print(f"mock seed {seed} done", flush=True)


