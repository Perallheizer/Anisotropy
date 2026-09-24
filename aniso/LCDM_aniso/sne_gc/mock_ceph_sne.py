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
from multiprocessing import Pool, Manager
from chi2_gc_k import chisq_gc

cut_name = sys.argv[1]
seed_start = int(sys.argv[2])
seed_end = int(sys.argv[3])

DIR = "./aniso/LCDM_aniso/sne_gc"

# ---- Load grid ----
grid = pd.read_csv(f"{DIR}/data/scan_grid_equatorial.csv")
v_grid = np.vstack([grid["vx_gal"], grid["vy_gal"], grid["vz_gal"]]).T
v_grid = v_grid / np.linalg.norm(v_grid, axis=1)[:, None]
grid_l = grid["l_g"].values
grid_b = grid["b_g"].values
Npix = len(v_grid)

# ---- Load SNe ----
pantheon = np.loadtxt(f"{DIR}/data/pantheon_{cut_name}.txt", skiprows=1)
vx_sn, vy_sn, vz_sn = pantheon[:,0], pantheon[:,1], pantheon[:,2]
z_sn = pantheon[:,3]
mb = pantheon[:,4]
mu_ceph = pantheon[:,6]
calibrator = pantheon[:,7]

cov = np.loadtxt(f"{DIR}/data/pantheon_cov_{cut_name}.txt")
N = len(z_sn)
cov = cov.reshape(N, N)

calib_idx = np.where(calibrator == 1)[0]
print(f"Total SNe: {N}, calibrators: {len(calib_idx)}", flush=True)


cluster = np.loadtxt(f"{DIR}/data/cluster_data.txt", skiprows=1)
vx_gc, vy_gc, vz_gc = cluster[:,0], cluster[:,1], cluster[:,2]
z_gc = cluster[:,3]
sig_L = cluster[:,4]
f = cluster[:,5]
logT_err = cluster[:,6]
tau = cluster[:,7]

v_gc = np.vstack([vx_gc, vy_gc, vz_gc]).T
v_gc = v_gc / np.linalg.norm(v_gc, axis=1)[:, None]

dot_grid_gc = np.dot(v_grid, v_gc.T)
hem_plus_gc = []
hem_minus_gc = []
for i in range(Npix):
    d_gc = dot_grid_gc[i]
    hem_plus_gc.append(np.where(d_gc > 0)[0])
    hem_minus_gc.append(np.where(d_gc <= 0)[0])

# ---- Fitting (same as bestfit.py, no hesse for speed) ----
from cosmo_model import Model
from chi2_ceph import chisq_sne
from iminuit import Minuit

def chi2_wrapper(H0_p, om_p, M_p, k_p, s_p, sig_p,
    H0_m, om_m, M_m, k_m, s_m, sig_m,

    idx_sn_p, idx_sn_m,
    idx_gc_p, idx_gc_m,

    z_sn, mb, L, mu_ceph, calibrator,
    z_gc, sig_L, f, logT_err, tau):
    
    params1 = [H0_p, om_p, M_p, H0_m, om_m, M_m]
    
    chi_sn = chisq_sne(params1, idx_sn_p, idx_sn_m, z_sn, mb, L, mu_ceph, calibrator)
    chi_gc_p = chisq_gc([H0_p, om_p, k_p, s_p, sig_p], idx_gc_p, z_gc, sig_L, f, logT_err, tau)
    chi_gc_m = chisq_gc([H0_m, om_m, k_m, s_m, sig_m], idx_gc_m, z_gc, sig_L, f, logT_err, tau)
    
    return (chi_sn + chi_gc_p + chi_gc_m)
    
def fit_joint(idx_sn_p, idx_sn_m, idx_gc_p, idx_gc_m, z_sn, mb, L, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau): 
    
    func = lambda H0_p, om_p, M_p, k_p, s_p, sig_p, H0_m, om_m, M_m, k_m, s_m, sig_m: chi2_wrapper(H0_p, om_p, M_p, k_p, s_p, sig_p, H0_m, om_m, M_m, k_m, s_m, sig_m, idx_sn_p, idx_sn_m, idx_gc_p, idx_gc_m, z_sn, mb, L, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau)
       
    m = Minuit(func, H0_p=70, om_p=0.3, M_p=-19.3, k_p = 1.136, s_p=2.102, sig_p=0.24, H0_m=70, om_m=0.3, M_m=-19.3, k_m = 1.136, s_m=2.102, sig_m=0.24)
    m.limits = [(0,150), (0, 1), (-20,-18), (1e-4, 3), (1e-4, 5), (1e-4, 2),(0,150), (0, 1), (-20,-18), (1e-4, 3), (1e-4, 5), (1e-4, 2)]
    m.errordef = 1

    m.migrad()
    #m.hesse()

    best = np.array([m.values["H0_p"], m.values["om_p"], m.values["M_p"], m.values["k_p"], m.values["s_p"], m.values["sig_p"], m.values["H0_m"], m.values["om_m"], m.values["M_m"], m.values["k_m"], m.values["s_m"], m.values["sig_m"]])
    return best

def compute_AL(theta_p, theta_m):
    return 2 * (theta_p - theta_m) / (np.abs(theta_p) + np.abs(theta_m))

def shuffle_calibrator_positions(vx_sn, vy_sn, vz_sn, calib_idx, rng):
    vx_new, vy_new, vz_new = vx_sn.copy(), vy_sn.copy(), vz_sn.copy()
    shuffled = rng.permutation(calib_idx)
    vx_new[calib_idx] = vx_sn[shuffled]
    vy_new[calib_idx] = vy_sn[shuffled]
    vz_new[calib_idx] = vz_sn[shuffled]
    return vx_new, vy_new, vz_new
    
def run_scan(v_sn, z_sn, mb, cov, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau, hem_plus_gc, hem_minus_gc, v_grid, Npix):
    v_sn = v_sn / np.linalg.norm(v_sn, axis=1)[:, None]
    dot = np.dot(v_grid, v_sn.T)
    H0_p = np.zeros(Npix)
    H0_m = np.zeros(Npix)
    M_p = np.zeros(Npix)
    M_m = np.zeros(Npix)
    for i in range(Npix):
        d = dot[i]
        idx_plus_sn = np.where(d > 0)[0]
        idx_minus_sn = np.where(d <= 0)[0]
        idx_all_sn = np.concatenate([idx_plus_sn, idx_minus_sn])
        
        cov_sub = cov[np.ix_(idx_all_sn, idx_all_sn)]
        L = np.linalg.cholesky(cov_sub)
        
        best = fit_joint(idx_plus_sn, idx_minus_sn, hem_plus_gc[i], hem_minus_gc[i], z_sn, mb, L, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau)
        H0_p[i] = best[0]
        H0_m[i] = best[6]
        M_p[i] = best[2]
        M_m[i] = best[8]
    return H0_p, H0_m, M_p, M_m

# ---- Real-data observed AL (from existing results) ----
#data = pd.read_csv(f"{DIR}/results/H0_ceph_{cut_name}.csv")
#AL_obs = np.nanmax(compute_AL(data["H0_p"].values, data["H0_m"].values))

#n_mock = 1000
n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print("Using cores:", n_cores, flush=True)
print(f"Seed range: {seed_start} -> {seed_end - 1}", flush=True)
out_path = (f"{DIR}/results/"
            f"AL_null_ceph_redis_{cut_name}_{seed_start}_{seed_end-1}.csv")
            
if not os.path.exists(out_path):
    pd.DataFrame(columns=[
        "seed",
        "AL_null",
        "idx_max",
        "l_max",
        "b_max",
        "H0_p_max",
        "H0_m_max",
        "M_p_max",
        "M_m_max",
        "n_calib_plus",
        "n_calib_minus",
        "n_total_plus",
        "n_total_minus",
    ]).to_csv(out_path, index=False)

def one_mock(args):
    seed, lock = args
    t_start = time.time()
    print(f"[seed {seed}] starting, PID={os.getpid()}", flush=True)
    rng = np.random.default_rng(seed)

    vx_m, vy_m, vz_m = shuffle_calibrator_positions(vx_sn, vy_sn, vz_sn, calib_idx, rng)
    v_sn_m = np.vstack([vx_m, vy_m, vz_m]).T

    H0_p_m, H0_m_m, M_p_m, M_m_m = run_scan(v_sn_m, z_sn, mb, cov, mu_ceph, calibrator, z_gc, sig_L, f, logT_err, tau, hem_plus_gc, hem_minus_gc, v_grid, Npix)

    AL_m = compute_AL(H0_p_m, H0_m_m)
    idx_max = np.nanargmax(AL_m)
    H0_p_at_max = H0_p_m[idx_max]
    H0_m_at_max = H0_m_m[idx_max]
    M_p_at_max = M_p_m[idx_max]
    M_m_at_max = M_m_m[idx_max]
    AL_val = AL_m[idx_max]
    
    # recompute hemisphere split at the peak pixel, using the SAME shuffled positions
    v_sn_m_norm = v_sn_m / np.linalg.norm(v_sn_m, axis=1)[:, None]
    d_at_max = np.dot(v_grid[idx_max], v_sn_m_norm.T)

    idx_plus_at_max = np.where(d_at_max > 0)[0]
    idx_minus_at_max = np.where(d_at_max <= 0)[0]

    n_calib_plus = np.sum(calibrator[idx_plus_at_max] == 1)
    n_calib_minus = np.sum(calibrator[idx_minus_at_max] == 1)
    
    n_total_plus = len(idx_plus_at_max)
    n_total_minus = len(idx_minus_at_max)

    l_max = np.degrees(grid_l[idx_max])
    b_max = np.degrees(grid_b[idx_max])

    row = pd.DataFrame({
        "seed": [seed],
        "AL_null": [AL_val],
        "idx_max": [idx_max],
        "l_max": [l_max],
        "b_max": [b_max],
        "H0_p_max": [H0_p_at_max],
        "H0_m_max": [H0_m_at_max],
        "M_p_max": [M_p_at_max],
        "M_m_max": [M_m_at_max],
        "n_calib_plus": [n_calib_plus],
        "n_calib_minus": [n_calib_minus],
        "n_total_plus": [n_total_plus],
        "n_total_minus": [n_total_minus],
    })
    with lock:
        row.to_csv(out_path, mode='a', header=False, index=False)

    elapsed = time.time() - t_start
    print(f"[seed {seed}] DONE in {elapsed:.1f}s, AL={AL_val:.4f}", flush=True)
    return seed, AL_val

if __name__ == "__main__":
    if os.path.exists(out_path):
        done_df = pd.read_csv(out_path)
        completed_seeds = set(done_df["seed"])
    else:
        completed_seeds = set()

    remaining_seeds = [s for s in range(seed_start,seed_end) if s not in completed_seeds]
    print(f"Already done: {len(completed_seeds)}, remaining: {len(remaining_seeds)}", flush=True)

    manager = Manager()
    lock = manager.Lock()
    tasks = [(s, lock) for s in remaining_seeds]

    with Pool(processes=n_cores) as pool:
        for seed, result in pool.imap_unordered(one_mock, tasks):
            print(f"mock seed {seed} done", flush=True)


















