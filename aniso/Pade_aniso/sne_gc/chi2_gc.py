import numpy as np
from cosmo_model import Model

k = 1.136

def chisq_gc(params, indices, z, sig_L, f, logT_err, tau): 
    
    #H0, omega_m, k, s, sig_int = params
    H0, q0, s, sig_int = params
    #H0, omega_m = params
    """Compute χ² for a subset of clusters"""
    if len(indices) == 0:
       return np.inf
    
    z = z[indices]
    sig_L = sig_L[indices]
    f = f[indices]
    logT_err = logT_err[indices]
    tau = tau[indices] 
    

    cosmo = Model(H0, q0)

    dL = np.vectorize(cosmo.dLz)
    Ez = np.vectorize(cosmo.Ez)
    
    # --- 1. Distance ------------------------------------
    DL_Mpc = dL(z)                        # Mpc
    DL_cm  = DL_Mpc * 3.086e24            # convert Mpc → cm

    # --- 2. Flux ----------------------------------------
    # f_obs is in units of (10^-12 erg/s/cm^2)
    f = f * 1e-12                     # erg/s/cm^2

    # --- 3. Luminosity Lx (erg/s) -----------------------
    Lx = 4 * np.pi * (DL_cm**2) * f       # (10^44) erg/s
    sig_L_true = (sig_L*Lx)/100
    
    #### Find actual value of sigma_L
    
    # Error: sigma_L in %
    logL_err = sig_L_true / (Lx * np.log(10))  # error in log10(Lx)

    # --- 4. Lx scaled by E(z) ---------------------------
    Lx_prime = Lx / Ez(z)                 # erg/s
    # Convert to units of 10^44 erg/s
    log_Lx_prime = np.log10(Lx_prime) - 44 ## Check

    # --- 5. Theoretical Lx from scaling relation --------
    # log Lx(th) in SAME UNITS (10^44 erg/s)
    # Lx_th = k * tau^s   ⇒ log10(Lx_th) = log10(k) + s*log10(tau)
    log_Lx_th = np.log10(k) + s * np.log10(tau)

    # --- 6. Variance -------------------------------------
    sigma2 = (logL_err**2 + (s**2) * (logT_err**2) + sig_int**2)

    # --- 7. Chi-square -----------------------------------
    csq = np.sum((log_Lx_prime - log_Lx_th)**2 / sigma2)

    return csq + np.sum(np.log(2*np.pi*sigma2))
    
    
    
