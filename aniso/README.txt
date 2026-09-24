1. LCDM_aniso contains code for LCDM model and Pade_aniso contains code for the Pade cosmography.
2. gc is for galaxy cluster only analysis, sne_gc is for SNe+GC analysis and sne_rev is for SNe-only analysis
3. In sne_gc:
     a. bestfit_gc.py, mock_sig_gc.py, chi2_gc.py and chi2_sne.py correspond to the case when k is fixed
     b. bestfit_ceph.py, mock_sig_ceph.py and chi2_gc_k.py and chi2_ceph.py correspond to the case when Cepheids are used for calibration.
     c. mock_ceph_sne.py correspond to the case when Cepheid-hosts are shuffled while keeping all othe SNe fixed.
     
