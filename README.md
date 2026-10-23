# Anisotropy
Codes used for 2604.04408

README.txt contains descriptions of the files. Data is not included as they are publicly available. For galaxy clusters data refer to 2004.03305 and the PantheonPlus dataset is available at https://github.com/CobayaSampler/sn_data.

analysis.py - Writes a file and plots the anisotropy sky map for the real data

cepheid_count.py - Counts the number of cepheids and gives the value of M_B at a particular specified direction

isotropic.py - Plots the histograms of the mock realizations

M_fitting.py - Computes M_B in the plus and minus hemispheres

norm_AL.py - Does the same as analysis.py but foe the normalized AL statistic

sep.py - Computes separation between two directions

summary.py - Computes p-value from mock realizations

bin_count.py - counts number of mocks for a particular redshift cut for which the maximum AL lies in the same grid as the observed maximum AL
