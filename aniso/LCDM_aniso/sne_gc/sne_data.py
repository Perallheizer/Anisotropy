import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

data = pd.read_csv('/home/shubhambarua/Supernova Datasets/sn_data-master/PantheonPlus/Pantheon+SH0ES.dat', sep = r'\s+')
cov_data = np.loadtxt(r"/home/shubhambarua/Supernova Datasets/sn_data-master/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov", skiprows = 1)

z_min_ceph = 0.00122
z_max_ceph = 0.01682

z_min = 0
z_max = 3

zhd = np.array(data['zHD'])
R = np.array(data['RA'])
D = np.array(data['DEC'])

coords_eq = SkyCoord(ra=R*u.degree, dec=D*u.degree, frame='icrs')
coords_gal = coords_eq.galactic

l_sn = coords_gal.l.radian     # Galactic longitude
b_sn = coords_gal.b.radian     # Galactic latitude

mb = np.array(data['m_b_corr'])
mb_err = np.array(data['m_b_corr_err_DIAG'])

mu_ceph = np.array(data['CEPH_DIST'])
calibrator = np.array(data['IS_CALIBRATOR'])

#cond_ceph = (zhd >= z_min_ceph) & (zhd <= z_max_ceph) & (calibrator == 1)
cond = (zhd >= z_min) & (zhd <= z_max)
#cond = cond_ceph | cond_flow

zhd = zhd[cond]
l_sn = l_sn[cond]
b_sn = b_sn[cond]
#l_sn = np.radians(R[cond])
#b_sn = np.radians(D[cond])

mb = mb[cond]
mb_err = mb_err[cond]

mu_ceph = mu_ceph[cond]
calibrator = calibrator[cond]


vx = np.cos(b_sn)*np.cos(l_sn)
vy = np.cos(b_sn)*np.sin(l_sn)
vz = np.sin(b_sn)

data = np.column_stack([vx, vy, vz, zhd, mb, mb_err, mu_ceph, calibrator])

header = "vx vy vz z mb mb_err mu_ceph calibrator"

cov = np.reshape(cov_data, (1701, 1701))

indices = cond.nonzero()[0]
cov = cov[np.ix_(indices, indices)]

np.savetxt("./data/pantheon_A.txt", data, header=header, comments='')
np.savetxt("./data/pantheon_cov_A.txt", cov)

print("PantheonPlus data is saved !")
print(len(zhd), np.shape(cov))
################################################################################################

