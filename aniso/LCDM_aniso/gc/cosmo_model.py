import numpy as np
from scipy.integrate import quad
from scipy.constants import c

c = c / 1000.0   # convert to km/s

class Model(object):
    def __init__(self, H0, omega_m, omega_r = 9e-5):
        self.H0 = float(H0)
        self.omega_m = float(omega_m)
        self.omega_r = float(omega_r)

    def Ez(self, z):
        return np.sqrt(self.omega_m*(1+z)**3 + self.omega_r*(1+z)**4 + (1 - self.omega_m - self.omega_r))
        
    def Hz(self, z):
        return self.H0 * self.Ez(z)

    def inv_Ez(self, z):
        return 1.0 / self.Ez(z)

    def dMz(self, z1):
        return c/self.H0 * quad(self.inv_Ez, 0, z1)[0]

    def dLz(self, z):
        return self.dMz(z) * (1 + z)




