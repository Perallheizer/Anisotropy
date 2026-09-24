import numpy as np
from scipy.integrate import quad
from scipy.constants import c

c = c / 1000.0   # convert to km/s

class Model(object):
    def __init__(self, H0, q0, j0=1):
        self.H0 = float(H0)
        self.q0 = float(q0)
        self.j0 = float(j0)

    def Ez(self, z):
        return (2*(z+1)**2*(self.j0*z - 3*self.q0**2*z - self.q0*(z+3)+z+3)**2)/(self.p0(z)+self.p1(z)*z+self.p2(z)*z**2)
        
    def p0(self, z):
        return 18*(-1+self.q0)**2
        
    def p1(self, z):
        return 6*(-1+self.q0)*(-5-2*self.j0+8*self.q0+3*self.q0**2)
        
    def p2(self, z):
        return 14 + 7*self.j0 + 2*self.j0**2 - 10*(4+self.j0)*self.q0 + (17-9*self.j0)*self.q0**2 + 18*self.q0**3 + 9*self.q0**4
        
        
    def Hz(self, z):
        return self.H0 * self.Ez(z)

    def inv_Ez(self, z):
        return 1.0 / self.Ez(z)

    def dMz(self, z1):
        return c/self.H0 * quad(self.inv_Ez, 0, z1)[0]
        
    def dLz(self, z):
        return (c/self.H0) * (z * (6*(self.q0 - 1) + (-5 - 2*self.j0 + self.q0*(8 + 3*self.q0)) * z)/(-2*(3 + z + self.j0*z)+ 2*self.q0*(3 + z + 3*self.q0*z)))


