from astropy.coordinates import SkyCoord
import astropy.units as u

# ---- Edit these four numbers ----
l1, b1 = 151.88, 9.59    # point 1: (l, b) in degrees
l2, b2 = 157.50, -4.78    # point 2: (l, b) in degrees
# ----------------------------------

p1 = SkyCoord(l=l1*u.degree, b=b1*u.degree, frame='galactic')
p2 = SkyCoord(l=l2*u.degree, b=b2*u.degree, frame='galactic')

sep = p1.separation(p2)

print(f"Point 1: l = {l1} deg, b = {b1} deg")
print(f"Point 2: l = {l2} deg, b = {b2} deg")
print(f"Angular separation: {sep.degree:.2f} degrees")
