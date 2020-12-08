from astropy import units as u
from math import sqrt

vsini = 8*u.km/u.s
b = 0.84
delta = 1.4e-2

u1 = 0.15
u2 = 0.1
mu = (1-b**2)**(1/2)

f_LD = 1 - u1 * (1 - mu) - u2*(1 - mu)**2

delta_V_RM = f_LD * delta * vsini * sqrt(1 - b**2)

print(delta_V_RM.to(u.m/u.s))
