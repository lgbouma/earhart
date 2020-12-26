"""
Integrate an orbit and compute uncertainties in Milky Way potential model

http://gala.adrian.pw/en/latest/examples/Milky-Way-model.html
"""

# Third-party
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np

# Gala
from gala.mpl_style import mpl_style
plt.style.use(mpl_style)
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import galactic

# Set the default Astropy Galactocentric frame parameters to the values adopted
# in Astropy v4.0:
coord.galactocentric_frame_defaults.set('v4.0')

# For the Milky Way model, we’ll use the built-in potential class in gala (see
# above for definition). Description is here:
# http://gala.adrian.pw/en/latest/potential/define-milky-way-model.html
potential = gp.MilkyWayPotential()

# For this example, we're going to look at the position and distance of the
# Draco satellite galaxy.
icrs = coord.SkyCoord(ra=coord.Angle('17h 20m 12.4s'),
                      dec=coord.Angle('+57° 54′ 55″'),
                      distance=76*u.kpc,
                      pm_ra_cosdec=0.0569*u.mas/u.yr,
                      pm_dec=-0.1673*u.mas/u.yr,
                      radial_velocity=-291*u.km/u.s)

icrs_err = coord.SkyCoord(ra=0*u.deg, dec=0*u.deg, distance=6*u.kpc,
                          pm_ra_cosdec=0.009*u.mas/u.yr,
                          pm_dec=0.009*u.mas/u.yr,
                          radial_velocity=0.1*u.km/u.s)

# Start by transforming the measured values to a Galactocentric reference frame
# so we can integrate an orbit in our Milky Way model
gc_frame = coord.Galactocentric()

# Transform the mean observed kinematics to this frame:
galcen = icrs.transform_to(gc_frame)

# Turn the `Galactocentric` object into orbital initial conditions, and
# integrate. Timestep: 0.5 Myr, and integrate back for 1e4 steps (5 Gyr).
w0 = gd.PhaseSpacePosition(galcen.data)
orbit = potential.integrate_orbit(w0, dt=-0.5*u.Myr, n_steps=10000)

fig = orbit.plot()
fig.savefig('../results/test_results/sandbox_gala_integration.png')

print(orbit.pericenter(), orbit.apocenter(), orbit.eccentricity())

# Now we’ll sample from the error distribution over the distance, proper
# motions, and radial velocity, compute orbits, and plot distributions of mean
# pericenter and apocenter:
n_samples = 32
dist = np.random.normal(icrs.distance.value, icrs_err.distance.value,
                        n_samples) * icrs.distance.unit

pm_ra_cosdec = np.random.normal(icrs.pm_ra_cosdec.value,
                                icrs_err.pm_ra_cosdec.value,
                                n_samples) * icrs.pm_ra_cosdec.unit

pm_dec = np.random.normal(icrs.pm_dec.value,
                          icrs_err.pm_dec.value,
                          n_samples) * icrs.pm_dec.unit

rv = np.random.normal(icrs.radial_velocity.value, icrs_err.radial_velocity.value,
                      n_samples) * icrs.radial_velocity.unit

ra = np.full(n_samples, icrs.ra.degree) * u.degree
dec = np.full(n_samples, icrs.dec.degree) * u.degree

icrs_samples = coord.SkyCoord(ra=ra, dec=dec, distance=dist,
                              pm_ra_cosdec=pm_ra_cosdec,
                              pm_dec=pm_dec, radial_velocity=rv)

galcen_samples = icrs_samples.transform_to(gc_frame)

w0_samples = gd.PhaseSpacePosition(galcen_samples.data)
orbit_samples = potential.integrate_orbit(w0_samples, dt=-1*u.Myr, n_steps=4000)

pers = orbit_samples.pericenter(approximate=True)
apos = orbit_samples.apocenter(approximate=True)
eccs = orbit_samples.eccentricity(approximate=True)

plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].hist(pers.to_value(u.kpc), bins='auto')
axes[0].set_xlabel('pericenter [kpc]')

axes[1].hist(apos.to_value(u.kpc), bins='auto')
axes[1].set_xlabel('apocenter [kpc]')

axes[2].hist(eccs.value, bins='auto')
axes[2].set_xlabel('eccentricity')
fig.savefig('../results/test_results/sandbox_gala_sampling.png')
