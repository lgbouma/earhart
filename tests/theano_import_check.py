"""
The following "simple" check of whether your exoplanet + theano installation
works should pass. (2020/12/02, it does not, with a theano compilation error)

This is b/c of some kind of catalina problem, I think.

----------
Tue Dec 22 13:23:28 2020
Update:

It's a gcc compiler flag issue. Along with the "cxxflags" line below, it can be
fixed (or shunted aside) by

` printf '[gcc]\ncxxflags=-Wno-c++11-narrowing' > ~/.theanorc `

(assuming you haven't already put together a ~/.theanorc file)
"""
import numpy as np
import matplotlib.pyplot as plt

import theano
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"
import exoplanet as xo

# The light curve calculation requires an orbit
orbit = xo.orbits.KeplerianOrbit(period=3.456)

# Compute a limb-darkened light curve using starry
# Note: the `eval` is needed because this is using Theano in
# the background
t = np.linspace(-0.1, 0.1, 1000)
u = [0.3, 0.2]
light_curve = (
    xo.LimbDarkLightCurve(u)
    .get_light_curve(orbit=orbit, r=0.1, t=t, texp=0.02)
    .eval()
)

plt.plot(t, light_curve, color="C0", lw=2)
plt.ylabel("relative flux")
plt.xlabel("time [days]")
_ = plt.xlim(t.min(), t.max())

outpath = '../results/test_results/theano_import_check.png'
plt.savefig(outpath)
print(f'saved {outpath}')
