"""
Same data as "full_kinematics_X_rotation", but in XYZ coordinates, and with
physical (on-sky) velocity differences from the cluster mean.

Also makes histogram_physical_X_rotation_fullfaint
"""

import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'physical_X_rotation')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for show1937 in [0,1]:
    ep.plot_physical_X_rotation(PLOTDIR, basedata='fullfaint', show1937=show1937) # deeper
