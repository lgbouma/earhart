"""
Do the dynamics of TOI 1937A imply that in the past, it was closer to the core
of the cluster?  (More generally: is the top-down S-shape of the halo
consistent with it being sheared apart by galactic rotation?)

Most of this calculation is ported from tests/sandbox_gala, which is direct
from the tutorials Adrian wrote up in his documentation.
"""
import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

basedata = 'fullfaint_edr3'

ep.plot_backintegration_ngc2516(basedata)
ep.plot_backintegration_ngc2516(basedata, fix_rvs=1)
