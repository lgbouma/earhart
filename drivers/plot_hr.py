import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'hr')

ep.plot_hr(PLOTDIR, include_extinction=0, color0='phot_bp_mean_mag')
ep.plot_hr(PLOTDIR, include_extinction=0, color0='phot_g_mean_mag')
ep.plot_hr(PLOTDIR, include_extinction=1)
