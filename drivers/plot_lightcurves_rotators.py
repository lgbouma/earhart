import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lightcurves_rotators')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_lightcurves_rotators(PLOTDIR, color=1)
ep.plot_lightcurves_rotators(PLOTDIR)
