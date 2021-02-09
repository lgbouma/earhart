import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_X_RUWE')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for yscale in ['linear', 'log']:
    ep.plot_rotation_X_RUWE(PLOTDIR, 'viridis', emph_1937=1, yscale=yscale)
    ep.plot_rotation_X_RUWE(PLOTDIR, 'nipy_spectral', yscale=yscale)
    ep.plot_rotation_X_RUWE(PLOTDIR, 'viridis', yscale=yscale)
