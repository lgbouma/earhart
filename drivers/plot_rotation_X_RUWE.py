import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_X_RUWE')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_rotation_X_RUWE(PLOTDIR, 'viridis', emph_1937=1)
ep.plot_rotation_X_RUWE(PLOTDIR, 'nipy_spectral')
ep.plot_rotation_X_RUWE(PLOTDIR, 'viridis')
