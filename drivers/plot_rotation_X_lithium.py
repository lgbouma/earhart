import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_X_lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_rotation_X_lithium(PLOTDIR, 'nipy_spectral')
ep.plot_rotation_X_lithium(PLOTDIR, 'viridis')
