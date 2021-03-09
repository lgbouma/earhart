import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'vtangential_projection')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_vtangential_projection(PLOTDIR)
