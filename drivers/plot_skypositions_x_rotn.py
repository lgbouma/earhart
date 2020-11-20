import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'skypositions')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_skypositions_x_rotn(PLOTDIR)
