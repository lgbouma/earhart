import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lumfunction_vs_position')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_lumfunction_vs_position(PLOTDIR)
