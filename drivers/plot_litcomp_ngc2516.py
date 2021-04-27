import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'litcomp_ngc2516')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_litcomp_ngc2516(PLOTDIR)

