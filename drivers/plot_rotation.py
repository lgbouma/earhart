import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation')

ep.plot_rotation(PLOTDIR, BpmRp=0)
ep.plot_rotation(PLOTDIR, BpmRp=1)
