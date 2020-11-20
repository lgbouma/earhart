import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation')

ep.plot_rotation(PLOTDIR, BpmRp=0)
ep.plot_rotation(PLOTDIR, BpmRp=1)

ep.plot_rotation(PLOTDIR, BpmRp=1, include_ngc2516=1, ngc_core_halo=0)
ep.plot_rotation(PLOTDIR, BpmRp=1, include_ngc2516=1, ngc_core_halo=1)
