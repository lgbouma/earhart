import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

ep.plot_full_kinematics(PLOTDIR)
