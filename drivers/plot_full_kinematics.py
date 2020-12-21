import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

ep.plot_full_kinematics(PLOTDIR, basedata='bright') # standard
ep.plot_full_kinematics(PLOTDIR, basedata='fullfaint') # deeper
ep.plot_full_kinematics(PLOTDIR, basedata='fullfaint_edr3') # deeper + EDR3
