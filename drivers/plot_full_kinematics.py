import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'cluster_membership')

for show1937 in [0,1]:
    for galacticframe in [0,1]:
        ep.plot_full_kinematics(PLOTDIR, basedata='fullfaint', show1937=show1937, galacticframe=galacticframe) # deeper
        # ep.plot_full_kinematics(PLOTDIR, basedata='bright', show1937=show1937, galacticframe=galacticframe) # standard
        # ep.plot_full_kinematics(PLOTDIR, basedata='fullfaint_edr3', show1937=show1937, galacticframe=galacticframe) # deeper + EDR3
