import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

basedatas = ['fullfaint_edr3'] #, 'bright', 'fullfaint', 'denis_fullfaint_edr3']

PLOTDIR = os.path.join(RESULTSDIR, 'phot_binaries')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for basedata in basedatas:

    for c in ['phot_bp_mean_mag']:

        ep.plot_phot_binaries(PLOTDIR, color0=c, basedata=basedata)
