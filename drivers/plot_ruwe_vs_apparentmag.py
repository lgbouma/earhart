import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

basedata = 'fullfaint_edr3'
# basedata = 'extinctioncorrected' # NOTE TODO

PLOTDIR = os.path.join(RESULTSDIR, 'gaia_ruwe_vs_apparentmag')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_ruwe_vs_apparentmag(PLOTDIR, basedata=basedata, smallylim=0)
ep.plot_ruwe_vs_apparentmag(PLOTDIR, basedata=basedata, smallylim=1)
