import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

# basedata = 'bright' # NOTE: deprecated
basedata = 'fullfaint'
# basedata = 'extinctioncorrected' # NOTE TODO

PLOTDIR = os.path.join(RESULTSDIR, 'gaia_rv_scatter_vs_brightness')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_gaia_rv_scatter_vs_brightness(PLOTDIR, basedata=basedata)
