import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

# basedata = 'bright' # NOTE: deprecated
basedata = 'fullfaint'
# basedata = 'extinctioncorrected' # NOTE TODO

PLOTDIR = os.path.join(RESULTSDIR, 'hr')

for highlight_companion in [1,0]:
    ep.plot_hr(PLOTDIR, color0='phot_bp_mean_mag',
               highlight_companion=highlight_companion, basedata=basedata)
    ep.plot_hr(PLOTDIR, color0='phot_g_mean_mag',
               highlight_companion=highlight_companion, basedata=basedata)
