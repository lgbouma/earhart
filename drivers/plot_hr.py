import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

# basedata = 'bright' # NOTE: deprecated
# basedata = 'fullfaint' # NOTE: good; DR2 baseline
basedata = 'fullfaint_edr3'
# basedata = 'extinctioncorrected' # NOTE TODO

PLOTDIR = os.path.join(RESULTSDIR, 'hr')

for c in ['phot_bp_mean_mag', 'phot_g_mean_mag']:

    # check for differential extinction
    ep.plot_hr(PLOTDIR, color0=c,
               highlight_companion=0, basedata=basedata, colorhalobyglat=1)

    # base HR diagrams
    for highlight_companion in [1,0]:
        ep.plot_hr(PLOTDIR, color0=c,
                   highlight_companion=highlight_companion, basedata=basedata)


