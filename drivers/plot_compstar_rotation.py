import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'compstar_rotation')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

#for yscale in ['linear', 'log']:
for yscale in ['linear']:
    for cleaning in [
        'defaultcleaning','defaultcleaning_cutProtColor'
    ]:
        ep.plot_compstar_rotation(
            PLOTDIR, yscale=yscale, corehalosplit=1,
            cleaning=cleaning
        )
        ep.plot_compstar_rotation(
            PLOTDIR, yscale=yscale, showPleiadesQuad=1,
            cleaning=cleaning
        )
