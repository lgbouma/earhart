import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_randich_lithium(PLOTDIR)
ep.plot_randich_lithium(PLOTDIR, corehalosplit=1)

assert 0

ep.plot_galah_dr3_lithium(PLOTDIR)
ep.plot_galah_dr3_lithium(PLOTDIR, corehalosplit=1)

assert 0
ep.plot_lithium(PLOTDIR)
