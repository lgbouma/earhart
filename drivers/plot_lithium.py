import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for vs_rotators in [0,1]:

    ep.plot_galah_dr3_lithium(PLOTDIR, vs_rotators=vs_rotators)
    ep.plot_galah_dr3_lithium(PLOTDIR, corehalosplit=1, vs_rotators=vs_rotators)

    ep.plot_randich_lithium(PLOTDIR, vs_rotators=vs_rotators)
    ep.plot_randich_lithium(PLOTDIR, corehalosplit=1, vs_rotators=vs_rotators)


assert 0
# TODO: write something that merges both
ep.plot_lithium(PLOTDIR)
