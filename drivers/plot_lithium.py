import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lithium')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_lithium_EW_vs_color(
    PLOTDIR, gaiaeso=1, galahdr3=1, corehalosplit=1
)
ep.plot_lithium_EW_vs_color(
    PLOTDIR, gaiaeso=1, galahdr3=1, corehalosplit=1, showkepfield=1
)
ep.plot_lithium_EW_vs_color(
    PLOTDIR, gaiaeso=1, galahdr3=0
)
ep.plot_lithium_EW_vs_color(
    PLOTDIR, gaiaeso=0, galahdr3=1
)
ep.plot_lithium_EW_vs_color(
    PLOTDIR, gaiaeso=1, galahdr3=1, corehalosplit=0
)

ep.plot_galah_dr3_lithium_abundance(PLOTDIR)
ep.plot_galah_dr3_lithium_abundance(PLOTDIR, corehalosplit=1)

# DEPRECATED
# ep.plot_randich_lithium(PLOTDIR)
# ep.plot_randich_lithium(PLOTDIR, corehalosplit=1)
