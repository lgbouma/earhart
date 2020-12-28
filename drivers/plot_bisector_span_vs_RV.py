import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'bisector_span_vs_RV')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_bisector_span_vs_RV(PLOTDIR, which='Gummi')
ep.plot_bisector_span_vs_RV(PLOTDIR, which='Hartman')
