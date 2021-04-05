"""
Quick analysis of the photometric binarity
"""
import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'slowfast_photbinarity_comparison')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_slowfast_photbinarity_comparison(
    PLOTDIR
)

