"""
Plot venn diagram of overlapping cluster memberships
"""

import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'venn')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_venn(PLOTDIR)
