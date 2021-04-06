import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'rotation_X_positions')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

for approach in [2,1]:
    for cleaning in ['defaultcleaning','periodogram_match']:
        if approach == 1:
            ep.plot_rotation_X_positions(PLOTDIR, cmapname='magma',
                                         approach=approach, cleaning=cleaning)
            ep.plot_rotation_X_positions(PLOTDIR, cmapname='cividis',
                                         approach=approach, cleaning=cleaning)
            ep.plot_rotation_X_positions(PLOTDIR, cmapname='viridis',
                                         approach=approach, cleaning=cleaning)
            ep.plot_rotation_X_positions(PLOTDIR, cmapname='plasma',
                                         approach=approach, cleaning=cleaning)
        else:
            ep.plot_rotation_X_positions(PLOTDIR, cmapname=None, approach=approach,
                                         cleaning=cleaning)
