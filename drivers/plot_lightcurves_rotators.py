import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'lightcurves_rotators')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

N_show, seed = 20, 123
ep.plot_lightcurves_rotators(PLOTDIR, color=1, N_show=N_show, seed=seed,
                             talk_aspect=1)

N_show, seed = 50, 123
ep.plot_lightcurves_rotators(PLOTDIR, color=1, N_show=N_show, seed=seed)
ep.plot_lightcurves_rotators(PLOTDIR, N_show=N_show, seed=seed)
