import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'ngc2516_corehalo_3panel')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

basedatas = ['bright']#, 'fullfaint']
for basedata in basedatas:
    ep.plot_ngc2516_corehalo_3panel(PLOTDIR, basedata=basedata)
    ep.plot_ngc2516_corehalo_3panel(PLOTDIR, basedata=basedata, emph_1937=1)
