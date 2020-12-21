import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

PLOTDIR = os.path.join(RESULTSDIR, 'edr3_blending_vs_apparentmag')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

ep.plot_edr3_blending_vs_apparentmag(PLOTDIR, basedata='fullfaint_edr3',
                                     num='phot_bp_n_blended_transits')
ep.plot_edr3_blending_vs_apparentmag(PLOTDIR, basedata='fullfaint_edr3',
                                     num='phot_rp_n_blended_transits')

ep.plot_edr3_blending_vs_apparentmag(PLOTDIR, basedata='fullfaint_edr3',
                                     num='phot_bp_n_contaminated_transits')
ep.plot_edr3_blending_vs_apparentmag(PLOTDIR, basedata='fullfaint_edr3',
                                     num='phot_rp_n_contaminated_transits')
