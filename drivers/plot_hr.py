import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

basedatas = ['fullfaint_edr3'] #, 'bright', 'fullfaint', 'denis_fullfaint_edr3']
isochrones = ['parsec', None, 'mist'] # could be None, mist, parsec.

PLOTDIR = os.path.join(RESULTSDIR, 'hr')

for talk_aspect in [1,0]:
    for basedata in basedatas:
        for isochrone in isochrones:

            if 'denis' not in basedata:

                for c in ['phot_bp_mean_mag', 'phot_g_mean_mag']:

                    # base HR diagrams
                    for show1937 in [0,1]:
                        for highlight_companion in [0,1]:
                            if highlight_companion and not show1937:
                                continue
                            ep.plot_hr(PLOTDIR, isochrone=isochrone, color0=c,
                                       highlight_companion=highlight_companion,
                                       basedata=basedata, show1937=show1937,
                                       talk_aspect=talk_aspect)

                    if isochrone is None:
                        # check for differential extinction
                        ep.plot_hr(PLOTDIR, color0=c,
                                   highlight_companion=0, basedata=basedata,
                                   colorhalobyglat=1, talk_aspect=talk_aspect)

            else:

                # base HR diagrams
                for highlight_companion in [1,0]:
                    ep.plot_hr(PLOTDIR, color0=None,
                               highlight_companion=highlight_companion,
                               basedata=basedata, talk_aspect=talk_aspect)
