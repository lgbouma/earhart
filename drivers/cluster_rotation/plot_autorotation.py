import os
import earhart.plotting as ep
from earhart.paths import RESULTSDIR

RUNID_EXTINCTION_DICT = {
#    'compstar_NGC_2516': 0.1343, # 0.25 for KC19
    'NGC_2516': 0.1343, # 0.25 for KC19
#     'IC_2602': 0.0799,  # avg E(B-V) from Randich+18, *1.31 per Stassun+2019
#     'CrA': 0.06389, # KC19 ratio used
#     'kc19group_113': 0.1386, # 0.258 from KC19 -- take ratio
#     'Orion': 0.1074, # again KC19 ratio used
#     'VelaOB2': 0.2686, # KC19 ratio
#     'ScoOB2': 0.161, # KC19 ratio
}


for runid, _ in RUNID_EXTINCTION_DICT.items():

    E_BpmRp = RUNID_EXTINCTION_DICT[runid]

    PLOTDIR = os.path.join(RESULTSDIR, 'rotation', runid)
    if not os.path.exists(PLOTDIR):
        os.mkdir(PLOTDIR)

    for yscale in ['linear', 'log']:
        for cleaning in ['nocleaning', 'periodogram_match', 'match234_alias']:
            ep.plot_auto_rotation(
                PLOTDIR, runid, E_BpmRp, core_halo=0, yscale=yscale, cleaning=cleaning
            )
            ep.plot_auto_rotation(
                PLOTDIR, runid, E_BpmRp, core_halo=1, yscale=yscale, cleaning=cleaning
            )
