"""
How far apart, spatially, are the maximal points in NGC 2516?
"""

import os
import numpy as np, pandas as pd

from earhart.helpers import (
    get_gaia_basedata, get_autorotation_dataframe,
    _get_median_ngc2516_core_params
)
from earhart.physicalpositions import (
    calc_dist, append_physicalpositions
)

def get_distances(outpath, df):

    selcols = ['x_pc','y_pc','z_pc']

    dist_list = []

    if not os.path.exists(outpath):
        cnt = 0
        for i, r in df.iterrows():

            print(f'{cnt}/{len(df)}...')

            x0,y0,z0 = r[selcols]
            s0 = r['source_id']

            for j, _r in df.iterrows():

                x1,y1,z1 = _r[selcols]
                s1 = _r['source_id']

                dist_list.append(
                    (s0, s1, calc_dist(x0, y0, z0, x1, y1, z1))
                )

            cnt += 1

        dist_arr = np.array(dist_list)
        dist_df = pd.DataFrame(
            {'s0':dist_arr[:,0], 's1':dist_arr[:,1], 'dist_pc':dist_arr[:,2]}
        )

        dist_df.sort_values(by='dist_pc').to_csv(outpath, index=False)

    dist_df = pd.read_csv(outpath)

    return dist_df



basedata = 'fullfaint'
nbhd_df, core_df, halo_df, full_df, trgt_df = get_gaia_basedata(basedata)
rot_df, lc_df = get_autorotation_dataframe(
    runid='NGC_2516', returnbase=True, cleaning='defaultcleaning'
)
med_df, _ = _get_median_ngc2516_core_params(core_df, basedata)

full_df = append_physicalpositions(full_df, med_df)

from earhart.priors import TEFF, P_ROT, AVG_EBpmRp
get_BpmRp0 = lambda df: (df['phot_bp_mean_mag'] - df['phot_rp_mean_mag'] - AVG_EBpmRp)
sel_color = lambda df: (get_BpmRp0(df) > 0.5) & (get_BpmRp0(df) < 1.2)
sel_autorot = lambda df: df.source_id.isin(rot_df.source_id)
sel_haslc = lambda df: df.source_id.isin(lc_df.source_id)

sel_comp = lambda df: (sel_color(df)) & (sel_haslc(df))
sel_rotn =  lambda df: (sel_color(df)) & (sel_autorot(df))

# x_pc, y_pc, z_pc
fdf = full_df[sel_rotn(full_df)]
cdf = full_df[sel_comp(full_df)]

full_df['source_id'] = full_df.source_id.astype(str)
fdf['source_id'] = fdf.source_id.astype(str)
cdf['source_id'] = cdf.source_id.astype(str)

outpath = '../results/maximal_cluster_distance/full_cache.csv'
full_dist_df = get_distances(outpath, full_df)

outpath = '../results/maximal_cluster_distance/rot_cache.csv'
rot_dist_df = get_distances(outpath, fdf)

outpath = '../results/maximal_cluster_distance/comp_cache.csv'
comp_dist_df = get_distances(outpath, cdf)

rot_dist_df['dist_pc'] = rot_dist_df.dist_pc.astype(float)
rot_dist_df = rot_dist_df.sort_values(by='dist_pc')

comp_dist_df['dist_pc'] = comp_dist_df.dist_pc.astype(float)
comp_dist_df = comp_dist_df.sort_values(by='dist_pc')

full_dist_df['dist_pc'] = full_dist_df.dist_pc.astype(float)
full_dist_df = full_dist_df.sort_values(by='dist_pc')


print(rot_dist_df.drop_duplicates('dist_pc'))

print(full_dist_df.drop_duplicates('dist_pc'))
