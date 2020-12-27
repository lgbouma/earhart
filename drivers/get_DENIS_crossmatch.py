"""
Crossmatch Gaia (EDR3) J2016's to the DENIS J2000's. (Perhaps using the vizier
propagations, which could be easier, since they account for proper motions).
NB. TOI 1937B is not resolved in the DENIS source catalog.

Going with the following approach:

Start with Gaia EDR3 J2016 positions. Using the proper motions, propagate them
to J2000, ignoring the radial velocities. For the CG18 core cluster members,
this typically results in a ~0.2 arcsecond shift.

    (I thought about and decided against other approaches, including the wrong
    approach of just using the Gaia J2016's directly, and crossmatching against
    Vizier since they compute nice J2000 coordinates).

Then, search many DENIS cones. For each cone, the closest match in position is
"the match", though the angDist and magnitude difference are both available.

Outcome is:
    Started with 13207 neighbors
    Started with 1106 in core
    Started with 1912 in corona
    DENIS xmatch yielded 13207 neighbors, 12170 w/ Imags
    DENIS xmatch yielded 1106 in core, 938 w/ Imags
    DENIS xmatch yielded 1912 in corona, 1795 w/ Imags

which are all saved in
    targetpath = '../data/denis/target_gaia_denis_xm.csv'
    cg18path = '../data/denis/cg18_gaia_denis_xm.csv'
    kc19path = '../data/denis/kc19_gaia_denis_xm.csv'
    nbhdpath = '../data/denis/nbhd_gaia_denis_xm.csv'
"""

import numpy as np, pandas as pd
from earhart.helpers import (
    _get_fullfaint_edr3_dataframes, get_denis_xmatch
)
from cdips.utils.coordutils import precess_gaia_coordinates
from astropy import units as u
from astropy.table import Table
from numpy import array as nparr

basedata = 'fullfaint_edr3'

if basedata == 'fullfaint_edr3':
    nbhd_df, cg18_df, kc19_df, target_df = _get_fullfaint_edr3_dataframes()
else:
    raise NotImplementedError

c_cg18, c_cg18_j2000 = precess_gaia_coordinates(cg18_df)
c_kc19, c_kc19_j2000 = precess_gaia_coordinates(kc19_df)
c_target, c_target_j2000 = precess_gaia_coordinates(target_df)

# require parallax S/N>5 to avoid negative parallaxes
sel_nbhd = (nbhd_df.parallax / nbhd_df.parallax_error) > 5
nbhd_df = nbhd_df[sel_nbhd]
c_nbhd, c_nbhd_j2000 = precess_gaia_coordinates(nbhd_df)

# now search many DENIS cones through astroquery's vizier.
get_mag = lambda df: nparr(df.phot_g_mean_mag)
get_ids = lambda df: nparr(df.source_id)

get_merge = lambda df0, df_dxm: df0.merge(
    df_dxm, left_on='source_id', right_on='_id', how='left'
)

target_dxm = get_denis_xmatch(c_target_j2000, _id=get_ids(target_df),
                              mag=get_mag(target_df))
out_target = get_merge(target_df, target_dxm)

cg18_dxm = get_denis_xmatch(c_cg18_j2000, _id=get_ids(cg18_df),
                            mag=get_mag(cg18_df))
out_cg18 = get_merge(cg18_df, cg18_dxm)

kc19_dxm = get_denis_xmatch(c_kc19_j2000, _id=get_ids(kc19_df),
                            mag=get_mag(kc19_df))
out_kc19 = get_merge(kc19_df, kc19_dxm)

nbhd_dxm = get_denis_xmatch(c_nbhd_j2000, _id=get_ids(nbhd_df),
                            mag=get_mag(nbhd_df))
out_nbhd = get_merge(nbhd_df, nbhd_dxm)

print(f'Started with {len(nbhd_df)} neighbors')
print(f'Started with {len(cg18_df)} in core')
print(f'Started with {len(kc19_df)} in corona')
print(f'DENIS xmatch yielded {len(out_nbhd)} neighbors, {len(out_nbhd[~pd.isnull(out_nbhd.Imag)])} w/ Imags')
print(f'DENIS xmatch yielded {len(out_cg18)} in core, {len(out_cg18[~pd.isnull(out_cg18.Imag)])} w/ Imags')
print(f'DENIS xmatch yielded {len(out_kc19)} in corona, {len(out_kc19[~pd.isnull(out_kc19.Imag)])} w/ Imags')

targetpath = '../data/denis/target_gaia_denis_xm.csv'
cg18path = '../data/denis/cg18_gaia_denis_xm.csv'
kc19path = '../data/denis/kc19_gaia_denis_xm.csv'
nbhdpath = '../data/denis/nbhd_gaia_denis_xm.csv'

out_target.to_csv(targetpath, index=False)
out_cg18.to_csv(cg18path, index=False)
out_kc19.to_csv(kc19path, index=False)
out_nbhd.to_csv(nbhdpath, index=False)
