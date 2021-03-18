"""
Once cluster_rotation.get_auto_rotation_periods has been run, this script
populates results.tables with the table containing:

    * Rotation period + neighboring star count information.
    * Subcluster information.
    * "Cleaned subset" information ("subset A" or "subset B" of cleaning in the
    paper's terminology).
    * Lithium information.
"""
import os
import numpy as np, pandas as pd
from earhart.paths import DATADIR, RESULTSDIR
from earhart.helpers import get_autorotation_dataframe
from earhart.lithium import _get_lithium_EW_df

include_lithium = False
N_members = 3298

# made by cluster_rotation.get_auto_rotation_periods
# already contains most information except for "clean subset" information.
df = pd.read_csv(
    os.path.join(DATADIR, 'rotation', 'NGC_2516_rotation_periods.csv')
)
assert len(df) == N_members

from earhart.priors import AVG_EBpmRp
df['(Bp-Rp)_0'] = df.phot_bp_mean_mag - df.phot_rp_mean_mag - AVG_EBpmRp

cleanings = ['defaultcleaning', 'periodogram_match', 'match234_alias']

# add the "in_{CLEANING}" columns
for c in cleanings:
    df_A = get_autorotation_dataframe('NGC_2516', cleaning=c)
    df[f'in_{c}'] = df.source_id.isin(df_A.source_id)

assert len(df) == N_members

dropcols = ['level_0', 'source_id_2', 'index', 'datalink_url',
            'epoch_photometry_url',  'priam_flags', 'teff_val',
            'teff_percentile_lower', 'teff_percentile_upper', 'a_g_val',
            'a_g_percentile_lower', 'a_g_percentile_upper', 'e_bp_min_rp_val',
            'e_bp_min_rp_percentile_lower', 'e_bp_min_rp_percentile_upper',
            'flame_flags', 'radius_val', 'radius_percentile_lower',
            'radius_percentile_upper', 'lum_val', 'lum_percentile_lower',
            'lum_percentile_upper']

df = df.drop(dropcols, axis=1)


# by default, release all columns. it's machine-readable...
outpath = os.path.join(RESULTSDIR, 'tables', 'NGC_2516_Prot_cleaned.csv')

df.to_csv(outpath, index=False)
print(f'Wrote {outpath}')

if include_lithium:
    #
    # TODO TODO: THE RANDICH+18 GAIA-ESO MERGE HAS NON-UNIQUE SOURCE_IDS. 
    # A FIXME IS TO DEBUG THIS
    #

    # get lithium data, first gaia-eso / R+18. rename columns to ensure that
    # sources with both GaiaESO and GALAH spectra get their EWs reported.
    ldf0 = _get_lithium_EW_df(1, 0)
    newcoldict = {}
    for c in ldf0.columns:
        if "Li" in c:
            newcoldict[c] =c+"_GaiaESO"
    ldf = ldf0.rename(newcoldict, axis='columns')
    mdf = df.merge(ldf0, how='left', on='source_id')

    ldf1 = _get_lithium_EW_df(0, 1)
    newcoldict = {}
    for c in ldf1.columns:
        if "Li" in c:
            newcoldict[c] =c+"_GALAH"
    outdf = mdf.merge(ldf1, how='left', on='source_id')

    assert len(outdf) == N_members

    # by default, release all columns. it's machine-readable...
    outpath = os.path.join(RESULTSDIR, 'tables', 'NGC_2516_Prot_cleaned.csv')

    outdf.to_csv(outpath, index=False)
    print(f'Wrote {outpath}')
