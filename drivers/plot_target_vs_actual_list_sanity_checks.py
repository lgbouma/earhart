"""
The 3018 candidate cluster members are split into the "core" and
"halo" samples, below. Source ID's are from Gaia DR2.

For rotation periods, 2238 of the stars had light curves produced.

Where are the other 780?
"""

import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from earhart.paths import RESULTSDIR, DATADIR
from aesthetic.plot import savefig, format_ax, set_style

PLOTDIR = os.path.join(
    RESULTSDIR, 'target_vs_actual_list_sanity_checks'
)
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

halo_df = pd.read_csv(
    os.path.join(RESULTSDIR, 'tables', 'NGC_2516_halo_fullfaint.csv')
)
core_df = pd.read_csv(
    os.path.join(RESULTSDIR, 'tables', 'NGC_2516_core_fullfaint.csv')
)
ngc_df = pd.concat((core_df, halo_df))
print(f'NGC candidate member total: {len(ngc_df)}')

rot_df = pd.read_csv(
    os.path.join(DATADIR, 'rotation', 'NGC_2516_rotation_periods.csv')
)
lccut_df = pd.read_csv(
    '/Users/luke/Dropbox/proj/cdips/data/cluster_data/cdips_catalog_split/OC_MG_FINAL_v0.4_publishable_CUT_NGC_2516.csv'
)
print(f'... with CDIPS light curves: {len(lccut_df)}')

#
# first plot: ra vs dec.
#
plt.close('all')
f,ax = plt.subplots(figsize=(4,3))
ax.scatter(
    ngc_df.ra, ngc_df.dec, c='C0', alpha=0.8, zorder=2,
    s=5, rasterized=True, linewidths=0, label='NGC2516', marker='.'
)
ax.scatter(
    lccut_df.ra, lccut_df.dec, c='C1', alpha=0.8, zorder=3,
    s=5, rasterized=True, linewidths=0, label='w/ CDIPS LC', marker='.'
)
ax.legend(loc='best')
ax.set_xlabel('ra')
ax.set_ylabel('dec')
outpath = os.path.join(PLOTDIR, f'ra_vs_dec_check.png')
savefig(f, outpath, dpi=400)

#
# next plot: HR.
#
get_xval = (
    lambda _df: np.array(
        _df['phot_bp_mean_mag'] - _df['phot_rp_mean_mag']
    )
)
get_yval = (
    lambda _df: np.array(
        _df['phot_g_mean_mag'] + 5*np.log10(_df['parallax']/1e3) + 5
    )
)
plt.close('all')
f,ax = plt.subplots(figsize=(4,3))
ax.scatter(
    get_xval(ngc_df), get_yval(ngc_df), c='C0', alpha=0.8, zorder=2,
    s=5, rasterized=True, linewidths=0, label='NGC2516', marker='.'
)
ax.scatter(
    get_xval(lccut_df), get_yval(lccut_df), c='C1', alpha=0.8, zorder=2,
    s=5, rasterized=True, linewidths=0, label='w/ CDIPS LC', marker='.'
)
ax.legend(loc='best')
ax.set_xlabel('Bp-Rp')
ax.set_ylabel('Abs G')
ylim = ax.get_ylim()
ax.set_ylim((max(ylim),min(ylim)))
outpath = os.path.join(PLOTDIR, f'hr_check.png')
savefig(f, outpath, dpi=400)

print(42*'-')
print(f'NGC candidate member total: {len(ngc_df)}')
print(f'... with CDIPS light curves: {len(lccut_df)}')
print(f'NGC candidate member total with Rp<16: {len(ngc_df[ngc_df.phot_rp_mean_mag<16])}')
print(42*'-')
