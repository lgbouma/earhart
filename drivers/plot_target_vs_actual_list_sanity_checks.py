"""
The 3298 candidate cluster members are split into the "core" and "halo" samples
(CG18, KC19+M21 respectively). Source ID's are from Gaia DR2.

How many had CDIPS light curves produced?

Which ones did not, and why?
"""

import os
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from earhart.helpers import get_gaia_basedata
from earhart.paths import RESULTSDIR, DATADIR
from aesthetic.plot import savefig, format_ax, set_style

set_style()

PLOTDIR = os.path.join(
    RESULTSDIR, 'target_vs_actual_list_sanity_checks'
)
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)


runid = "NGC_2516"
basedata = 'fullfaint'
nbhd_df, core_df, halo_df, full_df, trgt_df = get_gaia_basedata(basedata)

# made by drivers.cluster_rotation.get_auto_rotation_periods
rot_df = pd.read_csv(
    os.path.join(DATADIR, 'rotation', 'NGC_2516_rotation_periods.csv')
)
lccut_df = rot_df[rot_df.n_cdips_sector>0]
print(f'... with CDIPS light curves: {len(lccut_df)}')

#
# first plot: ra vs dec.
#
plt.close('all')
f,ax = plt.subplots(figsize=(4,3))
ax.scatter(
    full_df.ra, full_df.dec, c='C0', alpha=0.8, zorder=2,
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
    get_xval(full_df), get_yval(full_df), c='C0', alpha=0.8, zorder=2,
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
print(f'NGC candidate member total: {len(full_df)}')
print(f'... with CDIPS light curves: {len(lccut_df)}')
print(f'NGC candidate member total with Rp<16: {len(full_df[full_df.phot_rp_mean_mag<16])}')
sel = (
    (full_df.phot_rp_mean_mag<16)
    &
    (full_df.in_CG18 | full_df.in_KC19)
)
print(f'NGC candidate member from CG18 or KC19 with Rp<16: {len(full_df[sel])}')
print(42*'-')
print(lccut_df.phot_rp_mean_mag.describe())

#
# next plot: l vs b sky coverage
#
import matplotlib as mpl

plt.close('all')
f,ax = plt.subplots(figsize=(4,3))

cmap = mpl.cm.viridis
bounds = np.arange(1, 14, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cax = ax.scatter(
    lccut_df.l, lccut_df.b, c=lccut_df.n_cdips_sector, cmap=cmap,
    alpha=0.8, zorder=3, s=5, rasterized=True, linewidths=0,
    marker='.', norm=norm
)

cb = f.colorbar(cax)
cb.set_label("N CDIPS sectors", rotation=270)

ax.set_ylim((-25,-10))

ax.set_xlabel('l [deg]')
ax.set_ylabel('b [deg]')
outpath = os.path.join(PLOTDIR, f'l_vs_b_nsector_check.png')
savefig(f, outpath, dpi=400)


plt.close('all')
f,ax = plt.subplots(figsize=(4,3))

cmap = mpl.cm.viridis
bounds = np.arange(1, 14, 1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

cax = ax.scatter(
    lccut_df.ra, lccut_df.dec, c=lccut_df.n_cdips_sector, cmap=cmap,
    alpha=0.8, zorder=3, s=5, rasterized=True, linewidths=0,
    marker='.', norm=norm
)

cb = f.colorbar(cax)
cb.set_label("N CDIPS sectors", rotation=270)

ax.set_xlabel('ra [deg]')
ax.set_ylabel('dec [deg]')
outpath = os.path.join(PLOTDIR, f'ra_vs_dec_nsector_check.png')
savefig(f, outpath, dpi=400)
