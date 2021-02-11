"""
Given _manually downloaded_ GALAH DR3 spectra (from
drivers.oneoffs.MANUAL_get_galah_dr3_spectra), measure the Li EWs.
"""

from earhart.paths import DATADIR, RESULTSDIR
import os
from glob import glob
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from cdips_followup.spectools import (
    read_galah_given_sobject_id, viz_1d_spectrum, get_Li_6708_EW
)

OUTDIR = os.path.join(RESULTSDIR, 'lithium', 'GALAH_DR3')
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)

#
# the sourcelist that was used as the manual input file, made via
# earhart.plotting.plot_galah_dr3_lithium
#
inpath = '/Users/luke/Dropbox/proj/earhart/results/lithium/kinematic_X_galah_dr3.csv'
df = pd.read_csv(inpath)

#
# get the available spectra
#
workingdir = os.path.join(DATADIR, 'galah_dr3_spectra', 'galah',
                          'dr3', 'spectra', 'hermes')

red_specpaths = glob(os.path.join(workingdir, '*3.fits'))

sobject_ids = np.array(
    [np.int64(os.path.basename(p).replace('3.fits','')) for p in red_specpaths]
)

_df = pd.DataFrame({'sobject_id': sobject_ids})

mdf = _df.merge(df, how='left', on='sobject_id')

if len(df) > len(mdf):
    print(f'WRN! Expected to get {len(df)} GALAH DR3 sources with spectra...')
    print(f'WRN! Found only {len(mdf)}...')

for _, r in mdf.iterrows():

    single_ccd = 3
    sobject_id = np.int64(r['sobject_id'])
    dr2_source_id = np.int64(r['source_id'])

    outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GALAHDR3_ccd{single_ccd}.png')
    ew_outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GALAHDR3_Li6708_EW.png')
    csv_outpath = ew_outpath.replace('.png','_results.csv')

    if not os.path.exists(csv_outpath):

        flx, wav = read_galah_given_sobject_id(
            sobject_id, workingdir, verbose=False, single_ccd=single_ccd
        )
        viz_1d_spectrum(flx, wav, outpath)

        specpath = glob(os.path.join(workingdir, f'{sobject_id}3.fits'))
        assert len(specpath) == 1
        specpath = specpath[0]

        LiI_a = 6707.76
        LiI_b = 6707.91
        Li_avg = (LiI_a + LiI_b)/2

        xshift = 6708.30 - Li_avg
        get_Li_6708_EW(specpath, xshift=xshift, outpath=ew_outpath,
                       writecsvresults=True)

    else:
        pass

#
# compare with Gaia ESO
#
result_csvs = glob(os.path.join(OUTDIR, 'gaiadr2_*_results.csv'))
sourceids, dfs = [], []
for csvpath in result_csvs:
    sourceid = np.int64(os.path.basename(csvpath).split('_')[1])
    _df = pd.read_csv(csvpath)
    sourceids.append(sourceid)
    dfs.append(_df)

li_df = pd.concat(dfs)
li_df['source_id'] = sourceids
FUDGE = 65
li_df['Fitted_Li_EW_mA_plus_fudge'] = li_df['Fitted_Li_EW_mA'] + FUDGE

galah_li_path = os.path.join(
    DATADIR, 'lithium', 'galahdr3_fullfaintkinematic_xmatch_20210211.csv'
)
li_df.to_csv(galah_li_path, index=False)
print(f'Saved {galah_li_path}')

datapath = os.path.join(DATADIR, 'lithium',
                        'randich_fullfaintkinematic_xmatch_20210128.csv')
if not os.path.exists(datapath):
    from earhart.lithium import _make_Randich18_xmatch
    _make_Randich18_xmatch(datapath, vs_rotators=0)
rdf = pd.read_csv(datapath)

mdf = li_df.merge(rdf, how='inner', on='source_id')

plt.close('all')
f,ax = plt.subplots(figsize=(4,3))
ax.scatter(mdf.EWLi, mdf.Fitted_Li_EW_mA)
ax.scatter(mdf.EWLi, mdf.Fitted_Li_EW_mA+FUDGE)
ax.plot(np.arange(-50,250,1), np.arange(-50,250,1), 'k--')
ax.set_xlabel('Randich+18 EW Li from Gaia-ESO [mA]')
ax.set_ylabel('My EW Li from GALAHDR3 [mA]')
outpath = '../results/lithium/validate_my_GALAH_EWs_vs_Randich18.png'
f.savefig(outpath, bbox_inches='tight', dpi=400)
plt.close('all')

