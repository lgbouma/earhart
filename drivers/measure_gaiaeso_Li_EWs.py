"""
Given _manually downloaded_ GAIA ESO DR4 spectra (from
get_gaia_eso_dr4_spectra.py -> running wget), measure the Li EWs.
"""

from earhart.paths import DATADIR, RESULTSDIR
import os
from glob import glob
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
from cdips_followup.spectools import (
    read_gaiaeso, viz_1d_spectrum, get_Li_6708_EW
)

OUTDIR = os.path.join(RESULTSDIR, 'lithium', 'GAIAESO_DR4')
if not os.path.exists(OUTDIR):
    os.mkdir(OUTDIR)

#
# the sourcelist that was used as the manual input file
#
inpath = "/Users/luke/Dropbox/proj/earhart/data/gaiaeso_dr4_spectra/gaiaeso_dr4_meta_X_fullfaint_kinematic.csv"
df = pd.read_csv(inpath)

SPECDIR = "/Users/luke/Dropbox/proj/earhart/data/gaiaeso_dr4_spectra/spectra"

for _, r in df.iterrows():

    dr2_source_id = np.int64(r['source_id'])
    dp_id = str(r['dp_id'])
    specpath = os.path.join(SPECDIR, dp_id)

    outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GAIAESODR4.png')
    ew_outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GAIAESODR4_Li6708_EW.png')
    csv_outpath = ew_outpath.replace('.png','_results.csv')

    if not os.path.exists(csv_outpath):

        flx, wav = read_gaiaeso(specpath)
        viz_1d_spectrum(flx, wav, outpath)

        # observed frame -> rest frame conversion. ~24km/s cluster
        # mean velocity, shifts by ~0.5A
        LiI_a = 6707.76
        LiI_b = 6707.91
        Li_avg = (LiI_a + LiI_b)/2
        xshift = 6708.30 - Li_avg

        try:
            get_Li_6708_EW(specpath, xshift=xshift, outpath=ew_outpath,
                           writecsvresults=True)
        except Exception as e:
            print(f'ERROR! {e}')

    else:
        print(f'found {csv_outpath}')
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

li_path = os.path.join(
    DATADIR, 'lithium', 'gaiaesodr4_fullfaintkinematic_xmatch_20210421.csv'
)
li_df.to_csv(li_path, index=False)
print(f'Saved {li_path}')

datapath = os.path.join(DATADIR, 'lithium',
                        'randich_fullfaintkinematic_xmatch_20210310.csv')
if not os.path.exists(datapath):
    from earhart.lithium import _make_Randich18_xmatch
    _make_Randich18_xmatch(datapath, vs_rotators=0)
rdf = pd.read_csv(datapath)

mdf = li_df.merge(rdf, how='inner', on='source_id')
mdf = mdf.drop_duplicates(subset='source_id', keep='first')
mdf = mdf[(~pd.isnull(mdf.EWLi))]
print(f'Got {len(mdf)} calibration stars')

plt.close('all')
f,ax = plt.subplots(figsize=(4,4))
ax.scatter(mdf.EWLi, mdf.Fitted_Li_EW_mA, c='k', s=2, zorder=1)
sel = (mdf.Teff < 4400)
sdf = mdf[sel]
ax.scatter(sdf.EWLi, sdf.Fitted_Li_EW_mA, c='r', s=2, label='Teff<4400K',
           zorder=2)
ax.plot(np.arange(-50,250,1), np.arange(-50,250,1), 'k--')
ax.legend()
ax.set_xlabel('Randich+18 EW Li from Gaia-ESO [mA]')
ax.set_ylabel('My EW Li from GAIAESODR4 [mA]')
outpath = '../results/lithium/validate_my_GAIAESODR4_EWs_vs_Randich18.png'
f.savefig(outpath, bbox_inches='tight', dpi=400)
print(f'Made {outpath}')
plt.close('all')

print(42*'-')
print('Check the following to ensure you understand differences...')
smdf = mdf.sort_values(by='Fitted_Li_EW_mA')[['Fitted_Li_EW_mA','EWLi','source_id','Teff']]
print(smdf.head(n=20))

import IPython; IPython.embed()
