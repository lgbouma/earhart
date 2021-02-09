"""
Given _manually downloaded_ GALAH DR3 spectra (from
drivers.oneoffs.MANUAL_get_galah_dr3_spectra), measure the Li EWs.
"""

from earhart.paths import DATADIR, RESULTSDIR
import os
from glob import glob
import pandas as pd, numpy as np
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
workingdir = os.path.join(DATADIR, 'galah_dr3_spectra', 'EXAMPLE', 'galah',
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

    sobject_id = np.int64(r['sobject_id'])
    dr2_source_id = np.int64(r['source_id'])

    single_ccd = 3
    flx, wav = read_galah_given_sobject_id(
        sobject_id, workingdir, verbose=False, single_ccd=single_ccd
    )
    outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GALAHDR3_ccd{single_ccd}.png')
    viz_1d_spectrum(flx, wav, outpath)

    specpath = glob(os.path.join(workingdir, f'{sobject_id}3.fits'))
    assert len(specpath) == 1
    specpath = specpath[0]


    outpath = os.path.join(OUTDIR,
                           f'gaiadr2_{dr2_source_id}_GALAHDR3_Li6708_EW.png')
    get_Li_6708_EW(specpath, outpath=outpath)
