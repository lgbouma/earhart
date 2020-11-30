import os, collections
import numpy as np, pandas as pd
from glob import glob

from astropy.io import fits

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe


PHOTDIR = '/Users/luke/Dropbox/proj/earhart/data/photometry'

def get_toi1937_lightcurve():

    # Use the CDIPS IRM2 light curves as starting base.
    # 5489726768531119616_s09_llc.fits
    lcpaths = glob(os.path.join(PHOTDIR, '*_s??_llc.fits'))

    ##########################################
    # next ~45 lines pinched from cdips.allvariable_report_making
    ##########################################
    #
    # detrend systematics. each light curve yields tuples of:
    #   primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
    #
    dtr_infos = []
    for lcpath in lcpaths:
        dtr_info = dtr.detrend_systematics(lcpath)
        dtr_infos.append(dtr_info)

    #
    # stitch all available light curves
    #
    import IPython; IPython.embed()
    ap = dtr_infos[0][2]
    timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
    maglist = [d[1][f'IRM{ap}'] for d in dtr_infos]
    magerrlist = [d[1][f'IRE{ap}'] for d in dtr_infos]

    extravecdict = {}
    extravecdict[f'IRM{ap}'] = [d[1][f'IRM{ap}'] for d in dtr_infos]
    for i in range(0,7):
        extravecdict[f'CBV{i}'] = [d[3][i, :] for d in dtr_infos]

    try:
        time, flux, fluxerr, vec_dict = lcu.stitch_light_curves(
            timelist, maglist, magerrlist, extravecdict
        )
    except ValueError:
        lc_info = {'n_sectors': len(lcpaths), 'lcpaths': lcpaths,
                   'detrending_completed': False}
        ppu.save_status(statuspath, 'lc_info', lc_info)
        return 0


    #
    # mask orbit edges
    #
    s_time, s_flux, inds = moe.mask_orbit_start_and_end(
        time, flux, raise_expectation_error=False, orbitgap=0.7,
        return_inds=True
    )
    s_fluxerr = fluxerr[inds]



    #FIXME merger???

    if not os.path.exists(lcfile):
        ticid = simbad_to_tic('WASP 4')
        lcfiles = get_two_minute_spoc_lightcurves(ticid, download_dir=TESTDATADIR)
        lcfile = lcfiles[0]

    hdul = fits.open(lcfile)
    d = hdul[1].data

    yval = 'PDCSAP_FLUX'
    time = d['TIME']
    _f, _f_err = d[yval], d[yval+'_ERR']
    flux = _f/np.nanmedian(_f)
    flux_err = _f_err/np.nanmedian(_f)
    qual = d['QUALITY']

    sel = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    tess_texp = np.nanmedian(np.diff(time[sel]))

    return (
        time[sel].astype(np.float64),
        flux[sel].astype(np.float64),
        flux_err[sel].astype(np.float64),
        tess_texp
    )
