import os, collections, pickle
import numpy as np, pandas as pd
from glob import glob

from astropy.io import fits

import cdips.utils.lcutils as lcu
import cdips.lcproc.detrend as dtr
import cdips.lcproc.mask_orbit_edges as moe

from earhart.paths import PHOTDIR, RESULTSDIR

def get_toi1937_lightcurve():
    """
    Create the stitched CDIPS FFI light curve for TOI 1937. (Starting from the
    raw light curves, and the PCA eigenvectors previously made for this
    sector). Note: the main execution of this PCA detrending happens on
    phtess2.

    A few notes:
        * 3 eigenvectors were used, plus the background light BGV timeseries.
        * a +/-12 hour orbit edge mask was used (to avoid what looked like
        scattered light)
        * the output can be checked at
        /results/quicklook_lcs/5489726768531119616_allvar_report.pdf
    """

    picklepath = os.path.join(
        PHOTDIR, 'toi1937_merged_stitched_s7s9_lc_20201130.pkl'
    )

    if not os.path.exists(picklepath):

        # Use the CDIPS IRM2 light curves as starting base.
        # 5489726768531119616_s09_llc.fits
        lcpaths = glob(os.path.join(PHOTDIR, '*_s??_llc.fits'))
        assert len(lcpaths) == 2

        infodicts = [
            {'SECTOR': 7, 'CAMERA': 3, 'CCD': 4, 'PROJID': 1527},
            {'SECTOR': 9, 'CAMERA': 3, 'CCD': 3, 'PROJID': 1558},
        ]

        ##########################################
        # next ~45 lines pinched from cdips.drivers.do_allvariable_report_making
        ##########################################
        #
        # detrend systematics. each light curve yields tuples of:
        #   primaryhdr, data, ap, dtrvecs, eigenvecs, smooth_eigenvecs
        #
        dtr_infos = []
        for lcpath, infodict in zip(lcpaths, infodicts):
            dtr_info = dtr.detrend_systematics(
                lcpath, infodict=infodict, max_n_comp=3
            )
            dtr_infos.append(dtr_info)

        #
        # stitch all available light curves
        #
        ap = dtr_infos[0][2]
        timelist = [d[1]['TMID_BJD'] for d in dtr_infos]
        maglist = [d[1][f'PCA{ap}'] for d in dtr_infos]
        magerrlist = [d[1][f'IRE{ap}'] for d in dtr_infos]

        extravecdict = {}
        extravecdict[f'IRM{ap}'] = [d[1][f'IRM{ap}'] for d in dtr_infos]
        for i in range(0,7):
            extravecdict[f'CBV{i}'] = [d[3][i, :] for d in dtr_infos]

        time, flux, fluxerr, vec_dict = lcu.stitch_light_curves(
            timelist, maglist, magerrlist, extravecdict
        )

        #
        # mask orbit edges
        #
        s_time, s_flux, inds = moe.mask_orbit_start_and_end(
            time, flux, raise_expectation_error=False, orbitgap=0.7,
            orbitpadding=12/24,
            return_inds=True
        )
        s_fluxerr = fluxerr[inds]

        #
        # save output
        #

        ap = dtr_infos[0][2]
        lcdict = {
            'source_id': np.int64(5489726768531119616),
            'E_BpmRp': 0.1343,
            'ap': ap,
            'TMID_BJD': time,
            f'IRM{ap}': vec_dict[f'IRM{ap}'],
            f'PCA{ap}': flux,
            f'IRE{ap}': fluxerr,
            'STIME': s_time.astype(np.float64),
            f'SPCA{ap}': s_flux.astype(np.float64),
            f'SPCAE{ap}': s_fluxerr.astype(np.float64),
            'dtr_infos': dtr_infos,
            'vec_dict': vec_dict,
            'tess_texp': np.nanmedian(np.diff(s_time))
        }

        with open(picklepath , 'wb') as f:
            pickle.dump(lcdict, f)

        #
        # verify output
        #
        from cdips.plotting.allvar_report import make_allvar_report
        plotdir = os.path.join(RESULTSDIR, 'quicklook_lcs')
        outd = make_allvar_report(lcdict, plotdir)

    with open(picklepath, 'rb') as f:
        print(f'Found {picklepath}: loading it!')
        lcdict = pickle.load(f)

    return (
        lcdict['STIME'].astype(np.float64) - 2457000,
        lcdict['SPCA2'].astype(np.float64),
        lcdict['SPCAE2'].astype(np.float64),
        lcdict['tess_texp']
    )
