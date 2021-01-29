"""
Contents:
    get_GalahDR3_lithium
    get_Randich18_NGC2516
"""
import os
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.table import Table

from earhart.paths import DATADIR, RESULTSDIR

def get_GalahDR3_lithium(verbose=1, defaultflags=0):
    """
    Get astropy table of stellar parameter flag == 0 stars from GALAH DR3, with
    lithium detections or upper limits.
    """

    # downloaded via wget, per
    # https://github.com/svenbuder/GALAH_DR3/blob/master/tutorials/tutorial1_dr3_main_catalog_overview.ipynb
    dr3path = os.path.join(DATADIR, 'lithium', 'GALAH_DR3_main_allstar_v1.fits')

    dr3_tab = Table.read(dr3path)

    if defaultflags:
        # stellar parameter flag
        qual = np.array(dr3_tab['flag_sp'])
        binary_repr_vec = np.vectorize(np.binary_repr)
        qual_binary = binary_repr_vec(qual, width=11)

        # 11 total flag bits:
        # 2^0 = 1: Gaia DR2 RUWE > 1.4
        # 2^1 = 2: unreliable broadening
        # 2^2 = 4: low S/N
        # 2^3 = 8: reduction issues (wvlen soln, t-SNE reduction issues,
        #    weird fluxes, spikes, etc.
        # 4   = 16: t-SNE projected emission features
        # 5   = 32: t-SNE projected binaries
        # 6   = 64: on binary sequence / PMS sequence
        # 7   = 128: S/N dependent high SME chi2 (bad fit)
        # 8   = 256: problems with Fe: line flux not between 0.03 and 1.00,
        #      [Fe/H] unreliable, or blending suspects and SME didnt finish
        # 9   = 512: SME did not finish. Either a) No convergence -> nonfinite stellar
        #      parameters. Or b) Gaussian RV fit failed.
        # 2^10 = 1024: MARCS grid limit reached or outside reasonable
        #       parameter range.

        # Need reliable broadening, S/N, etc.
        # 2^1 = 2: unreliable broadening
        # 2^2 = 4: low S/N
        # 2^3 = 8: reduction issues (wvlen soln, t-SNE reduction issues,
        #    weird fluxes, spikes, etc.
        # 8   = 256: problems with Fe: line flux not between 0.03 and 1.00,
        #      [Fe/H] unreliable, or blending suspects and SME didnt finish
        # 9   = 512: SME did not finish. Either a) No convergence -> nonfinite stellar
        #      parameters. Or b) Gaussian RV fit failed.
        # 2^10 = 1024: MARCS grid limit reached or outside reasonable
        #       parameter range.
        badbits = [1,2,3]

        sel = np.isfinite(dr3_tab['source_id'])
        for bb in badbits:
            # zero -> one-based count here to convert bitwise flags to
            # python flags
            sel &= ~(np.array([q[bb-1] for q in qual_binary]).astype(bool))

    else:
        sel = np.isfinite(dr3_tab['source_id'])

    if verbose:
        print(f'All stars in GALAH DR3: {len(dr3_tab)}')
        if defaultflags:
            print(f'All stars in GALAH DR3 with bitflags {repr(badbits)} not set: {len(dr3_tab[sel])}')

    return dr3_tab[sel]

def get_Randich18_NGC2516():

    hl = fits.open(
        os.path.join(DATADIR, 'lithium',
                     'Randich_2018_NGC2516_all796_entries_vizier.fits')
    )

    t_df = Table(hl[1].data).to_pandas()

    return t_df
